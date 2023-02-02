import torch
import torch.nn as nn

import torch.utils.checkpoint

from graph import GraphTripleConv, GraphTripleConvNet
from layers import Interpolate, get_normalization_2d
from layout import masks_to_layout
from generators import tissue_image_generator, pix2pix_generator


import sys
sys.path.insert(0,'./hovernet')
from hovernet import InferManager


class SynClayModel(nn.Module):

    def __init__(self, vocab, object_embed_dim, image_size=(256, 256),
                 gconv_dim=8, gconv_hidden_dim=16,
                 gconv_pooling='avg', gconv_num_layers=3, mask_channels=3,
                 generator='residual',
                 include_safron=False,
                 include_channel_reducer_network=True,
                 integrate_hovernet=False,
                 hovernet_model_path='',
                 type_info_path='',
                 mode='train',
                 normalization='batch', activation='leakyrelu-0.2',
                 mask_size=None, mlp_normalization='none', embed_noise_dim=2,
                 **kwargs):
        super(Sg2ImModel, self).__init__()

        # We used to have some additional arguments:
        # vec_noise_dim, gconv_mode, box_anchor, decouple_obj_predictions
        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.vocab = vocab
        self.image_size = image_size
        self.embed_noise_dim = embed_noise_dim
        self.include_safron = include_safron
        self.mode = mode
        mask_size = mask_size[0]

        embedding_dim = object_embed_dim + embed_noise_dim #original object dim + noise dim

        if gconv_num_layers == 0:
            self.gconv = nn.Linear(embedding_dim, gconv_dim)
            self.gconv.cuda()
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim': embedding_dim,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'mlp_normalization': mlp_normalization,
                'noise_dim': embed_noise_dim
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
                'noise_dim': 0
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        self.mask_net = None
        if mask_size is not None and mask_size > 0:
            self.mask_net = self.nuclei_mask_net(embedding_dim, mask_size)
            self.mask_net.cuda()

        self.include_channel_reducer_network = include_channel_reducer_network

        next_input_channels = embedding_dim

        self.generator = generator
        if(self.generator == 'pix2pix'):
            self.image_generator = pix2pix_generator(in_channels=next_input_channels)
        elif(self.generator == 'residual'):
            self.context_encoder_block = self.build_context_encoder_block(next_input_channels)
            self.image_generator = tissue_image_generator(input_nc=next_input_channels,
                                                         output_nc=3,
                                                         n_blocks_global=5,
                                                         n_downsample_global=3,
                                                         ngf=64,
                                                         norm='instance')
        else:
            raise "Give valid generator name"

        self.context_encoder_block.cuda()
        self.image_generator.cuda()

        self.integrate_hovernet = integrate_hovernet
        if(integrate_hovernet):
            self.hovernet_mask_generator = InferManager(model_path=hovernet_model_path,
                                                        mode="fast",
                                                        type_info_path=type_info_path)


    def build_context_encoder_block(self, dim):
        layers = []
        layers.append(nn.Conv2d(dim, dim, kernel_size=33))
        layers.append(nn.BatchNorm2d(dim))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)


    def nuclei_mask_net(self, dim, mask_size):
        output_dim = 1
        layers, cur_size = [], 1
        while cur_size < mask_size:
            layers.append(Interpolate(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(get_normalization_2d(dim,"batch"))
            layers.append(nn.ReLU())
            cur_size *= 2
        if cur_size != mask_size:
            raise ValueError('Mask size must be a power of 2')
        layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
        mask_net = nn.Sequential(*layers)
        return mask_net


    def forward(self, mask, object_indices, object_coordinates, object_embeddings, triples, label_gt=None,
                objects_boxes_gt=None, object_bounding_boxes_constructed=None, objects_masks_gt=None,
                object_trimasks_gt=None):

        object_masks_pred = None
        mask_pred = None

        try:
            s, o, edge_weights = triples.chunk(3, dim=1)  # All have shape (T, 1)
        except:
            print("Error while triples chunk")
            return None,None,None,None,None,None

        s = s.long()
        o = o.long()

        s, o, edge_weights = [x.squeeze(1) for x in [s, o, edge_weights]]  # Now have shape (T,)

        num_objects = object_embeddings.size()[0]
        embedding_dim = object_embeddings.size()[1]

        # Uncomment if no affine transformation
        O = object_embeddings.size(0)
        layout_noise = torch.randn((O, self.embed_noise_dim), dtype=object_embeddings.dtype,
                                   device=object_embeddings.device)
        object_embeddings = torch.cat([object_embeddings, layout_noise], dim=1)

        object_embeddings = object_embeddings.cuda()
        object_masks_pred = None
        if self.mask_net is not None:
            mask_scores = self.mask_net(object_embeddings.view(num_objects, -1, 1, 1))
            object_masks_pred = mask_scores.squeeze(1).sigmoid()

        object_boxes = objects_boxes_gt
        if (object_boxes is None):
            object_boxes = object_bounding_boxes_constructed
            if(object_boxes is None):
                object_boxes_pred = self.box_net(object_embeddings)
                object_boxes = object_boxes_pred

        object_boxes = object_boxes.cuda()

        H, W = self.image_size

        mask_pred = masks_to_layout(object_embeddings, object_boxes, object_masks_pred, H, W, gpu_id=object_embeddings.get_device())

        mask_pred = mask_pred.cuda()

        image_pred = self.image_generator(mask_pred) #change it later

        if(self.integrate_hovernet):
            label_pred_hovernet_patches = self.hovernet_mask_generator.get_segmentation(image_pred.permute(0,2,3,1))
            if(label_gt is not None):
                label_gt=label_gt.squeeze()[:,:,None]
                label_gt_hovernet_patches = self.hovernet_mask_generator.tensor_to_patches(label_gt,centre_crop=164)
            else:
                label_gt_hovernet_patches = None
        else:
            label_pred_hovernet_patches = None
            label_gt_hovernet_patches = None

        return image_pred, mask_pred, label_pred_hovernet_patches, object_masks_pred, object_boxes, label_gt_hovernet_patches