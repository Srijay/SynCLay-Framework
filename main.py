import argparse
import math, csv
from collections import defaultdict
import json
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data.thecot_model.thecot_graph import TheCOTGraph
from data.pannuke import PanNukeDataset, pannuke_collate_fn
from data.conic import CoNiCDataset, conic_collate_fn
from model import SynClayModel
from discriminators import PatchDiscriminator, AcCropDiscriminator, Pix2PixDiscriminator
from generators import weights_init
from losses import get_gan_losses
from utils import *
import os

ROOT_DIR = os.path.expanduser('F:/Datasets/conic/CoNIC_Challenge/challenge')

parser = argparse.ArgumentParser()

parser.add_argument('--train_image_dir',
                    default=os.path.join(ROOT_DIR, 'train/images'))
parser.add_argument('--train_mask_dir',
                    default=os.path.join(ROOT_DIR, 'train/masks'))
parser.add_argument('--train_inst_label_dir',
                    default=os.path.join(ROOT_DIR, 'train/labels'))

parser.add_argument('--val_image_dir',
                    default=os.path.join(ROOT_DIR, 'valid/images'))
parser.add_argument('--val_mask_dir',
                    default=os.path.join(ROOT_DIR, 'valid/masks'))
parser.add_argument('--val_inst_label_dir',
                    default=os.path.join(ROOT_DIR, 'valid/labels'))

parser.add_argument('--test_image_dir', default=os.path.join(ROOT_DIR, 'valid/images'))
parser.add_argument('--test_mask_dir', default=os.path.join(ROOT_DIR, 'valid/masks'))
parser.add_argument('--test_inst_label_dir', default=os.path.join(ROOT_DIR, 'valid/labels'))

# Object vector parameters
parser.add_argument('--use_size_feature', default=0, type=int)
parser.add_argument('--use_loc_feature', default=1, type=int)

# Optimization hyperparameters
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_iterations', default=312000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=9000000, type=int)

# Dataset options
parser.add_argument('--image_size', default='256,256', type=int_tuple)

parser.add_argument('--num_train_samples', default=10, type=int)
parser.add_argument('--num_val_samples', default=10, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=0, type=int)


# Object Mask, Mask, Generic Image Generator options
parser.add_argument('--mask_size', default='64,64', type=int_tuple)
parser.add_argument('--embed_noise_dim', default=4, type=int)
parser.add_argument('--gconv_hidden_dim', default=8, type=int)
parser.add_argument('--gconv_dim', default=8, type=int) #8 for almost all experiments
parser.add_argument('--gconv_num_layers', default=3, type=int) #3 for almost all experiments , 6 for one
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)
parser.add_argument('--mask_channels', default=3, type=int)


# Image Generator options
parser.add_argument('--generator', default='residual') #pix2pix or residual
parser.add_argument('--include_channel_reducer_network', default=False) #Want to generate masks as well?
parser.add_argument('--l1_pixel_image_loss_weight', default=1.0, type=float) #1.0
parser.add_argument('--l2_mse_mask_loss_weight', default=1.0, type=float) #1.0
parser.add_argument('--hovernet_label_loss', default=1.0, type=float) #1.0

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=0.01, type=float) #0.01
parser.add_argument('--gan_loss_type', default='gan')

#Gland discriminator options
parser.add_argument('--crop_size', default=64, type=int)
parser.add_argument('--d_clip', default=None, type=float)
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='same')
parser.add_argument('--d_activation', default='leakyrelu-0.2')
parser.add_argument('--d_obj_arch', default='C3-16-2,C3-32-2,C3-64-2')
parser.add_argument('--d_obj_weight', default=1.0, type=float)  # 1.0 multiplied by d_loss_weight
parser.add_argument('--ac_loss_weight', default=0.1, type=float) #0.1

# Image discriminator
parser.add_argument('--discriminator', default='patchgan') #patchgan or standard
parser.add_argument('--d_img_arch', default='C3-64-2,C3-128-2,C3-256-2') #for standard discriminator
parser.add_argument('--d_img_weight', default=1.0, type=float)  # 1.0 multiplied by d_loss_weight

# Output options
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=1000, type=int)

parser.add_argument('--type_info_path', default='./hovernet/type_info/conic.json')
parser.add_argument('--hovernet_model_path', default='./hovernet/trained_models/conic.tar')
parser.add_argument('--output_dir', default='./outputs')

# Experiment related parameters
parser.add_argument('--experimentname', default='test')
parser.add_argument('--dataset', default='conic')
parser.add_argument('--integrate_hovernet', default=False, type=bool_flag)

parser.add_argument('--checkpoint_name', default='model.pt')
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)
parser.add_argument('--test_output_dir', default=os.path.join('./output'))

# combine with TheCoT
parser.add_argument('--cellular_layout_folder', default="./cellular_layouts")
parser.add_argument('--cells_size_distribution_file', default="./data/thecot_model/cells_size_distributions.obj")
parser.add_argument('--thecot_output_dir', default=os.path.join('./output'))
parser.add_argument('--draw_edges_in_graph', default=False, type=bool_flag)

# If you want to test model, set mode to test
# If want to generate thecot images, put mode to thecot
parser.add_argument('--mode', default='train', type=str)


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss


def build_dsets(args):

    if (args.mode == "train"):
        dset_kwargs = {
            'image_dir': args.train_image_dir,
            'mask_dir': args.train_mask_dir,
            'label_dir': args.train_inst_label_dir,
            'image_size': args.image_size,
            'object_mask_size': args.mask_size,
            'use_size_feature': args.use_size_feature,
            'use_loc_feature': args.use_loc_feature
        }

        if(args.dataset == "conic"):
            train_dset = CoNiCDataset(**dset_kwargs)
        else:
            train_dset = PanNukeDataset(**dset_kwargs)

        num_imgs = len(train_dset)
        print('Training dataset has %d images' % (num_imgs))

        vocab = train_dset.vocab

        return vocab, train_dset
    else:
        dset_kwargs = {
            'image_dir': args.test_image_dir,
            'mask_dir': args.test_mask_dir,
            'inst_label_dir': args.test_inst_label_dir,
            'image_size': args.image_size,
            'object_mask_size': args.mask_size,
            'use_size_feature': args.use_size_feature,
            'use_loc_feature': args.use_loc_feature
        }
        test_dset = CoNiCDataset(**dset_kwargs)

        num_imgs = len(test_dset)
        vocab = test_dset.vocab
        print('Testing dataset has %d images' % (num_imgs))

        return vocab, test_dset


def build_loaders(args):

    if (args.mode == "train"):
        vocab, train_dset = build_dsets(args)
        if (args.dataset == "conic"):
            collate_fn = conic_collate_fn
        else:
            collate_fn = pannuke_collate_fn

        loader_kwargs = {
            'batch_size': args.batch_size,
            'num_workers': args.loader_num_workers,
            'shuffle': True,
            'collate_fn': collate_fn,
        }

        train_loader = DataLoader(train_dset, **loader_kwargs)

        return vocab, train_dset.embedding_dim, train_loader
    else:
        vocab, test_dset, _ = build_dsets(args)
        if (args.dataset == "conic"):
            collate_fn = conic_collate_fn
        else:
            collate_fn = pannuke_collate_fn
        loader_kwargs = {
            'batch_size': args.batch_size,
            'num_workers': args.loader_num_workers,
            'shuffle': False,
            'collate_fn': collate_fn,
        }
        test_loader = DataLoader(test_dset, **loader_kwargs)
        return vocab, test_dset.embedding_dim, test_loader


def build_model(args, vocab, object_embed_dim):
    kwargs = {
        'vocab': vocab,
        'image_size': args.image_size,
        'gconv_dim': args.gconv_dim,
        'gconv_hidden_dim': args.gconv_hidden_dim,
        'gconv_num_layers': args.gconv_num_layers,
        'mlp_normalization': args.mlp_normalization,
        'normalization': args.normalization,
        'activation': args.activation,
        'mask_channels': args.mask_channels,
        'generator': args.generator,
        'include_channel_reducer_network': args.include_channel_reducer_network,
        'integrate_hovernet':args.integrate_hovernet,
        'hovernet_model_path':args.hovernet_model_path,
        'type_info_path':args.type_info_path,
        'mask_size': args.mask_size,
        'object_embed_dim': object_embed_dim,
        'embed_noise_dim': args.embed_noise_dim,
        'mode': args.mode
    }
    model = SynClayModel(**kwargs)
    return model, kwargs


def build_img_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_img_weight = args.d_img_weight
  if d_weight == 0 or d_img_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'arch': args.d_img_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
  }

  if(args.discriminator == 'patchgan'):
      discriminator = Pix2PixDiscriminator(in_channels=3)
  elif(args.discriminator == 'standard'):
      discriminator = PatchDiscriminator(**d_kwargs)
  else:
      raise "Give proper name of discriminator"

  discriminator = discriminator.apply(weights_init)

  return discriminator, d_kwargs


def build_obj_discriminator(args, vocab):
    discriminator = None
    d_kwargs = {}
    d_weight = args.discriminator_loss_weight
    d_obj_weight = args.d_obj_weight
    if d_weight == 0 or d_obj_weight == 0:
        return discriminator, d_kwargs
    d_kwargs = {
        'vocab': vocab,
        'arch': args.d_obj_arch,
        'normalization': args.d_normalization,
        'activation': args.d_activation,
        'padding': args.d_padding,
        'object_size': args.crop_size,
    }
    discriminator = AcCropDiscriminator(**d_kwargs)
    return discriminator, d_kwargs


def check_model(args, t, loader, model, mode):

    experiment_output_dir = os.path.join(args.output_dir,args.experimentname)

    if torch.cuda.is_available():
        float_dtype = torch.cuda.FloatTensor
        long_dtype = torch.cuda.LongTensor
    else:
        float_dtype = torch.FloatTensor
        long_dtype = torch.LongTensor

    num_samples = 0

    output_dir = os.path.join(experiment_output_dir, "training_output", mode)
    mkdir(output_dir)

    with torch.no_grad():
        for batch in loader:

            if len(batch) == 11:
                image_name, image_gt, mask_gt, label_gt, object_indices, object_coordinates, object_bounding_boxes_gt, object_bounding_boxes_constructed, object_embeddings, triples, class_vectors = batch
            elif len(batch) == 12:
                image_name, image_gt, mask_gt, label_gt, object_indices, object_coordinates, object_bounding_boxes_gt, object_bounding_boxes_constructed, object_masks_gt, object_embeddings, triples, class_vectors = batch
            else:
                assert False

            if (len(image_gt) == 0):
                continue

            if torch.cuda.is_available():
                image_gt = image_gt.cuda()
                mask_gt = mask_gt.cuda()
                label_gt = label_gt.cuda()
                object_bounding_boxes_gt = object_bounding_boxes_gt.cuda()
                object_bounding_boxes_constructed = object_bounding_boxes_constructed.cuda()
                object_masks_gt = object_masks_gt.cuda()
                triples = triples.cuda()
                object_embeddings = object_embeddings.cuda()

            # Run the model as it has been run during training

            try:
                model_out = model(mask=mask_gt,
                                  object_indices=object_indices,
                                  object_embeddings=object_embeddings,
                                  object_coordinates=object_coordinates,
                                  triples=triples,
                                  objects_boxes_gt=object_bounding_boxes_gt,
                                  object_bounding_boxes_constructed=object_bounding_boxes_constructed,
                                  objects_masks_gt=object_masks_gt,
                                  label_gt=label_gt)
            except Exception as e:
                print(e)
                continue

            image_pred, mask_pred, label_pred_hovernet_patches, object_masks_pred, object_bounding_boxes_preds, label_gt_hovernet_patches = model_out

            num_samples += image_gt.size(0)
            if num_samples >= 10:
                break

            im_initial = image_name.split(".")[0]

            if (image_pred is not None):

                image_gt_path = os.path.join(output_dir, im_initial + "_gt_image.png")
                save_image(image_gt, image_gt_path)

                image_pred_path = os.path.join(output_dir, im_initial + "_pred_image.png")
                save_image(image_pred, image_pred_path)

                mask_gt_path = os.path.join(output_dir, im_initial + "_gt_mask.png")
                save_image(mask_gt, mask_gt_path)

                # Save the hovernet predicted mask
                if(args.integrate_hovernet):
                    hovernet_pred_mask_output_dir = os.path.join(output_dir, im_initial + "_hovernet_pred_mask.png")
                    label_pred_hovernet_class_output = np.argmax(label_pred_hovernet_patches.cpu().numpy(), axis=1)
                    label_pred = hovernet_class_output_to_class_image(label_pred_hovernet_class_output, args.image_size[0])
                    mask_pred_img = colored_images_from_classes(args, label_pred)
                    save_numpy_image_FLOAT(mask_pred_img, hovernet_pred_mask_output_dir)


def test_model(args, loader, model):

    if torch.cuda.is_available():
        float_dtype = torch.cuda.FloatTensor
        long_dtype = torch.cuda.LongTensor
    else:
        float_dtype = torch.FloatTensor
        long_dtype = torch.LongTensor

    test_output_dir = os.path.join(args.output_dir,args.test_output_dir)
    gt_image_output_dir = os.path.join(test_output_dir,"gt_image")
    pred_image_output_dir = os.path.join(test_output_dir,"pred_image")
    gt_mask_output_dir = os.path.join(test_output_dir,"gt_mask")
    pred_mask_output_dir = os.path.join(test_output_dir,"pred_mask")
    pred_labels_output_dir = os.path.join(test_output_dir,"pred_labels")
    pred_hovernet_mask_output_dir = os.path.join(test_output_dir,"pred_hovernet_mask")

    mkdir(gt_image_output_dir)
    mkdir(pred_image_output_dir)
    mkdir(gt_mask_output_dir)
    mkdir(pred_mask_output_dir)
    mkdir(pred_labels_output_dir)
    mkdir(pred_hovernet_mask_output_dir)

    t = 1

    with torch.no_grad():

        for batch in loader:

            if len(batch) == 10:
                image_name, image_gt, mask_gt, label_gt, object_indices, object_coordinates, object_bounding_boxes_gt, object_bounding_boxes_constructed, object_embeddings, triples, class_vectors = batch
            elif len(batch) == 12:
                image_name, image_gt, mask_gt, label_gt, object_indices, object_coordinates, object_bounding_boxes_gt, object_bounding_boxes_constructed, object_masks_gt, object_embeddings, triples, class_vectors = batch
            else:
                assert False

            if (len(image_gt) == 0):
                continue

            if torch.cuda.is_available():
                image_gt = image_gt.cuda()
                mask_gt = mask_gt.cuda()
                label_gt = label_gt.cuda()
                object_bounding_boxes_gt = object_bounding_boxes_gt.cuda()
                object_bounding_boxes_constructed = object_bounding_boxes_constructed.cuda()
                object_masks_gt = object_masks_gt.cuda()
                triples = triples.cuda()
                object_embeddings = object_embeddings.cuda()


            model_out = model(mask=mask_gt,
                          object_indices=object_indices,
                          object_embeddings=object_embeddings,
                          object_coordinates=object_coordinates,
                          triples=triples,
                          label_gt=label_gt,
                          objects_boxes_gt=object_bounding_boxes_gt,
                          object_bounding_boxes_constructed=object_bounding_boxes_constructed,
                          objects_masks_gt=object_masks_gt)

            image_pred, mask_pred, label_pred_hovernet_patches, object_masks_pred, object_boxes, label_gt_hovernet_patches = model_out

            if (image_pred is None):
                continue

            if (image_gt is None):
                continue


            #Save the ground truth image
            image_gt_path = os.path.join(gt_image_output_dir, image_name)
            save_image(image_gt, image_gt_path)

            #Save the predicted image
            image_pred_path = os.path.join(pred_image_output_dir, image_name)
            save_image(image_pred, image_pred_path)

            mask_gt_path = os.path.join(gt_mask_output_dir, image_name)
            save_image(mask_gt, mask_gt_path)


            # Save the predicted hovernet mask
            if (args.integrate_hovernet):
                label_pred_hovernet_class_output = np.argmax(label_pred_hovernet_patches.cpu().numpy(), axis=1)
                class_pred = hovernet_class_output_to_class_image(label_pred_hovernet_class_output, args.image_size[0])
                np.save(os.path.join(pred_labels_output_dir, image_name), class_pred)
                mask_pred_img = colored_images_from_classes(args, class_pred)
                save_numpy_image_FLOAT(mask_pred_img, os.path.join(pred_hovernet_mask_output_dir, image_name))

            t += 1


def generate_thecot_images(args, thecot_graph, model):

    mkdir(args.thecot_output_dir)
    pred_image_output_dir = os.path.join(args.thecot_output_dir, "pred_image")
    pred_mask_output_dir = os.path.join(args.thecot_output_dir, "pred_hovernet_mask")
    pred_label_output_dir = os.path.join(args.thecot_output_dir,"pred_label")
    shrinked_cellular_layout_dir = os.path.join(args.thecot_output_dir,"shrinked_cellular_layouts")
    cell_counts_file = os.path.join(args.thecot_output_dir,"cell_counts.csv")
    cell_counts_file = open(cell_counts_file, 'w', encoding='UTF8')
    cell_counts_writer = csv.writer(cell_counts_file)
    cell_counts_header = ['image_name', 'neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connectivetissue']
    cell_counts_writer.writerow(cell_counts_header)

    if(args.draw_edges_in_graph):
        graph_output_dir = os.path.join(args.thecot_output_dir, "graph")
    else:
        graph_output_dir = os.path.join(args.thecot_output_dir, "layout")

    mkdir(pred_image_output_dir)
    mkdir(pred_mask_output_dir)
    mkdir(pred_label_output_dir)
    mkdir(graph_output_dir)
    mkdir(shrinked_cellular_layout_dir)

    for matlab_cellular_layout_file in os.listdir(args.cellular_layout_folder):
        image_id = matlab_cellular_layout_file.split(".")[0]
        imname = image_id+".png"
        im_label_name = image_id+".npy"

        object_embeddings, object_bounding_boxes, edge_triplets, count_dict = thecot_graph.sample_graph(draw=True,
                                                                                                        draw_edges=args.draw_edges_in_graph,
                                                                                                        matlab_cellular_layout_file=os.path.join(args.cellular_layout_folder, matlab_cellular_layout_file),
                                                                                                        shrinked_cellular_layout_file=os.path.join(shrinked_cellular_layout_dir, im_label_name),
                                                                                                        output_path=os.path.join(graph_output_dir, imname))

        cell_counts = [count_dict[x] for x in cell_counts_header[1:]]
        cell_counts = [image_id] + cell_counts
        cell_counts_writer.writerow(cell_counts)

        if torch.cuda.is_available():

            object_bounding_boxes_gt = object_bounding_boxes.cuda()
            triples = edge_triplets.cuda()
            object_embeddings = object_embeddings.cuda()

            with torch.no_grad():
                # Run the model as it has been run during training
                model_out = model(mask=object_embeddings, #just a placeholder
                                  object_indices=[],
                                  object_embeddings=object_embeddings,
                                  object_coordinates=[],
                                  triples=triples,
                                  label_gt=None,
                                  objects_boxes_gt=object_bounding_boxes_gt,
                                  object_bounding_boxes_constructed=None,
                                  objects_masks_gt=None)

                image_pred, mask_pred, label_pred_hovernet_patches, object_masks_pred, object_bounding_boxes_preds, label_gt_hovernet_patches = model_out

                # Save the predicted image
                image_pred_path = os.path.join(pred_image_output_dir, imname)
                save_image(image_pred, image_pred_path)

                if(args.integrate_hovernet):
                    label_pred_hovernet_class_output = np.argmax(label_pred_hovernet_patches.cpu().numpy(), axis=1)
                    label_pred = hovernet_class_output_to_class_image(label_pred_hovernet_class_output, args.image_size[0])
                    np.save(os.path.join(pred_label_output_dir, im_label_name), label_pred)
                    mask_pred_img = colored_images_from_classes(args, label_pred)
                    save_numpy_image_FLOAT(mask_pred_img, os.path.join(pred_mask_output_dir, imname))

    cell_counts_file.close()


def calculate_model_losses(args, object_masks_gt, object_masks_pred, mask_gt, mask_pred, image_gt, image_pred,
                label_gt_hovernet_patches,label_pred_hovernet_patches,object_bounding_boxes_gt, object_bounding_boxes_preds):
    total_loss = torch.zeros(1).to(mask_gt)
    losses = {}

    #Mask mse Loss
    if(mask_gt.shape[1] == mask_pred.shape[1]):
        l2_pixel_loss_masks = F.mse_loss(mask_gt.float(), mask_pred)
        total_loss = add_loss(total_loss, l2_pixel_loss_masks, losses, 'L2_mse_loss_mask', args.l2_mse_mask_loss_weight)

    #Object mse Loss
    l2_pixel_loss_object_masks = F.mse_loss(object_masks_pred,object_masks_gt.float())
    total_loss = add_loss(total_loss, l2_pixel_loss_object_masks, losses, 'L2_mse_object_loss',
                          args.l2_mse_mask_loss_weight)

    #Image L1 Loss
    l1_pixel_loss_images = F.l1_loss(image_pred,image_gt.float())
    total_loss = add_loss(total_loss, l1_pixel_loss_images, losses, 'L1_pixel_loss_images',
                          args.l1_pixel_image_loss_weight)

    #Label Loss
    if (args.integrate_hovernet):
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss()
        label_loss = loss(logsoftmax(label_pred_hovernet_patches),label_gt_hovernet_patches)
        total_loss = add_loss(total_loss, label_loss, losses, 'hovernet_label_loss',
                              args.hovernet_label_loss)

    return total_loss, losses


def colored_images_from_classes(args, x):
    color_dict = json.load(open(args.type_info_path, "r"))
    color_dict = {
        int(k): list(v[1]) for k, v in color_dict.items()
        }
    image_size = args.image_size[0]
    k = np.zeros((image_size, image_size, 3))
    for i in range(0, image_size):
        for j in range(0, image_size):
            k[i][j] = color_dict[x[i][j]]
    return k/255.0


def hovernet_class_output_to_class_image(x, image_size):
    h = np.concatenate((x[0], x[2]), axis=1)
    v = np.concatenate((x[1], x[3]), axis=1)
    img = np.vstack((h, v))
    img = np.squeeze(img[: image_size, : image_size])  # crop back to original shape
    return img


def main(args):

    torch.cuda.empty_cache()

    experiment_output_dir = os.path.join(args.output_dir,args.experimentname)
    model_dir = os.path.join(experiment_output_dir, "model")

    if torch.cuda.is_available():
        float_dtype = torch.cuda.FloatTensor
    else:
        float_dtype = torch.FloatTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.mode == "train"):

        mkdir(experiment_output_dir)
        mkdir(model_dir)

        with open(os.path.join(experiment_output_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        vocab, object_embed_dim, train_loader = build_loaders(args)

    elif (args.mode == "thecot"):
        thecot_graph = TheCOTGraph(cellular_layout_folder=args.cellular_layout_folder,
                                  cells_size_distribution_file=args.cells_size_distribution_file,
                                  image_size=args.image_size[0],
                                  use_loc_feature=args.use_loc_feature)
        vocab, object_embed_dim, _ = build_loaders(args)

    else:
        vocab, object_embed_dim, test_loader = build_loaders(args)

    model, model_kwargs = build_model(args, vocab, object_embed_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    #Nuclei Discriminator
    obj_discriminator, d_obj_kwargs = build_obj_discriminator(args, vocab)
    if obj_discriminator is not None:
        obj_discriminator.cuda()
        obj_discriminator.type(float_dtype)
        obj_discriminator.train()
        optimizer_d_obj = torch.optim.Adam(obj_discriminator.parameters(),
                                           lr=args.learning_rate)

    # Image Discriminator
    image_discriminator, d_img_kwargs = build_img_discriminator(args, vocab)
    if image_discriminator is not None:
        image_discriminator.cuda()
        image_discriminator.type(float_dtype)
        image_discriminator.train()
        optimizer_d_image = torch.optim.Adam(image_discriminator.parameters(),
                                             lr=args.learning_rate)


    gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)

    if args.restore_from_checkpoint or args.mode=="test" or args.mode=="random":

        print("Restoring")
        restore_path = args.checkpoint_name
        restore_path = os.path.join(model_dir, restore_path)

        if (device == "cpu"):
            checkpoint = torch.load(restore_path, map_location="cpu")
        else:
            checkpoint = torch.load(restore_path, map_location="cpu") #to avoid memory surge

        model.load_state_dict(checkpoint['model_state'])

        if(args.mode=="train"):
            # optimizer.load_state_dict(checkpoint['optim_state']) #strict argument is not supported here

            if obj_discriminator is not None:
                obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
                optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])
                obj_discriminator.cuda()
            

            if image_discriminator is not None:
                image_discriminator.load_state_dict(checkpoint['d_image_state'])
                optimizer_d_image.load_state_dict(checkpoint['d_image_optim_state'])
                image_discriminator.cuda()
            


        if (args.mode == "test"):
            model.eval()
            test_model(args, test_loader, model)
            print("Testing has been done and results are saved")
            return


        if (args.mode == "thecot"):

            model.eval()
            generate_thecot_images(args, thecot_graph, model)
            print("Images are generated")

            return 0

        t = 0
        if 0 <= args.eval_mode_after <= t:
            model.eval()
        else:
            model.train()

        epoch = checkpoint['counters']['epoch']

        print("Starting Epoch : ",epoch)

    else:

        starting_epoch = 0

        if (args.mode == "test"):
            raise Exception("Give proper restoring model path")

        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'vocab': vocab,
            'model_kwargs': model_kwargs,
            'losses_ts': [],
            'losses': defaultdict(list),
            'd_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [],
            'train_samples': [],
            'train_iou': [],
            'val_batch_data': [],
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'model_state': None, 'model_best_state': None, 'optim_state': None,
            'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
            'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
            'd_mask_state': None, 'best_t': [],
        }

    #Loss Curves
    training_loss_out_dir = os.path.join(experiment_output_dir, 'training_loss_graph')
    mkdir(training_loss_out_dir)

    def draw_curve(epoch_list, loss_list, loss_name):
        plt.clf()
        plt.plot(epoch_list, loss_list, 'bo-', label=loss_name)
        plt.legend()
        plt.savefig(os.path.join(training_loss_out_dir,loss_name+'.png'))

    epoch_list = []
    monitor_epoch_losses = defaultdict(list)

    while True:

        if t >= args.num_iterations:
            break

        for batch in train_loader:

            if t == args.eval_mode_after:
                print('switching to eval mode')
                model.eval()
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

            if len(batch) == 11:
                image_name, image_gt, mask_gt, label_gt, object_indices, object_coordinates, object_bounding_boxes_gt, object_bounding_boxes_constructed, object_embeddings, triples, class_vectors = batch
            elif len(batch) == 12:
                image_name, image_gt, mask_gt, label_gt, object_indices, object_coordinates, object_bounding_boxes_gt, object_bounding_boxes_constructed, object_masks_gt, object_embeddings, triples, class_vectors = batch
            else:
                print(len(batch))
                assert False

            if (len(image_gt)==0):
                continue

            if torch.cuda.is_available():
                image_gt = image_gt.cuda()
                mask_gt = mask_gt.cuda()
                label_gt = label_gt.cuda()
                object_bounding_boxes_gt = object_bounding_boxes_gt.cuda()
                object_bounding_boxes_constructed = object_bounding_boxes_constructed.cuda()
                object_masks_gt = object_masks_gt.cuda()
                triples = triples.cuda()
                object_embeddings = object_embeddings.cuda()
                class_vectors = class_vectors.cuda()


            with timeit('forward', args.timing):
                # try:
                model_out = model(mask=mask_gt,
                                  object_indices=object_indices,
                                  object_embeddings=object_embeddings,
                                  object_coordinates=object_coordinates,
                                  triples=triples,
                                  objects_boxes_gt=object_bounding_boxes_gt,
                                  object_bounding_boxes_constructed=object_bounding_boxes_constructed,
                                  objects_masks_gt=object_masks_gt,
                                  label_gt=label_gt
                                  )
                
                image_pred, mask_pred, label_pred_hovernet_patches, object_masks_pred, object_bounding_boxes_preds, label_gt_hovernet_patches = model_out

            if (image_pred is None):
                continue

            image_pred = image_pred.cuda()
            

            total_loss, losses = calculate_model_losses(
                args, object_masks_gt, object_masks_pred, mask_gt, mask_pred, image_gt, image_pred,
                label_gt_hovernet_patches, label_pred_hovernet_patches, object_bounding_boxes_gt,
                object_bounding_boxes_preds)


            if obj_discriminator is not None:  # Object Images
                scores_fake, ac_loss = obj_discriminator(image_pred, object_bounding_boxes_gt, class_vectors)
                ac_loss = ac_loss.cuda()
                total_loss = add_loss(total_loss, ac_loss, losses, 'ac_loss', args.ac_loss_weight)
                weight = args.discriminator_loss_weight * args.d_obj_weight
                total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                                      'g_gan_obj_loss', weight)

            if image_discriminator is not None:
                scores_image_fake = image_discriminator(mask_gt.float(), image_pred)
                # scores_image_fake = scores_image_fake.cuda()
                weight = args.discriminator_loss_weight * args.d_img_weight
                total_loss = add_loss(total_loss, gan_g_loss(scores_image_fake), losses,
                                      'g_gan_image_loss', weight)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            with timeit('backward', args.timing):
                try:
                    total_loss.backward()
                except Exception as e:
                    # print(e)
                    print("Memory OOM : Iter number ",t, " image name ",image_name)
                    # torch.cuda.empty_cache()
                    continue
            optimizer.step()

            image_fake = image_pred.detach()
            image_real = image_gt.detach()

            if obj_discriminator is not None:

                d_obj_losses = LossManager()  # For object masks
                scores_fake, ac_loss_fake = obj_discriminator(image_fake, object_bounding_boxes_gt, class_vectors)
                scores_real, ac_loss_real = obj_discriminator(image_real.float(), object_bounding_boxes_gt, class_vectors)

                d_obj_gan_loss = gan_d_loss(scores_real, scores_fake)
                d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')

                if args.ac_loss_weight:
                    d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
                    d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

                optimizer_d_obj.zero_grad()
                d_obj_losses.total_loss.backward()
                optimizer_d_obj.step()
                obj_discriminator.cuda()


            if image_discriminator is not None:
                d_image_losses = LossManager()  # For image
                scores_fake = image_discriminator(mask_gt.float(), image_fake)
                scores_real = image_discriminator(mask_gt.float(), image_real.float())
                d_image_gan_loss = gan_d_loss(scores_real, scores_fake)
                d_image_losses.add_loss(d_image_gan_loss, 'd_image_gan_loss')
                optimizer_d_image.zero_grad()
                d_image_losses.total_loss.backward()
                optimizer_d_image.step()
                image_discriminator.cuda()

            t += 1


            if t % args.print_every == 0:

                print('t = %d / %d' % (t, args.num_iterations))
                for name, val in losses.items():
                    print(' G [%s]: %.4f' % (name, val))

                if obj_discriminator is not None:
                    for name, val in d_obj_losses.items():
                        print(' D_obj [%s]: %.4f' % (name, val))

                if image_discriminator is not None:
                    for name, val in d_image_losses.items():
                        print(' D_img [%s]: %.4f' % (name, val))

            if t % args.checkpoint_every == 0:

                print('checking on train')
                check_model(args, t, train_loader, model, "train")

                checkpoint['model_state'] = model.state_dict()

                if obj_discriminator is not None:
                    checkpoint['d_obj_state'] = obj_discriminator.state_dict()
                    checkpoint['d_obj_optim_state'] = optimizer_d_obj.state_dict()

                if image_discriminator is not None:
                    checkpoint['d_image_state'] = image_discriminator.state_dict()
                    checkpoint['d_image_optim_state'] = optimizer_d_image.state_dict()

                checkpoint['optim_state'] = optimizer.state_dict()
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint_path = os.path.join(model_dir, args.checkpoint_name)
                print('Saving checkpoint to ', checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

        #Plot the loss curves
        epoch += 1
        epoch_list.append(epoch)
        for k, v in losses.items():
            monitor_epoch_losses[k].append(v)
            draw_curve(epoch_list, monitor_epoch_losses[k], k)


if __name__ == '__main__':
    print("CONTROL")
    args = parser.parse_args()
    main(args)

