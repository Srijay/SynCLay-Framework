import torch
import torch.nn as nn
import torch.nn.functional as F

from bilinear import crop_bbox_batch
from layers import GlobalAvgPool, Flatten, get_activation, build_cnn


class PatchDiscriminator(nn.Module):
  def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
               padding='same', pooling='avg', input_size=(256,256),
               layout_dim=0):
    super(PatchDiscriminator, self).__init__()
    #input_dim = 1 + layout_dim
    input_dim = 3

    arch = 'I%d,%s' % (input_dim, arch)

    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling,
      'padding': padding,
    }
    self.cnn, output_dim = build_cnn(**cnn_kwargs)
    self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

  def forward(self, x, layout=None):
    if layout is not None:
      x = torch.cat([x, layout], dim=1)
    return self.cnn(x)


class AcDiscriminator(nn.Module):

  def __init__(self, vocab, arch, normalization='none', activation='relu',
               padding='same', pooling='avg'):
    super(AcDiscriminator, self).__init__()
    self.vocab = vocab

    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling, 
      'padding': padding,
    }
    cnn, D = build_cnn(**cnn_kwargs)

    self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 32))

    num_objects = len(vocab)

    self.real_classifier = nn.Linear(32, 1)
    self.obj_classifier = nn.Linear(32, num_objects)

  def forward(self, x, y):
    if x.dim() == 3:
      x = x[:, None]
    vecs = self.cnn(x)
    # vecs = checkpoint_sequential(self.cnn, 2, x)
    real_scores = self.real_classifier(vecs)
    obj_scores = self.obj_classifier(vecs)
    ac_loss = F.cross_entropy(obj_scores, y)
    return real_scores, ac_loss


class AcCropDiscriminator(nn.Module):

  def __init__(self, vocab, arch, normalization='none', activation='relu',
               object_size=64, padding='same', pooling='avg'):
    super(AcCropDiscriminator, self).__init__()

    self.vocab = vocab
    self.discriminator = AcDiscriminator(vocab, arch, normalization,
                                         activation, padding, pooling)
    self.object_size = object_size

  def forward(self, imgs, boxes, y):
    crops = crop_bbox_batch(imgs, boxes, self.object_size, self.object_size)
    real_scores, ac_loss = self.discriminator(crops, y)
    #real_scores = self.discriminator(crops)
    return real_scores, ac_loss


class MaskDiscriminator(nn.Module):

  def __init__(self, in_channels=3):
    super(MaskDiscriminator, self).__init__()
    self.mask_discriminator = self.build_mask_discriminator(in_channels)


  def forward(self, X):
    return self.mask_discriminator(X)


  def build_mask_discriminator(self,in_channels):

    layers = []

    layers.append(nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False))
    layers.append(nn.BatchNorm2d(32))
    layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers.append(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False))
    layers.append(nn.BatchNorm2d(64))
    layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers.append(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False))
    layers.append(nn.BatchNorm2d(128))
    layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False))
    layers.append(nn.BatchNorm2d(256))
    layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers.append(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(169, 1))
    #layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


class Pix2PixDiscriminator(nn.Module):

  def __init__(self, in_channels=3):
    super(Pix2PixDiscriminator, self).__init__()

    def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
      """Returns downsampling layers of each discriminator block"""
      layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
      if normalization:
        layers.append(nn.InstanceNorm2d(out_filters))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    self.model = nn.Sequential(

      #If model loading failed, try this discriminator and keep kernel size as 5
      # *discriminator_block(in_channels, 16, normalization=False),
      # *discriminator_block(16, 32),
      # *discriminator_block(32, 64),
      # *discriminator_block(64, 128),
      # *discriminator_block(128, 256),
      # #nn.ZeroPad2d((1, 0, 1, 0)),
      # nn.Conv2d(256, 1, 5, padding=1, bias=False),
      # #nn.ReLU()

      *discriminator_block(in_channels, 64, normalization=False),
      *discriminator_block(64, 128),
      *discriminator_block(128, 256),
      # *discriminator_block(256, 512), #Added to safronize framework
      *discriminator_block(256, 512, stride=1),
      # nn.ZeroPad2d((1, 0, 1, 0)),
      nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=False),
      # nn.Sigmoid() #It was not there when trained with residual generator so turn it off while its inference
    )

  def forward(self, mask, img):
    # Concatenate image and condition image by channels to produce input
    # img_input = torch.cat((mask, img), 1)
    # output = checkpoint_sequential(self.model, 1, img)
    output = self.model(img)
    return output



