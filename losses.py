import torch

def get_gan_losses(gan_type):

  if gan_type == 'gan':
    return gan_g_loss, gan_d_loss
  else:
    raise ValueError('Improper GAN type "%s"' % gan_type)


def bce_loss(input, target):
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def _make_targets(x, y):

  return torch.full_like(x, y)


def gan_g_loss(scores_fake):

  if scores_fake.dim() > 1:
    scores_fake = scores_fake.view(-1)
  y_fake = _make_targets(scores_fake, 1)
  return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):

  assert scores_real.size() == scores_fake.size()
  if scores_real.dim() > 1:
    scores_real = scores_real.view(-1)
    scores_fake = scores_fake.view(-1)
  y_real = _make_targets(scores_real, 1)
  y_fake = _make_targets(scores_fake, 0)
  loss_real = bce_loss(scores_real, y_real)
  loss_fake = bce_loss(scores_fake, y_fake)
  return loss_real + loss_fake