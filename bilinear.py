import torch
import torch.nn.functional as F


def crop_bbox_batch(feats, bbox, HH, WW=None):

  N, C, H, W = feats.size()

  B = bbox.size(0)
  if WW is None: WW = HH
  dtype, device = feats.dtype, feats.device
  crops = torch.zeros(B, C, HH, WW, dtype=dtype, device=device)

  for i in range(N):
    n = B
    cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
    cur_bbox = bbox
    cur_crops = crop_bbox(cur_feats, cur_bbox, HH, WW)
    crops = cur_crops
  return crops


def crop_bbox(feats, bbox, HH, WW=None, backend='cudnn'):
  N = feats.size(0)
  assert bbox.size(0) == N
  assert bbox.size(1) == 4
  if WW is None: WW = HH
  if backend == 'cudnn':
    # Change box from [0, 1] to [-1, 1] coordinate system
    bbox = 2 * bbox - 1
  x0, y0 = bbox[:, 0], bbox[:, 1]
  x1, y1 = bbox[:, 2], bbox[:, 3]
  X = tensor_linspace(x0, x1, steps=WW).view(N, 1, WW).expand(N, HH, WW)
  Y = tensor_linspace(y0, y1, steps=HH).view(N, HH, 1).expand(N, HH, WW)
  if backend == 'jj':
    return bilinear_sample(feats, X, Y)
  elif backend == 'cudnn':
    grid = torch.stack([X, Y], dim=3)
    return F.grid_sample(feats, grid)


def uncrop_bbox(feats, bbox, H, W=None, fill_value=0):
  """
  Inverse operation to crop_bbox; construct output images where the feature maps
  from feats have been reshaped and placed into the positions specified by bbox.

  Inputs:
  - feats: Tensor of shape (N, C, HH, WW)
  - bbox: Bounding box coordinates of shape (N, 4) in the format
    [x0, y0, x1, y1] in the [0, 1] coordinate space.
  - H, W: Size of output.
  - fill_value: Portions of the output image that are outside the bounding box
    will be filled with this value.

  Returns:
  - out: Tensor of shape (N, C, H, W) where the portion of out[i] given by
    bbox[i] contains feats[i], reshaped using bilinear sampling.
  """
  N, C = feats.size(0), feats.size(1)
  assert bbox.size(0) == N
  assert bbox.size(1) == 4
  if W is None: H = W

  x0, y0 = bbox[:, 0], bbox[:, 1]
  x1, y1 = bbox[:, 2], bbox[:, 3]
  ww = x1 - x0
  hh = y1 - y0

  x0 = x0.contiguous().view(N, 1).expand(N, H)
  x1 = x1.contiguous().view(N, 1).expand(N, H)
  ww = ww.view(N, 1).expand(N, H)

  y0 = y0.contiguous().view(N, 1).expand(N, W)
  y1 = y1.contiguous().view(N, 1).expand(N, W)
  hh = hh.view(N, 1).expand(N, W)
  
  X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).to(feats)
  Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).to(feats)

  X = (X - x0) / ww
  Y = (Y - y0) / hh

  # For ByteTensors, (x + y).clamp(max=1) gives logical_or
  X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
  Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)
  out_mask = (X_out_mask + Y_out_mask).clamp(max=1)
  out_mask = out_mask.view(N, 1, H, W).expand(N, C, H, W)

  X = X.view(N, 1, W).expand(N, H, W)
  Y = Y.view(N, H, 1).expand(N, H, W)

  out = bilinear_sample(feats, X, Y)
  out[out_mask] = fill_value
  return out


def bilinear_sample(feats, X, Y):
  N, C, H, W = feats.size()
  assert X.size() == Y.size()
  assert X.size(0) == N
  _, HH, WW = X.size()

  X = X.mul(W)
  Y = Y.mul(H)

  # Get the x and y coordinates for the four samples
  x0 = X.floor().clamp(min=0, max=W-1)
  x1 = (x0 + 1).clamp(min=0, max=W-1)
  y0 = Y.floor().clamp(min=0, max=H-1)
  y1 = (y0 + 1).clamp(min=0, max=H-1)

  y0x0_idx = (W * y0 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
  y1x0_idx = (W * y1 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
  y0x1_idx = (W * y0 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)
  y1x1_idx = (W * y1 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)

  feats_flat = feats.view(N, C, H * W)
  v1 = feats_flat.gather(2, y0x0_idx.long()).view(N, C, HH, WW)
  v2 = feats_flat.gather(2, y1x0_idx.long()).view(N, C, HH, WW)
  v3 = feats_flat.gather(2, y0x1_idx.long()).view(N, C, HH, WW)
  v4 = feats_flat.gather(2, y1x1_idx.long()).view(N, C, HH, WW)

  # Compute the weights for the four samples
  w1 = ((x1 - X) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
  w2 = ((x1 - X) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)
  w3 = ((X - x0) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
  w4 = ((X - x0) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)

  # Multiply the samples by the weights to give our interpolated results.
  out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
  return out


def tensor_linspace(start, end, steps=10):
  assert start.size() == end.size()
  view_size = start.size() + (1,)
  w_size = (1,) * start.dim() + (steps,)
  out_size = start.size() + (steps,)

  start_w = torch.linspace(1, 0, steps=steps).to(start)
  start_w = start_w.view(w_size).expand(out_size)
  end_w = torch.linspace(0, 1, steps=steps).to(start)
  end_w = end_w.view(w_size).expand(out_size)

  start = start.contiguous().view(view_size).expand(out_size)
  end = end.contiguous().view(view_size).expand(out_size)

  out = start_w * start + end_w * end
  return out