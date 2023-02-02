import torch
import torch.nn.functional as F

def pix2pix_masks_to_layout(img_in, boxes, H, W=None, pooling='sum'):

    O = img_in.shape[0]
    obj_to_img = torch.tensor([0] * O)
    if torch.cuda.is_available():
        obj_to_img = obj_to_img.cuda()

    grid = _boxes_to_grid(boxes, H, W)

    sampled = F.grid_sample(img_in, grid)  # Bilinear Interpolation

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)

    return out


def masks_to_layout(vecs, boxes, masks, H, W=None, pooling='sum', gpu_id=0):
    """
    Inputs:
    - vecs: Tensor of shape (O, D) giving vectors
    - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
    [x0, y0, x1, y1] in the [0, 1] coordinate space
    - masks: Tensor of shape (O, M, M) giving binary masks for each object
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - H, W: Size of the output image.

    Returns:
    - out: Tensor of shape (N, D, H, W)
    """

    O, D = vecs.size()
    obj_to_img = torch.tensor([0] * O)
    if torch.cuda.is_available():
        obj_to_img = obj_to_img.cuda(gpu_id)
    M = masks.size(1)

    assert masks.size() == (O, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    img_in = vecs.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)

    sampled = F.grid_sample(img_in, grid)  # Bilinear Interpolation

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)

    return out


def trimasks_to_layout(vecs, boxes, trimasks, H, W=None, pooling='sum'):

    O, D = vecs.size()
    obj_to_img = torch.tensor([0] * O)
    if torch.cuda.is_available():
        obj_to_img = obj_to_img.cuda()
    M = trimasks.size(2)

    assert trimasks.size() == (O, 3, M, M)

    if W is None:
        W = H

    trigrid = _boxes_to_trigrid(boxes, H, W)

    img_in = vecs.view(O, D, 1, 1, 1) * trimasks.float().view(O, 1, 3, M, M)

    sampled = F.grid_sample(img_in, trigrid)  # Bilinear Interpolation

    print(sampled.size())
    exit(0)

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)

    return out


def _boxes_to_trigrid(boxes, H, W):

    O = boxes.size(0)
    print(boxes.size())
    boxes = boxes.view(O, 4, 1, 1)
    print(boxes.size())
    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]
    ww = x1 - x0
    hh = y1 - y0


    X = torch.linspace(0, 1, steps=W).view(1, 1, 1, W).to(boxes) #second 1 is for channels
    Y = torch.linspace(0, 1, steps=H).view(1, 1, H, 1).to(boxes)
    print(X.size())
    X = (X - x0) / ww  # (O, 1, W)
    print(X.size())
    exit(0)
    Y = (Y - y0) / hh  # (O, H, 1)

    print(X.size())
    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    print(X.size())
    exit(0)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid


def _boxes_to_grid(boxes, H, W):

    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]
    ww = x1 - x0
    hh = y1 - y0

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid


def _pool_samples(samples, obj_to_img, pooling='sum'):

    dtype, device = samples.dtype, samples.device

    O, D, H, W = samples.size()
    N = obj_to_img.data.max().item() + 1  # obj_to_img is fixed, so N=1 here

    # Use scatter_add to sum the sampled outputs for each image
    out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
    idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)

    out = out.scatter_add(0, idx, samples)

    if pooling == 'avg':
        # Divide each output mask by the number of objects; use scatter_add again
        # to count the number of objects per image.
        ones = torch.ones(O, dtype=dtype, device=device)
        obj_counts = torch.zeros(N, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
        print(obj_counts)
        obj_counts = obj_counts.clamp(min=1)
        out = out / obj_counts.view(N, 1, 1, 1)
    elif pooling != 'sum':
        raise ValueError('Invalid pooling "%s"' % pooling)

    return out