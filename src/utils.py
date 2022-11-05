import mindspore 
import mindspore.nn as nn
from mindspore import ops
from mindspore import Parameter
import math
import numpy as np

def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not
            # include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not
            # include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0},
            {'params': decay_params}]


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape((1, gs_old_h, gs_old_w, -1)).transpose(0, 3, 1, 2)
    resize_bilinear = nn.ResizeBilinear()
    posemb_grid = resize_bilinear(posemb_grid, size=(gs_h, gs_w))
    posemb_grid = posemb_grid.transpose(0, 2, 3, 1).reshape((1, gs_h * gs_w, -1))

    concat = ops.Concat(1)
    posemb = concat((posemb_tok, posemb_grid)) # [8,2305,192]
    return posemb

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 1 + ("dist_token" in state_dict.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict

def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.shape[2], im.shape[3]
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        pad = nn.Pad(paddings=((0, pad_h), (0, pad_w)), mode="CONSTANT")
        im_padded = pad(im)
        # im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.shape[2], y.shape[3]
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


def resize(im, smaller_size):
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        resize_bilinear = nn.ResizeBilinear()
        im_res = resize_bilinear(im, size=(int(h_res), int(w_res)))
    else:
        im_res = im
    return im_res


def sliding_window(im, flip, window_size, window_stride):
    B, C, H, W = im.shape
    ws = window_size

    windows = {"crop": [], "anchors": []}
    h_anchors = np.arange(0, H, window_stride)
    w_anchors = np.arange(0, W, window_stride)
    h_anchors = [h.item() for h in h_anchors if h < H - ws] + [H - ws]
    w_anchors = [w.item() for w in w_anchors if w < W - ws] + [W - ws]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, ha : ha + ws, wa : wa + ws]
            windows["crop"].append(window)
            windows["anchors"].append((ha, wa))
    windows["flip"] = flip
    windows["shape"] = (H, W)
    return windows


def merge_windows(windows, window_size, ori_shape):
    ws = window_size
    im_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    C = im_windows[0].shape[0]
    H, W = windows["shape"]
    flip = windows["flip"]

    zeros = ops.Zeros()
    logit = zeros((C, H, W), mindspore.float32)
    count = zeros((1, H, W), mindspore.float32)
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, ha : ha + ws, wa : wa + ws] += window
        count[:, ha : ha + ws, wa : wa + ws] += 1
    logit = logit / count

    resize_bilinear = nn.ResizeBilinear()
    expand_dims = ops.ExpandDims()
    logit = expand_dims(logit, 0)
    logit = resize_bilinear(logit, size=ori_shape)

    if flip:
        op = ops.ReverseV2(axis=[2])
        logit = op(logit)
        # logit = torch.flip(logit, (2,))
    # softmax = ops.Softmax(axis=0)
    # result = softmax(logit)
    return logit


def inference(
    model,
    ims,
    ims_metas,
    ori_shape,
    window_size,
    window_stride,
    batch_size,
):
    C = model.n_cls
    zeros = ops.Zeros()
    seg_map = zeros((C, ori_shape[0], ori_shape[1]), mindspore.float32)
    # seg_map = torch.zeros((C, ori_shape[0], ori_shape[1]), device=ptu.device)
    for im, im_metas in zip(ims, ims_metas):
        # im = im.to(ptu.device)
        im = resize(im, window_size)
        flip = im_metas["flip"]
        windows = sliding_window(im, flip, window_size, window_stride)
        stack = ops.Stack()
        crops = stack(windows.pop("crop"))[:, 0]      
        # crops = torch.stack(windows.pop("crop"))[:, 0]
        B = len(crops)
        WB = batch_size
        seg_maps = zeros((B, C, window_size, window_size), mindspore.float32)
        # with torch.no_grad():
        for i in range(0, B, WB):
            seg_maps[i : i + WB] = model.forward(crops[i : i + WB])
        windows["seg_maps"] = seg_maps
        im_seg_map = merge_windows(windows, window_size, ori_shape)
        seg_map += im_seg_map
    seg_map /= len(ims)
    return seg_map


def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    Prod = ops.ReduceProd(keep_dims=False)
    n_params = sum([Prod(mindspore.Tensor(p.size())) for p in model_parameters])
    return n_params.item()

def get_confusion_matrix(label, pred, shape, num_class, ignore=-1):
    """Calcute the confusion matrix by given label and pred."""
    output = pred.asnumpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8) # (b,1024,2048)
    seg_gt = np.asarray(label.asnumpy()[:, :shape[-2], :shape[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix

def get_confusion_matrix1(label, pred, shape, num_class, ignore=-1):
    """Calcute the confusion matrix by given label and pred."""
    pred = pred.transpose((0, 2, 3, 1)).argmax(axis=3)
    seg_pred = pred.asnumpy()
    # output = np.argmax(output, axis=3)
    # seg_pred = np.asarray(output, dtype=np.uint8)
    seg_gt = label.asnumpy()[:, :shape[-2], :shape[-1]]
    # seg_gt = np.asarray(label[:, :shape[-2], :shape[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index_ = (seg_gt * num_class + seg_pred).astype('int32')
    
    label_count = np.bincount(index_)

    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix


def cosine_lr(base_lr, decay_steps, total_steps):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))


def poly_lr(base_lr, decay_steps, total_steps, end_lr=0.0001, power=0.9):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield (base_lr - end_lr) * ((1.0 - step_ / decay_steps) ** power) + end_lr


def exponential_lr(base_lr, decay_steps, decay_rate, total_steps, staircase=False):
    for i in range(total_steps):
        if staircase:
            power_ = i // decay_steps
        else:
            power_ = float(i) / decay_steps
        yield base_lr * (decay_rate ** power_)
