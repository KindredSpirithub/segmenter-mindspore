from pathlib import Path
import yaml
import math
import os
import mindspore 
import mindspore.nn as nn

from src.vit import VisionTransformer
from src.utils import checkpoint_filter_fn
from src.decoder import MaskTransformer
from src.segmenter import Segmenter
# from vit_1 import *


# @register_model
# def vit_base_patch8_384(pretrained=False, **kwargs):
#     """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer(
#         "vit_base_patch8_384",
#         pretrained=pretrained,
#         default_cfg=dict(
#             url="",
#             input_size=(3, 384, 384),
#             mean=(0.5, 0.5, 0.5),
#             std=(0.5, 0.5, 0.5),
#             num_classes=1000,
#         ),
#         **model_kwargs,
#     )
#     return model

from mindspore.train.serialization import load_checkpoint, load_param_into_net

# mindspore官方VIT实现
# def create_vit1(model_cfg):
#     model_cfg = model_cfg.copy()
#     backbone = model_cfg.pop("backbone")
#     model_cfg["n_cls"] = 1000
#     mlp_expansion_ratio = 2
#     model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]
#     model = get_network(backbone_name="vit_base_patch32", args=model_cfg)

#     pretrained = "./vit_ascend_v160_imagenet2012_official_cv_acc74.17.ckpt"
#     param_dict = load_checkpoint(ckpt_file_name=pretrained)
#     # 因为imagenet是224的尺寸，因此pos_embedding对应不上
#     param_dict.pop("pos_embedding", None)
#     param_dict.pop("adam_v.pos_embedding", None)
#     param_dict.pop("adam_m.pos_embedding", None)
#     # yy = param_dict.keys()
#     # state_dict = model.parameters_dict()

#     load_param_into_net(model, param_dict)

#     # mindspore.load_checkpoint(pretrained, model, strict_load=False)
#     # print("load pretrained models success")
#     return model

# pytorch版segmenter的VIT实现
def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    # if backbone in default_cfgs:
    #     default_cfg = default_cfgs[backbone]
    # else:
    #     default_cfg = dict(
    #         pretrained=False,
    #         num_classes=1000,
    #         drop_rate=0.0,
    #         drop_path_rate=0.0,
    #         drop_block_rate=None,
    #     )

    # default_cfg["input_size"] = (
    #     3,
    #     model_cfg["image_size"][0],
    #     model_cfg["image_size"][1],
    # )
    model = VisionTransformer(**model_cfg)
    # state_dict = model.parameters_dict()
    
    pretrained = "/cache/user-job-dir/code/" + backbone + "_ms.ckpt" # for modelarts training
    # pretrained = backbone + "_ms.ckpt"
    # pretrained = "vit_tiny_patch16_384_ms.ckpt"
    param_dict = load_checkpoint(ckpt_file_name=pretrained)
    load_param_into_net(model, param_dict)
    # state_dict1 = model.parameters_dict()
    print("load vit model success: {}".format(pretrained))
    # mindspore.save_checkpoint(model, "tt.ckpt")
    # print(state_dict.keys())
    # if backbone == "vit_base_patch8_384":
    #     path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
    #     state_dict = torch.load(path, map_location="cpu")
    #     filtered_dict = checkpoint_filter_fn(state_dict, model)
    #     model.load_state_dict(filtered_dict, strict=True)
    # elif "deit" in backbone:
    #     load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    # else:
    #     load_custom_pretrained(model, default_cfg)
    # state_dict1 = model.parameters_dict()
    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    # 下面两行是当采用mindspore官方的vit时所用
    # decoder_cfg["d_encoder"] = vit_cfg.d_model
    # decoder_cfg["patch_size"] = vit_cfg.patch_size

    # 下面两行是当采用torch版segmenter的vit时所用
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size
    ### "mask_transformer":
    # dim = vit_cfg.d_model # 采用mindspore官方的vit时所用
    dim = encoder.d_model # 采用torch版segmenter的vit时所用
    n_heads = dim // 64
    decoder_cfg["n_heads"] = n_heads
    decoder_cfg["d_model"] = dim
    decoder_cfg["d_ff"] = 4 * dim
    decoder = MaskTransformer(**decoder_cfg)

    # pretrained = "decoder_192_ms.ckpt"
    # param_dict = load_checkpoint(ckpt_file_name=pretrained)
    # load_param_into_net(decoder, param_dict)
    # print("load decoder success: {}".format(pretrained))
    # state_dict = decoder.parameters_dict()
    return decoder


def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    state_dict = encoder.parameters_dict()
    decoder = create_decoder(encoder, decoder_cfg)
    state_dict1 = decoder.parameters_dict()
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model


def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    data = mindspore.load_checkpoint(model_path)
    # data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant
