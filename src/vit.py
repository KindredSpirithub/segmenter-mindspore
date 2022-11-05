
import mindspore 
import mindspore.nn as nn
from mindspore import ops
from mindspore import Parameter
import numpy as np
from src.blocks import Block
from src.utils import resize_pos_embed
from mindspore.common.initializer import initializer
import numpy

class PatchEmbedding(nn.Cell):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode='valid', has_bias=True)
        # self.dict = self.proj.parameters_dict()
    def construct(self, im):
        
        x = self.proj(im)
        B, C, H, W = x.shape
        x = x.reshape((B,C,H*W))
        x = x.transpose((0, 2, 1))
        return x

class VisionTransformer(nn.Cell):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.0,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
    ):
        # super(VisionTransformer, self).__init__()
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(keep_prob=1.-dropout)
        self.n_cls = n_cls

        initialization = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        # cls and pos tokens
        self.cls_token = Parameter(initializer(initialization, (1, 1, d_model)),
                                       name='cls_token', requires_grad=True)

        self.distilled = distilled
        if self.distilled:
            self.dist_token = Parameter(initializer(initialization, (1, 1, d_model)),
                                       name='dist_token', requires_grad=True)
            self.pos_embed = Parameter(initializer(initialization, (1, self.patch_embed.num_patches + 2, d_model)),
                                           name='pos_embed', requires_grad=True)
            # self.head_dist = nn.Dense(d_model, n_cls)
        else:
            # self.patch_embed.num_patches = 24*24
            self.pos_embed = Parameter(initializer(initialization, (1, self.patch_embed.num_patches + 1, d_model)),
                                           name='pos_embed', requires_grad=True)
        
        # linspace = ops.LinSpace()

        # dpr = [x.item() for x in np.linspace(0, drop_path_rate, n_layers)]
        dpr = [0., 0.00909090880304575, 0.0181818176060915, 0.027272727340459824, 0.036363635212183, 0.045454543083906174, 0.054545458406209946, 0.06363636255264282, 0.0727272778749466, 0.08181818574666977, 0.09090909361839294, 0.10000000149011612]
        
        # dpr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.blocks = nn.CellList([Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)])

        # output head
        self.norm = nn.LayerNorm(normalized_shape=[d_model],epsilon=1e-5)
        # self.head = nn.Dense(d_model, n_cls)
        # self.head.weight.set_data(initializer(initialization, [n_cls, d_model]))

        self.pre_logits = ops.Identity()
        self.ones = ops.Ones()
        
    def construct(self, im, return_features=True):
        B, _, H, W = im.shape
        PS = self.patch_size
        ################ 用于调试
        # tm = self.ones((1,3,768,768), mindspore.float32)
        # x = self.patch_embed(tm) # [8,2304,192]

        ############################
        x = self.patch_embed(im) # [8,2304,192]

        shape = (B, -1, -1)
        broadcast_to = ops.BroadcastTo(shape)
        cls_tokens = broadcast_to(self.cls_token)

        concat = ops.Concat(1)
        if self.distilled:
            dist_tokens = broadcast_to(self.dist_token)
            x = concat((cls_tokens, dist_tokens, x))
        else:
            x = concat((cls_tokens, x)) # [8,2305,192]

        pos_embed = self.pos_embed
        if self.distilled:
            num_extra_tokens=2
        else:
            num_extra_tokens=1
        if x.shape[1] != pos_embed.shape[1]:
            # print("jjjjjjjjjjjjjjjjjjj")
            pos_embed = resize_pos_embed(
                pos_embed,
                (24,24),
                # self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        ####### add
        # b = x.asnumpy() # 数据类型转换
        # numpy.save("out1.npy",b)
        # print("gggggggggggggggg")
        x = self.norm(x)

        if return_features:
            return x #[8,2305,192]

        # if self.distilled:
        #     x, x_dist = x[:, 0], x[:, 1]
        #     x = self.head(x)
        #     x_dist = self.head_dist(x_dist)
        #     x = (x + x_dist) / 2
        # else:

        # x = x[:, 0]
        # x = self.head(x)
        return x

    # def get_attention_map(self, im, layer_id):
    #     if layer_id >= self.n_layers or layer_id < 0:
    #         raise ValueError(
    #             f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
    #         )
    #     B, _, H, W = im.shape
    #     PS = self.patch_size

    #     x = self.patch_embed(im)
    #     cls_tokens = self.cls_token.expand(B, -1, -1)

    #     x = ops.Concat((cls_tokens, x), dim=1)

    #     pos_embed = self.pos_embed
    #     num_extra_tokens = 1 + self.distilled
    #     if x.shape[1] != pos_embed.shape[1]:
    #         pos_embed = resize_pos_embed(
    #             pos_embed,
    #             self.patch_embed.grid_size,
    #             (H // PS, W // PS),
    #             num_extra_tokens,
    #         )
    #     x = x + pos_embed

    #     for i, blk in enumerate(self.blocks):
    #         if i < layer_id:
    #             x = blk(x)
    #         else:
    #             return blk(x, return_attention=True)
