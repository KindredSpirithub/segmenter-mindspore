"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""
from pathlib import Path
import mindspore 
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import initializer
import numpy

class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        # super(FeedForward, self).__init__()
        super().__init__()
        self.fc1 = nn.Dense(dim, hidden_dim)
        initialization = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        # initialization = mindspore.common.initializer.XavierUniform()
        self.fc1.weight.set_data(initializer(initialization, [hidden_dim, dim])) # 与torch一致
        self.act = nn.GELU(approximate=False) # 输出值和torch有差异
        # self.act = nn.LeakyReLU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Dense(hidden_dim, out_dim)
        self.fc2.weight.set_data(initializer(initialization, [out_dim, hidden_dim]))
        self.drop = nn.Dropout(1.-dropout)

    @property
    def unwrapped(self):
        return self

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #######add
        # b = x.asnumpy() # 数据类型转换
        # numpy.save("act.npy",b)

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        # super(Attention, self).__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None
        initialization = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        # initialization = mindspore.common.initializer.XavierUniform()
        self.qkv = nn.Dense(dim, dim * 3)
        self.qkv.weight.set_data(initializer(initialization, [dim * 3, dim]))

        self.attn_drop = nn.Dropout(1.-dropout)
        self.proj = nn.Dense(dim, dim)
        self.proj.weight.set_data(initializer(initialization, [dim, dim]))

        self.proj_drop = nn.Dropout(1.-dropout)
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.attn_matmul_v = ops.BatchMatMul()
        self.softmax = ops.Softmax()
    @property
    def unwrapped(self):
        return self

    def construct(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape((B, N, 3, self.heads, C // self.heads))
            .transpose((2, 0, 3, 1, 4))
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        attn = self.q_matmul_k(q, k)* self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.attn_matmul_v(attn, v)
        x = x.transpose((0, 2, 1, 3)).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0) # always be 0

        self.shape = ops.Shape()
        self.ones = ops.Ones()
        self.dropout = nn.Dropout(self.keep_prob)

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x) # B N C
            mask = self.ones((x_shape[0], 1, 1), mindspore.float32)
            x = self.dropout(mask)*x
        return x

from mindspore import nn
from mindspore import ops
import mindspore.nn.probability.distribution as msd

class DropPathWithScale(nn.Cell):
    """
    DropPath function with keep prob scale.
​
    Args:
        drop_prob(float): Drop rate, (0, 1). Default:0.0
        scale_by_keep(bool): Determine whether to scale. Default: True.
​
    Returns:
        Tensor
    """

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPathWithScale, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - self.drop_prob
        self.scale_by_keep = scale_by_keep
        
        self.div = ops.Div()

    def construct(self, x):
        if self.drop_prob > 0.0 and self.training:
            self.bernoulli = msd.Bernoulli(probs=self.keep_prob)
            random_tensor = self.bernoulli.sample((x.shape[0],) + (1,) * (x.ndim - 1))
            if self.keep_prob > 0.0 and self.scale_by_keep:
                random_tensor = self.div(random_tensor, self.keep_prob)
            x = x * random_tensor

        return x

class Block(nn.Cell):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=[dim],epsilon=1e-5) 
        # self.norm2 = nn.LayerNorm([dim]) 
        self.norm2 = nn.LayerNorm(normalized_shape=[dim],epsilon=1e-5) # layernorm和pytorch不一致，应该是epsilon默认不一样导致
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) 
        # self.drop_path = DropPathWithScale(drop_path) if drop_path>0.0 else ops.Identity() # torch存在微小差异
        # self.drop_path = ops.Identity()
        # self.drop_path = DropPathWithScale(0.05)
    def construct(self, x, mask=None, return_attention=False):
        
        x_norm = self.norm1(x)
        # b = x_norm.asnumpy() # 数据类型转换
        # numpy.save("in.npy",b)

        y, attn = self.attn.construct(x_norm, mask) # 输出和torch一致
        if return_attention:
            return attn

        # b = y.asnumpy() # 数据类型转换
        # numpy.save("att.npy",b)

        x = x + self.drop_path(y)
        ######## add
        # b = x.asnumpy() # 数据类型转换
        # numpy.save("att.npy",b)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        ######## add
        # b = x.asnumpy() # 数据类型转换
        # numpy.save("out.npy",b)
        return x


