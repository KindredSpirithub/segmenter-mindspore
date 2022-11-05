import mindspore 
import mindspore.nn as nn
from mindspore import ops
from mindspore import Parameter
from src.blocks import Block
import numpy as np
from mindspore.common.initializer import initializer
    
class MaskTransformer(nn.Cell):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super(MaskTransformer, self).__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.CellList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        ) # 两层
        # initialization = mindspore.common.initializer.Normal(sigma=1.0)
        initialization = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        # initialization = mindspore.common.initializer.XavierUniform()

        self.cls_emb = Parameter(initializer(initialization, (1, n_cls, d_model)),
                                       name='cls_emb', requires_grad=True)

        self.proj_dec = nn.Dense(d_encoder, d_model)
        self.proj_dec.weight.set_data(initializer(initialization, [d_model, d_encoder]))

        # self.proj_patch = nn.Dense(d_model, d_model, has_bias=False)
        # self.proj_patch.weight.set_data(initializer(initialization, [d_model, d_model]))

        # self.proj_classes = nn.Dense(d_model, d_model, has_bias=False)
        # self.proj_classes.weight.set_data(initializer(initialization, [d_model, d_model]))
        
        self.proj_patch = Parameter(initializer(initialization, (d_model, d_model)),
                                       name='proj_patch', requires_grad=True)
        self.proj_classes = Parameter(initializer(initialization, (d_model, d_model)),
                                       name='proj_classes', requires_grad=True)

        self.decoder_norm = nn.LayerNorm(normalized_shape=[d_model],epsilon=1e-5)
        self.mask_norm = nn.LayerNorm(normalized_shape=[n_cls],epsilon=1e-5)

        self.matmul_tr = ops.BatchMatMul(transpose_b=True)
        self.matmul_tr1 = ops.MatMul(transpose_b=False)

        self.l2_normalize1 = ops.L2Normalize(axis=-1)
        self.l2_normalize2 = ops.L2Normalize(axis=-1)

    def construct(self,x,im_size):
        H, W = im_size
        input=x
        x = self.proj_dec(x) #[8,2304,192] ,作用是把encoder的特征维度转换为decoder的特征维度，但是这里是相等的
        
        shape = (x.shape[0], -1, -1)
        broadcast_to = ops.BroadcastTo(shape)
        cls_emb = broadcast_to(self.cls_emb) #[batch,cat_num,192]

        concat = ops.Concat(1)
        x = concat((x, cls_emb)) #[8,2323,192]
        t3=x
        for blk in self.blocks:
            x = blk(x) 
        x = self.decoder_norm(x) #[8,2323,192]

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]

        # patches = patches @ self.proj_patch #[8,2304,192]
        # cls_seg_feat = cls_seg_feat @ self.proj_classes # [8,19,192]

        b,l,c = patches.shape
        patches = self.matmul_tr1(patches.reshape((b*l, c)), self.proj_patch).reshape((b,l,c))#[8,2304,19] , 
        b, l1, c = cls_seg_feat.shape
        cls_seg_feat = self.matmul_tr1(cls_seg_feat.reshape((b*l1, c)), self.proj_classes).reshape((b,l1,c))#[8,2304,19] , 

        # patches = self.proj_patch(patches)
        # cls_seg_feat = self.proj_classes(cls_seg_feat)

        patches = self.l2_normalize1(patches)
        cls_seg_feat = self.l2_normalize2(cls_seg_feat)

        masks = self.matmul_tr(patches, cls_seg_feat)#[8,2304,19] , 
        tt1=masks
        masks = self.mask_norm(masks)
        b, n, c = masks.shape
        h = H // self.patch_size
        # h = int(H / self.patch_size) 
        # # 如果使用int类型转换，就不能用类名直接调用construct, 应该是mindspore的bug
        # w = int(n/h)   
        w = n//h
        masks = masks.reshape((b,h,w,c)).transpose((0,3,1,2))
        return  masks

