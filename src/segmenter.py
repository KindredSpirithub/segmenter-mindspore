import mindspore 
import mindspore.nn as nn
from mindspore import ops
from mindspore import Parameter
from src.utils import padding, unpadding
from src.vit import *

class Segmenter(nn.Cell):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super(Segmenter, self).__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, im):
        H_ori, W_ori = im.shape[2], im.shape[3]
        im = padding(im, self.patch_size)
        H, W = im.shape[2], im.shape[3]

        x = self.encoder.construct(im) # [8,2305,192]
 
        # remove CLS/DIST tokens for decoding
        if self.encoder.distilled:
            num_extra_tokens = 2
        else:
            num_extra_tokens = 1
        # num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:] # [8,2304,192]
        masks = self.decoder.construct(x, im_size=[H, W]) #[8,cat_num,48,48]

        resize_bilinear = nn.ResizeBilinear()
        masks = resize_bilinear(masks, size=(H, W))
        masks = unpadding(masks, (H_ori, W_ori)) #[8,cat_num,768,768]

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


#2) Using user-defined WithLossCell
class SegmenterWithLossCell(nn.Cell):
   def __init__(self, backbone, loss_fn):
       super(SegmenterWithLossCell, self).__init__(auto_prefix=False)
       self._backbone = backbone
       self._loss_fn = loss_fn

   def construct(self, x, label):
       out = self._backbone(x)
       return self._loss_fn(out, label)

   @property
   def backbone_network(self):
       return self._backbone

class CustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer):
        """入参有两个：训练网络，优化器"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                           # 定义前向网络
        self.network.set_grad()                          # 构建反向网络
        self.optimizer = optimizer                       # 定义优化器
        self.weights = self.optimizer.parameters         # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        loss = self.network(*inputs)                            # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        self.optimizer(grads)                                   # 使用优化器更新权重参数
        return loss