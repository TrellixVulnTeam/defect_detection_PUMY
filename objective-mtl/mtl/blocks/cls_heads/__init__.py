from .googlenet_cls_head import GoogLeNetClsHead
from .inception_cls_head import InceptionClsHead
from .linear_cls_head import LinearClsHead
from .fc_hidden_cls_head import FCHiddenClsHead
from .q2l_cls_head import Q2LTransformerClsHead
from .multilabel_cls_head import MultilabelClsHead
from .fc_mlp_cls_head import FCMlpClsHead

__all__ = [
    "GoogLeNetClsHead",
    "InceptionClsHead",
    "LinearClsHead",
    "FCHiddenClsHead",
    "Q2LTransformerClsHead",
    "MultilabelClsHead",
    "FCMlpClsHead",
]
