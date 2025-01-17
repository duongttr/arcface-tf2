from re import S
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    EfficientNetB0
)
from modules.layers import (
    ArcMarginPenaltyLogits
)


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)

def Backbone(backbone_type='ResNet50', use_pretrain=True):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x_in):
        backbone = None
        if backbone_type == 'ResNet50':
            backbone = ResNet50(input_shape=x_in.shape[1:], include_top=False,
                                weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            backbone = MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'EfficientNetB0':
            backbone = EfficientNetB0(input_shape=x_in.shape[1:], include_top=False,
                                weights=weights)(x_in)
        else:
            raise TypeError('backbone_type error!')
        return backbone
    return backbone


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""
    def output_layer(x_in):
        inputs = Input(x_in.shape[1:])
        x = GlobalAveragePooling2D()(inputs)
        x = Dropout(rate=0.5)(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer


def ArcHead(num_classes, margin=0.5, scale=64., w_decay=5e-4, name='ArcHead'):
    """Arc Head"""
    def arc_head(x_in, y_in):
        inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogits(num_classes=num_classes,
                                    margin=margin,
                                    kernel_regularizer=_regularizer(w_decay),
                                    scale=scale, 
                                    name=f'margin_{name}')(inputs1, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head


def NormHead(num_classes, w_decay=5e-4, name='NormHead'):
    """Norm Head"""
    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, kernel_regularizer=_regularizer(w_decay))(x)
        return Model(inputs, x, name=name)(x_in)
    return norm_head

def ArcFaceModel(input_shape=None, categorical_labels=None, name='arcface_model',
                 margin=0.5, logist_scale=64., embd_shape=512,
                 backbone_type='MobileNetV2',
                 head_type='ArcHead',
                 backbone_trainable=True,
                 w_decay=5e-4, use_pretrain=True, training=False):
    """Arc Face Model"""

    x = inputs = Input(input_shape, name='input_image')
    backbone = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)

    embds = {}
    for category in categorical_labels.keys():
        embds[category] = OutputLayer(embd_shape, w_decay=w_decay, name=f'embds_{category}')(backbone)

    if training:
        labels = []
        logists = []
        for category, classes in categorical_labels.items():
            curr_label = Input([], name=f'label_{category}')
            curr_logist = None
            if head_type == 'ArcHead':
                m, s  = None, None
                if isinstance(margin, dict):
                    m = margin[category]
                else:
                    m = margin

                if isinstance(logist_scale, dict):
                    s = logist_scale[category]
                else:
                    s = logist_scale
                
                curr_logist = ArcHead(num_classes=len(classes), margin=m,
                                    scale=s, name=f'archead_{category}')\
                                        (embds[category], curr_label)
            elif head_type == 'NormHead':
                curr_logist = NormHead(num_classes=len(classes), name=f'normhead_{category}')(embds[category])
            labels.append(curr_label)
            logists.append(curr_logist)
        return Model([inputs]+labels, logists, name=name)
    else:
        return Model(inputs, list(embds.values()), name=name)
