import tensorflow as tf
import math


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        weight_init = tf.keras.initializers.he_uniform()
        self.w = tf.Variable(initial_value=weight_init(
            shape=[int(input_shape[-1]), self.num_classes],
            dtype=tf.float32
        ), trainable=True)
        # self.w = self.add_weight(
        #     "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name=f'cos_m_{self.name}')
        self.sin_m = tf.identity(math.sin(self.margin), name=f'sin_m_{self.name}')
        self.th = tf.identity(math.cos(math.pi - self.margin), name=f'th_{self.name}')
        self.mm = tf.multiply(self.sin_m, self.margin, name=f'mm_{self.name}')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name=f'normed_embd_{self.name}')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name=f'normed_weights_{self.name}')

        cos_t = tf.matmul(normed_embds, normed_w, name=f'cos_t_{self.name}')
        sin_t = tf.sqrt(tf.clip_by_value(1. - cos_t ** 2, 1e-9, 1.0, name=f'fix_nan_{self.name}'), name=f'sin_t_{self.name}')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name=f'cos_mt_{self.name}')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name=f'one_hot_mask_{self.name}')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists
    
    def get_config(self):
        config = super(ArcMarginPenaltyLogists, self).get_config()
        config.update({ 'num_classes': self.num_classes,
                        'margin': self.margin,
                        'logits_scale': self.logist_scale })
        return config