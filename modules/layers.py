import tensorflow as tf
import math


class ArcMarginPenaltyLogits(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, kernel_regularizer, margin=0.5, scale=64, name='arc_margin_penalty'):
        super(ArcMarginPenaltyLogits, self).__init__(name=name)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.w = self.add_weight(name='weights', initializer='glorot_uniform', 
                                shape=[input_shape[-1], self.num_classes],
                                trainable=True,
                                regularizer=self.kernel_regularizer)
        
        self.cos_m = tf.math.cos(self.margin)
        self.sin_m = tf.math.sin(self.margin)
        self.mm = self.sin_m * self.margin
        self.pi = tf.constant(math.pi)
        self.threshold = tf.math.cos(self.pi - self.margin)

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.scale * tf.subtract(tf.multiply(cos_t, self.cos_m), tf.multiply(sin_t, self.sin_m), name='cos_mt')
        cond_v = cos_t - self.threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        
        keep_val = self.scale * (cos_t - self.mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        s_cos_t = tf.multiply(self.scale, cos_t, name='scalar_cos_t')
        logits = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
        return logits
    
    def get_config(self):
        config = super(ArcMarginPenaltyLogits, self).get_config()
        config.update({ 'num_classes': self.num_classes,
                        'margin': self.margin,
                        'logits_scale': self.scale,
                        'kernel_regularizer': self.kernel_regularizer })
        return config