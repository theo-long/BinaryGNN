import tensorflow as tf


class BinaryL1Regularizer(tf.keras.regularizers.Regularizer):
    """L1 distance Binary Regularizer. Implements the function |alpha - |weight||."""

    def __init__(self, strength, alpha=1, regularizer_decay=1.0):
        self.strength = strength
        self.alpha = alpha
        self.regularizer_decay = regularizer_decay

    def __call__(self, x):
        result = self.strength * tf.reduce_sum(tf.abs(self.alpha - tf.abs(x)))
        self.strength *= self.regularizer_decay
        return result


class BinaryL2Regularizer(tf.keras.regularizers.Regularizer):
    """L2 distance Binary Regularizer. Implements the function (alpha - |weight|)^2."""

    def __init__(self, strength, alpha=1, regularizer_decay=1.0):
        self.strength = strength
        self.alpha = alpha
        self.regularizer_decay = regularizer_decay

    def __call__(self, x):
        result = self.strength * tf.reduce_sum(tf.square(self.alpha - tf.abs(x)))
        self.strength *= self.regularizer_decay
        return result
