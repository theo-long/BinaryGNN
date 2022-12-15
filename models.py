import tensorflow as tf
import larq as lq
from spektral.models import GCN

from typing import Tuple


class GraphConv(tf.keras.layers.Layer):
    def __init__(self, a):
        super(GraphConv, self).__init__()
        self.a = a

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        return tf.matmul(self.a, inputs)


def generate_quantized_gcn(
    channels: int,
    input_shapes: Tuple[Tuple, Tuple],
    output_features: int,
    dropout_rate: float,
    layers: int = 2,
    input_quantizer=lq.quantizers.SteSign,
    kernel_quantizer=lq.quantizers.MagnitudeAwareSign,
    **layer_kwargs
):
    node_features = tf.keras.Input(shape=(input_shapes[0]))
    adj_matrix = tf.keras.layers.Input(shape=(input_shapes[1]))
    x_intermediate = tf.keras.layers.BatchNormalization()(node_features)

    # Intermediate layers: Binary Activation - Dropout - Graph Convolution
    for layer in range(layers - 1):
        x_intermediate = tf.keras.layers.Dropout(rate=dropout_rate)(x_intermediate)
        x_intermediate = lq.layers.QuantDense(
            units=channels, kernel_quantizer=kernel_quantizer, **layer_kwargs
        )(x_intermediate)
        x_intermediate = GraphConv(adj_matrix)(x_intermediate)

    # Final layer: same as before but with specified number of output labels and softmax
    x_intermediate = input_quantizer()(x_intermediate)
    x_intermediate = tf.keras.layers.Dropout(rate=dropout_rate)(x_intermediate)
    x_intermediate = lq.layers.QuantDense(
        units=output_features,
        kernel_initializer="he_uniform",
        kernel_quantizer=kernel_quantizer,
    )(x_intermediate)
    x_intermediate = GraphConv(adj_matrix)(x_intermediate)
    outputs = tf.keras.layers.Softmax()(x_intermediate)

    model = tf.keras.Model(
        inputs=[node_features, adj_matrix], outputs=outputs, name="BiGCN"
    )

    # Generate models for intermediate layers, used for debugging activations
    intermediate_models = []
    for l in model.layers:
        if isinstance(l, lq.layers.QuantDense):
            intermediate_models.append(
                tf.keras.Model(inputs=node_features, outputs=l.output)
            )
    return model, intermediate_models


def generate_standard_gcn(
    channels: int,
    input_shapes: Tuple[Tuple, Tuple],
    output_features: int,
    dropout_rate: float,
    layers: int = 2,
    activation=tf.keras.layers.ReLU,
    **layer_kwargs
):
    node_features = tf.keras.Input(shape=(input_shapes[0]))
    adj_matrix = tf.keras.layers.Input(shape=(input_shapes[1]))
    x_intermediate = tf.keras.layers.BatchNormalization()(node_features)

    # Intermediate layers: Binary Activation - Dropout - Graph Convolution
    for layer in range(layers - 1):
        x_intermediate = input_quantizer()(x_intermediate)
        x_intermediate = tf.keras.layers.Dropout(rate=dropout_rate)(x_intermediate)
        x_intermediate = tf.keras.layers.Dense(units=channels, **layer_kwargs)(
            x_intermediate
        )
        x_intermediate = GraphConv(adj_matrix)(x_intermediate)

    # Final layer: same as before but with specified number of output labels and softmax
    x_intermediate = input_quantizer()(x_intermediate)
    x_intermediate = tf.keras.layers.Dropout(rate=dropout_rate)(x_intermediate)
    x_intermediate = lq.layers.QuantDense(
        units=output_features,
        kernel_initializer="he_uniform",
        kernel_quantizer=kernel_quantizer,
    )(x_intermediate)
    x_intermediate = GraphConv(adj_matrix)(x_intermediate)
    outputs = tf.keras.layers.Softmax()(x_intermediate)

    model = tf.keras.Model(
        inputs=[node_features, adj_matrix], outputs=outputs, name="BiGCN"
    )

    # Generate models for intermediate layers, used for debugging activations
    intermediate_models = []
    for l in model.layers:
        if isinstance(l, lq.layers.QuantDense):
            intermediate_models.append(
                tf.keras.Model(inputs=node_features, outputs=l.output)
            )
    return model, intermediate_models
