
import tensorflow as tf

LAST_CONV_LAYER = "multiply_6"


def generate_gradcam(model, processed_image, class_index=None):

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        feature_maps, predictions = grad_model(
            processed_image,
            training=False
        )

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = tf.math.divide_no_nan(
        heatmap,
        tf.reduce_max(heatmap)
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224)
    )

    return heatmap.numpy().squeeze()
