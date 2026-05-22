# model.py - Configuration: MobileNetV2_Light for Client 25
import tensorflow as tf

def build_model(num_classes=3):
    from tensorflow.keras.applications import MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0012),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model
