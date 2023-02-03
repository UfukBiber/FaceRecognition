from Inception_Model import build_inception_V4
import tensorflow as tf 
from input import load_ds



def build_model():
    inception_v4 = build_inception_V4()
    inp = tf.keras.layers.Input(shape= (299, 299, 3))
    out = inception_v4(inp)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dropout(0.2)(out)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(out)
    return tf.keras.models.Model(inp, out)



train_ds, val_ds = load_ds()
model = build_model()


model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(train_ds, validation_data = val_ds, epochs = 10)
