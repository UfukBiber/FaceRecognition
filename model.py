from Inception_Model import build_inception_V4
import tensorflow as tf 
from input import load_ds



def build_model():
    inception_v4 = build_inception_V4()
    img_1 = tf.keras.layers.Input(shape= (299, 299, 3))
    img_2 = tf.keras.layers.Input(shape= (299, 299, 3))
    encoding_1 = inception_v4(img_1)
    encoding_2 = inception_v4(img_2)
    out = tf.keras.layers.subtract()([encoding_1, encoding_2])
    out = tf.keras.layers.Dropout(0.2)(out)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(out)
    return tf.keras.models.Model(inp, out)



train_ds, val_ds = load_ds()
model = build_model()


model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(train_ds, validation_data = val_ds, epochs = 10)
