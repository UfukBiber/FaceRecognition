import tensorflow as tf 



def conv_with_batch_normalization(inp, filter_size, kernel_size, strides, padding):
    out = tf.keras.layers.Conv2D(filter_size, kernel_size, strides = strides, padding = padding)(inp)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    return out




def stem_block(inp_layer):

    out = conv_with_batch_normalization(inp_layer, 32, 3, 2, padding="VALID")
    out = conv_with_batch_normalization(out, 32, 3, 1, padding="VALID")
    out = conv_with_batch_normalization(out, 64, 3, 1, padding="SAME")
  

    out_1 = conv_with_batch_normalization(out, 96, 3, 2, padding = "VALID")
    out_2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides = (2, 2), padding = "VALID")(out)

    out = tf.keras.layers.Concatenate()([out_1, out_2])
  
    out_1 = conv_with_batch_normalization(out, 64, 1, padding="SAME", strides = 1)
    out_1 = conv_with_batch_normalization(out_1, 64, (7, 1), 1, "SAME")
    out_1 = conv_with_batch_normalization(out_1, 64, (1, 7), strides = 1, padding = "SAME")
    out_1 = conv_with_batch_normalization(out_1, 96, 3, 1, padding = "VALID")

    out_2 = conv_with_batch_normalization(out, 64, 1, 1, padding = "SAME")
    out_2 = conv_with_batch_normalization(out_2, 96, 3, 1, padding = "VALID")
   
    out = tf.keras.layers.Concatenate()([out_1, out_2])

    
    out_1 = conv_with_batch_normalization(out, 192, 3, 2, padding = "VALID")
    out_2 = tf.keras.layers.MaxPooling2D(pool_size = 3, strides = 2, padding = "VALID")(out)
    
    out = tf.keras.layers.Concatenate()([out_1, out_2])
    
    return out 


def Inception_A(inp):
    out_1 = tf.keras.layers.AveragePooling2D(pool_size=3, strides = 1, padding = "SAME")(inp)
    out_1 = conv_with_batch_normalization(out_1, 96, 1, 1, padding = "SAME")

    out_2 = conv_with_batch_normalization(inp, 96, 1, 1, "SAME")

    out_3 = conv_with_batch_normalization(inp, 64, 1, 1, "SAME")
    out_3 = conv_with_batch_normalization(out_3, 96, 3, 1, "SAME")

    out_4 = conv_with_batch_normalization(inp, 64, 1, 1, "SAME")
    out_4 = conv_with_batch_normalization(out_4, 96, 3, 1, "SAME")
    out_4 = conv_with_batch_normalization(out_4, 96, 3, 1, "SAME")

    out = tf.keras.layers.Concatenate()([out_1, out_2, out_3, out_4])

    return out 

def Inception_B(inp):
    out_1 = tf.keras.layers.AveragePooling2D(pool_size=3, strides = 1, padding = "SAME")(inp)
    out_1 = conv_with_batch_normalization(out_1, 128, 1, 1, padding = "SAME")

    out_2 = conv_with_batch_normalization(inp, 384, 1, 1, "SAME")

    out_3 = conv_with_batch_normalization(inp, 192, 1, 1, "SAME")
    out_3 = conv_with_batch_normalization(out_3, 224, (1, 7), 1, "SAME")
    out_3 = conv_with_batch_normalization(out_3, 256, (7, 1), 1, "SAME")

    out_4 = conv_with_batch_normalization(inp, 192, 1, 1, "SAME")
    out_4 = conv_with_batch_normalization(out_4, 192, (1, 7), 1, "SAME")
    out_4 = conv_with_batch_normalization(out_4, 224, (7, 1), 1, "SAME")
    out_4 = conv_with_batch_normalization(out_4, 224, (1, 7), 1, "SAME")
    out_4 = conv_with_batch_normalization(out_4, 256, (7, 1), 1, "SAME")

    out = tf.keras.layers.Concatenate()([out_1, out_2, out_3, out_4])

    return out 


def Inception_C(inp):
    out_1 = tf.keras.layers.AveragePooling2D(pool_size=3, strides = 1, padding = "SAME")(inp)
    out_1 = conv_with_batch_normalization(out_1, 128, 1, 1, padding = "SAME")

    out_2 = conv_with_batch_normalization(inp, 256, 1, 1, "SAME")

    out_3 = conv_with_batch_normalization(inp, 384, 1, 1, "SAME")
    out_31 = conv_with_batch_normalization(out_3, 256, (1, 3), 1, "SAME")
    out_32 = conv_with_batch_normalization(out_3, 256, (3, 1), 1, "SAME")

    out_4 = conv_with_batch_normalization(inp, 384, 1, 1, "SAME")
    out_4 = conv_with_batch_normalization(out_4, 448, (1, 3), 1, "SAME")
    out_4 = conv_with_batch_normalization(out_4, 512, (3, 1), 1, "SAME")
    out_41 = conv_with_batch_normalization(out_4, 256, (3, 1), 1, "SAME")
    out_42 = conv_with_batch_normalization(out_4, 256, (1, 3), 1, "SAME")

    out = tf.keras.layers.Concatenate()([out_1, out_2, out_31, out_32, out_41, out_42])

    return out 


def Reduction_A(inp):
    out_1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides = 2, padding = "VALID")(inp)

    out_2 = conv_with_batch_normalization(inp, 384, 3, 2, "VALID")

    out_3 = conv_with_batch_normalization(inp, 192, 1, 1, "SAME")
    out_3 = conv_with_batch_normalization(out_3, 224, 3, 1, "SAME")
    out_3 = conv_with_batch_normalization(out_3, 256, 3, 2, "VALID")

  

    out = tf.keras.layers.Concatenate()([out_1, out_2, out_3])

    return out 



def Reduction_B(inp):
    out_1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides = 2, padding = "VALID")(inp)

    out_2 = conv_with_batch_normalization(inp, 192, 1, 1, "SAME")
    out_2 = conv_with_batch_normalization(out_2, 192, 3, 2, "VALID")

    out_3 = conv_with_batch_normalization(inp, 256, 1, 1, "SAME")
    out_3 = conv_with_batch_normalization(out_3, 256, (1, 7), 1, "SAME")
    out_3 = conv_with_batch_normalization(out_3, 320, (7, 1), 1, "SAME")
    out_3 = conv_with_batch_normalization(out_3, 320, 3, 2, "VALID")

  

    out = tf.keras.layers.Concatenate()([out_1, out_2, out_3])

    return out 


def build_inception_V4():
    inp = tf.keras.layers.Input(shape = (299, 299, 3), name = "Inception_V4_Input")

    out = stem_block(inp)

    out = Inception_A(out)
    out = Inception_A(out)
    out = Reduction_A(out)
    out = Inception_B(out)
    out = Inception_B(out)
    out = Inception_B(out)
    out = Reduction_B(out)
    out = Inception_C(out)
    out = Inception_C(out)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dropout(0.2)(out)
    out = tf.keras.layers.Dense(1024, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.2)(out)
    out = tf.keras.layers.Dense(128)(out)

    return tf.keras.models.Model(inp, out, name = "Inception_V4")  
