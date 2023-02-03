import os 
import matplotlib.pyplot as plt 
import random
import tensorflow as tf 
import cv2

# IDENTITIES_PATH = "Identities.txt"

# VAL_RATIO = 0.1

# N_IMAGES = 10
SEED = 1234 

# random.seed(SEED)



# def prepare_identities():
#     identity_to_img_path = dict()
#     with open(IDENTITIES_PATH, "r") as f:
#         lines = f.readlines()
#         f.close()
#     for line in lines:
#         line = line.split(" ")
#         identity = int(line[1].replace("\n", ""))
#         identity_to_img_path[identity] = identity_to_img_path.get(identity, []) + [os.path.join(IMAGES_PATHS, line[0])]
#     return identity_to_img_path


# def prepare_data():
#     identity_to_images = prepare_identities()
#     inp = []
#     out = []
#     keys = list(identity_to_images.keys())
#     for i in range(len(keys)):
#         if len(identity_to_images[keys[i]]) < N_IMAGES:
#             continue
#         true_image_batch = []
#         false_image_batch = []
#         j = 0
#         while j < N_IMAGES:
#             img_1 = random.choice(identity_to_images[keys[i]])
#             img_2 = random.choice(identity_to_images[keys[i]])
#             if [img_1, img_2] not in true_image_batch:
#                 true_image_batch.append((img_1, img_2))
#                 j+= 1 
#                 out.append(1)
#         j = 0
#         while j < N_IMAGES:
#             key_2 = random.choice(keys)
#             if key_2 == keys[i]:
#                 continue
#             img_1 = random.choice(identity_to_images[keys[i]])
#             img_2 = random.choice(identity_to_images[key_2])
#             if [img_1, img_2] not in false_image_batch:
#                 false_image_batch.append((img_1, img_2))
#                 j+= 1 
#                 out.append(0)
#         print("%i / %i"%(i, len(keys)), end = "\r")
#         inp.extend(true_image_batch)
#         inp.extend(false_image_batch)
#     print()
#     return inp, out 




# def load_ds(img_size, batch_size):
#     inp, out = prepare_data()

#     random.seed(SEED)
#     random.shuffle(inp)
#     random.seed(SEED)
#     random.shuffle(out)

#     train_length = int(len(inp) * (1 - VAL_RATIO))
#     train_inp = inp[:train_length]
#     train_out = out[:train_length]
#     val_inp = inp[train_length:]
#     val_out = out[train_length:]

#     train_ds = tf.data.Dataset.from_tensor_slices((train_inp, train_out))
#     train_ds = train_ds.map(lambda paths, output : ((load_img(paths[0], img_size), load_img(paths[1], img_size)), output), num_parallel_calls=tf.data.AUTOTUNE)
#     train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

#     val_ds = tf.data.Dataset.from_tensor_slices((val_inp, val_out))
#     val_ds = val_ds.map(lambda paths, output : ((load_img(paths[0], img_size), load_img(paths[1], img_size)), output), num_parallel_calls=tf.data.AUTOTUNE)
#     val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return train_ds, val_ds



N_IMAGES_TRAIN = 32000
N_IMAGES_VAL = 2000


IMAGE_NET_PATH = "/home/ufuk/Desktop/imagenet-mini/train"
IMAGES_PATHS = "/home/ufuk/Desktop/img_align_celeba"




def load_img(img_path:str, image_size:list):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, size = image_size, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    return img


def load_image_net():
    folder_paths = [os.path.join(IMAGE_NET_PATH, folder_path) for folder_path in os.listdir(IMAGE_NET_PATH)]
    images_paths = []
    for folder_path in folder_paths:
        images_paths_f = [os.path.join(folder_path, image_path) for image_path in os.listdir(folder_path)]
        images_paths.extend(images_paths_f)
    return images_paths


def load_celeba():
    images_paths = [os.path.join(IMAGES_PATHS, image_path) for image_path in os.listdir(IMAGES_PATHS)]
    return images_paths




def show_imgs(celeba_paths, image_net_paths, n = 5):
    fig, ax = plt.subplots(2, n)
    for i in range(n):
        ax[0, i].imshow(plt.imread(celeba_paths[i]))
        ax[1, i].imshow(plt.imread(image_net_paths[i]))
    plt.show()



def load_ds():
    image_net_paths = load_image_net()
    celeba_paths = load_celeba()

    train_inp = image_net_paths[:N_IMAGES_TRAIN] + celeba_paths[:N_IMAGES_TRAIN]
    train_out = [0 for i in range(N_IMAGES_TRAIN)] + [1 for i in range(N_IMAGES_TRAIN)]

    random.seed(SEED)
    random.shuffle(train_inp)
    random.seed(SEED)
    random.shuffle(train_out)


    val_inp = image_net_paths[N_IMAGES_TRAIN:(N_IMAGES_TRAIN + N_IMAGES_VAL)] + celeba_paths[N_IMAGES_TRAIN:(N_IMAGES_TRAIN + N_IMAGES_VAL)]
    val_out = [0 for i in range(N_IMAGES_VAL)] + [1 for i in range(N_IMAGES_VAL)]

    random.seed(SEED)
    random.shuffle(val_inp)
    random.seed(SEED)
    random.shuffle(val_out)

    train_ds = tf.data.Dataset.from_tensor_slices((train_inp, train_out))
    val_ds = tf.data.Dataset.from_tensor_slices((val_inp, val_out))

    train_ds = train_ds.map(lambda inp, out:(load_img(inp, (299, 299)), out)).shuffle(512)
    val_ds = val_ds.map(lambda inp, out:(load_img(inp, (299, 299)), out)).shuffle(512)
    train_ds = train_ds.batch(32, drop_remainder = True)
    val_ds = val_ds.batch(32, drop_remainder = True)

    return train_ds, val_ds







