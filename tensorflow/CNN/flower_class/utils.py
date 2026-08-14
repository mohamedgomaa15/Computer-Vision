import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pathlib
import cv2
import numpy as np
from matplotlib import pyplot as plt


DATA_DIR = "datasets/flower_photos"
data_dir = pathlib.Path(DATA_DIR)

# Create a tf.data.Dataset from the image file paths
DATASET = tf.data.Dataset.list_files(str(data_dir/'*/*/*.jpg'))
# Set the batch size and prefetch the data for performance
AUTOTUNE = tf.data.AUTOTUNE
# convert the string labels to integer labels using StringLookup
LABEL_LOOKUP = tf.keras.layers.StringLookup(
    vocabulary=['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips'],
    mask_token=None,
    num_oov_indices=0
)
NUM_ClASSES = len(LABEL_LOOKUP.get_vocabulary())

# Get the file paths for each class (flower type)
daisy = list(data_dir.glob('*/daisy/*'))[:200]
dandelion = list(data_dir.glob('*/dandelion/*'))[:200]
roses = list(data_dir.glob('*/roses/*'))[:200]
sunflowers = list(data_dir.glob('*/sunflowers/*'))[:200]
tulips = list(data_dir.glob('*/tulips/*'))[:200]

flowers_images_dict = {
    'roses': roses,
    'daisy': daisy,
    'dandelion': dandelion,
    'sunflowers': sunflowers,
    'tulips': tulips,
}

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}

DATA_AUGMENTATION = keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomZoom(0.4),
])


def download_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)
    DATA_DIR = data_dir

def num_images_classes():
    num_images = len(list(data_dir.glob('*/*/*.jpg')))
    num_clases = len(list(data_dir.glob('*/*'))) - 1
    print(num_images, num_clases)

# Prepare the data for training and testing with cv2 for processing and sklearn for spliting
def prepare_data():
    X, y = [], []
    for flower_name, images in flowers_images_dict.items():
        for image in images:
            img = cv2.imread(str(image))
            resized_img = cv2.resize(img,(180,180))
            X.append(resized_img)
            y.append(flowers_labels_dict[flower_name])
    X = np.array(X)
    y = np.array(y)

    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def show_image_for_each_class():
    ax = plt.figure(figsize=(10, 10))
    def imshow(image_path, class_name, num):
        plt.subplot(3, 2, num)
        plt.imshow(plt.imread(str(image_path)))
        plt.axis('off')
        plt.title(class_name)

    imshow(roses[1], 'roses', 1)
    imshow(daisy[0], 'daisy', 2)
    imshow(dandelion[2], 'dandelion', 3)
    imshow(sunflowers[3], 'sunflowers', 4)
    imshow(tulips[4], 'tulips', 5)
    
    plt.show()


dataset = tf.data.Dataset.list_files(str(data_dir/'*/*/*.jpg'))
AUTOTUNE = tf.data.AUTOTUNE

# split the dataset into train, test and validation sets
def split_dataset(dataset, train_size=0.8, test_size=0.1):
    dataset = dataset.shuffle(buffer_size=1000, seed=42)
    dataset_size = len(list(dataset))
    train_dataset = dataset.take(int(train_size * dataset_size))
    test_dataset = dataset.skip(int(train_size * dataset_size)).take(int(test_size * dataset_size))
    val_dataset = dataset.skip(int((train_size + test_size) * dataset_size))
    return train_dataset, test_dataset, val_dataset


# get the label and image from the file path, decode the image and return the image and label
def get_label_and_image(file_path):
    label = tf.strings.split(file_path, os.path.sep)[-2]
    label = LABEL_LOOKUP(label)
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, label

def resize_rescale(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [160, 160])
    image = image / 255.0
    return image

def apply_augmentation_(image, training=True):
    return DATA_AUGMENTATION(image, training=training)

def prepare_dataset_pipeline(dt, batch_size=32, shuffle=False, augment=False):
    dt = dt.map(lambda x: get_label_and_image(x), num_parallel_calls=AUTOTUNE)
    dt = dt.map(lambda x, y: (resize_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        dt = dt.shuffle(1000)
    batch_dt = dt.batch(batch_size)

    if augment:
        batch_dt = batch_dt.map(lambda x, y: (apply_augmentation_(x), y), num_parallel_calls=AUTOTUNE)

    return batch_dt.prefetch(buffer_size=AUTOTUNE) 

def loss_accuracy_plot(history, val=True):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')

    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()



