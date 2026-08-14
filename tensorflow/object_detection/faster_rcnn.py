import tensorflow as tf
import tensorflow_hub as hub

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']


img = tf.io.read_file('image.png')
img = tf.image.decode_jpeg(img, channels=3)
converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

result = detector(converted_img)

scores = result["detection_scores"].numpy()
classes = result["detection_class_entities"].numpy()
boxes = result["detection_boxes"].numpy()

