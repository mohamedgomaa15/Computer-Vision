from PIL import Image
import cv2
import albumentations as A
import numpy as np


image = cv2.imread("data/IMG_20230227_121951_787.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [[0.504945, 0.566405, 0.920871, 0.867191]]

transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.Rotate(limit=40, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.2, hue=0.2, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
    ], p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=[]))

image_list = [image]
bboxes_list = [bboxes[0]]
for i in range(10):
    augmented = transform(image=image, bboxes=bboxes)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    image_list.append(augmented_image)
    bboxes_list.append(augmented_bboxes[0])

for idx, (img, bbox) in enumerate(zip(image_list, bboxes_list)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.rectangle(img, (int(bbox[0] * img.shape[1]), int(bbox[1] * img.shape[0])), 
                  (int((bbox[0] + bbox[2]) * img.shape[1]), int((bbox[1] + bbox[3]) * img.shape[0])), 
                  (255, 0, 0), 2)
    cv2.imwrite(f"augmented_image_{idx}.jpg", img)