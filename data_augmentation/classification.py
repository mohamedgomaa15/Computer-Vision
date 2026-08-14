from PIL import Image
import cv2
import albumentations as A
import numpy as np

image = Image.open("data/IMG_20230227_121951_787.jpg")

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
])

image_list = [image]
image = np.array(image)
for i in range(10):
    augmented = transform(image=image)
    augmented_image = augmented['image']
    image_list.append(augmented_image)


for idx, img in enumerate(image_list):
    img = np.array(img)
    cv2.imwrite(f"augmented_image_{idx}.jpg", img)

