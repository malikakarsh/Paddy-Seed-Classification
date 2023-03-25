import os
import numpy as np
from PIL import Image, ImageEnhance
import shutil

src_path = "path/to/src"
dst_path = "path/to/dst"


num_images = 1000 # Total number of images ( num_images = images present + augmented images )
brightness_range = (0.5, 1.5)
contrast_range = (0.5, 1.5)
sharpness_range = (0.5, 1.5)
hue_range = (-0.5, 0.5)
dirs = os.listdir(src_path)


def rep_image(src, name, cnt, input_image, rep=True):
    transformed_image = input_image.copy()
    brightness_factor = np.random.uniform(*brightness_range)
    contrast_factor = np.random.uniform(*contrast_range)
    sharpness_factor = np.random.uniform(*sharpness_range)
    hue_factor = np.random.uniform(*hue_range)
    transformed_image = ImageEnhance.Brightness(
        transformed_image).enhance(brightness_factor)
    transformed_image = ImageEnhance.Contrast(
        transformed_image).enhance(contrast_factor)
    transformed_image = ImageEnhance.Sharpness(
        transformed_image).enhance(sharpness_factor)
    transformed_image = ImageEnhance.Color(
        transformed_image).enhance(1 + hue_factor)

    if not rep:
        output_filename = os.path.join(
            dst_path + "/" + src, f"{name}_rem_{cnt}.jpg")
    else:
        output_filename = os.path.join(
            dst_path + "/" + src, f"{name}_rep_{cnt}.jpg")
    transformed_image.save(output_filename)
    print(f"saved: {output_filename}")


for par_dir in dirs:
    sub_dirs = os.listdir(src_path + "/" + par_dir)
    if not os.path.exists(dst_path + "/" + par_dir):
        os.mkdir(dst_path + "/" + par_dir)
    for dir in sub_dirs:
        new_dir_path = src_path + "/" + par_dir + "/" + dir

        _, _, files = next(os.walk(new_dir_path))
        count = len(files)
        rem = num_images % count  # number of images to be generated

        # number of images to be generated from each image
        rep_each_img = int(num_images / count) - 1
        for image in os.listdir(new_dir_path):
            img_path = new_dir_path + "/" + image
            print(image)
            shutil.copy(img_path, dst_path + "/" + par_dir)
            input_image = Image.open(img_path)

            # Generate the replica of each image
            for i in range(rep_each_img):
                print(f"Image name: {image.split('.')[0]} {i}")
                rep_image(par_dir, image.split('.')[0], i, input_image)

        # Generate the remaining images
        for i in range(rem):
            input_image = Image.open(
                new_dir_path + "/" + os.listdir(new_dir_path)[i])
            rep_image(par_dir, os.listdir(new_dir_path)[i].split('.')[
                0], i, input_image, False)
