from PIL import Image
import os

src_path = "path/to/src"
dst_path = "path/to/dst"
new_width = 200 


dirs = os.listdir(src_path)
for dir in dirs:
    sub_dirs = os.listdir(src_path + "/" + dir)
    if not os.path.exists(dst_path + "/" + dir):
        os.mkdir(dst_path + "/" + dir)
    for image in sub_dirs:
        print(image)
        img = Image.open(f"{src_path}/{dir}/{image}")

        width, height = img.size
        aspect_ratio = height / width
        new_height = int(aspect_ratio * new_width)

        resized_img = img.resize(
            (new_width, new_height), resample=Image.LANCZOS)

        resized_img.save(f"{dst_path}/{dir}/{image}")