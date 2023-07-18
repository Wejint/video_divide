from PIL import Image
import os

def convert_jpg_to_png(input_folder, output_folder):
    # 检查输出文件夹是否存在，若不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            # 打开jpg文件
            img = Image.open(os.path.join(input_folder, filename))
            # 将jpg文件转换为png文件
            new_filename = filename.replace(".jpg", ".png")
            # 保存png文件到输出文件夹
            img.save(os.path.join(output_folder, new_filename), "PNG")

    print("转换完成！")
from PIL import Image
import os

def convert_to_black_and_white(image_path):
    image = Image.open(image_path).convert("L")
    image_data = image.load()
    width, height = image.size

    for x in range(width):
        for y in range(height):
            pixel = image_data[x, y]
            if pixel != 0:
                image_data[x, y] = 255

    return image

def convert_folder_to_black_and_white(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            new_image = convert_to_black_and_white(image_path)
            new_image.save(image_path)


convert_folder_to_black_and_white('E:\\data\\dance_train_and_test1\\test\\mask')
