import torch
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import cv2 as cv
import numpy as np
import os
import weijiangtao_net
import torch.nn
def readData_train(image_dir):


    img_1 = cv.imread(image_dir, cv.IMREAD_UNCHANGED)
    img_1 = cv.resize(img_1, dsize=(160,240))
    return np.transpose(img_1 / 255, (2, 0, 1))
    # image = Image.open(image_dir).convert('RGB')
    #
    #
    # image = transform(image)
    #
    # return image

def readData_test(image_dir):
    img_1 = cv.imread(image_dir, 0)
    img_1 = cv.resize(img_1, dsize=(800, 1200))
    return img_1/255

    # mask = Image.open(image_dir).convert('L')
    #
    #
    # mask = transform1(mask)
    # return mask
def getPicture_train(folder_path):
    image_list = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):

            image_path = os.path.join(folder_path, filename)
            image = readData_train(image_path)
            image_list.append(np.asarray(image, dtype=float))
    return np.array(image_list)
def getPicture_test(folder_path):
    image_list = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):

            image_path = os.path.join(folder_path, filename)
            image = readData_test(image_path)
            image_list.append(np.asarray(image, dtype=float))
    return np.array(image_list)
def combine_images_and_save(images, output_path):
    # 将图像拼接在一起
    combined_image = torch.cat(images, dim=1)

    # 将拼接后的张量转换为PIL图像
    combined_image_pil = transforms.ToPILImage()(combined_image)

    # 保存拼接后的图像
    combined_image_pil.save(output_path)
if __name__ == '__main__':
    # torch.multiprocessing.freeze_support()
    # # device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # #define of network
    # NetWork = weijiangtao_net.MyUNet(1)
    # NetWork.load_state_dict(torch.load('./FinalWork_detect_MyUNet.pth'))
    # NetWork = NetWork.to(device)
    #
    # test_images_path = "E:\\data\\dance_video_divide"
    # test_labels_path = "E:\\data\\object_detect\\test_new\\mask"
    # test_images =  torch.from_numpy(getPicture_train(test_images_path)).float()
    # test_labels =  torch.from_numpy(getPicture_test(test_labels_path)).float()
    #
    #
    #
    # test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    #
    # batch_size = 1
    # num_workers = 1
    #
    # # 创建DataLoader对象
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
    #                                           num_workers=num_workers)
    # # 测试模式
    # NetWork.eval()
    # i = 0
    # os.makedirs("segment_images_UNet1", exist_ok=True)
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images = images.float()#images(1,3,800,400)
    #         images = images.to(device)
    #         labels = labels.to(device)#labels(1,800,400)
    #         labels_save = labels
    #         outputs = NetWork(images)
    #         outputs = outputs.to(device)
    #         outputs = torch.nn.Sigmoid()(outputs)
    #         binary_outputs = (outputs > 0.5).float()#binary_outputs(1,1,800,400)
    #         seg_imgs = labels.repeat( 3, 1, 1)
    #         image_save = images[0]
    #         predict = binary_outputs.squeeze(0).repeat(3,1,1)
    #
    #         imgss = torch.stack([image_save, seg_imgs, predict], dim=0)
    #
    #         save_image(imgss, "segment_images_UNet1_video/{}_img.jpg".format(i))
    #         i+=1
    torch.multiprocessing.freeze_support()
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # define of network
    NetWork = weijiangtao_net.UNet1(1)
    NetWork.load_state_dict(torch.load('./FinalWork_detect_UNet1_video.pth'))
    NetWork = NetWork.to(device)
    frames_dir = "E:\\data\\dance_video_divide"
    # 获取图像帧列表
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[1].split('.')[0].zfill(4)))

    # 逐帧写入视频
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
    # 测试模式
    i = 0
    NetWork.eval()
    with torch.no_grad():
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            images = torch.from_numpy(readData_train(frame_path)).float()
            image =  images.unsqueeze(0)
            image = image.float()  # images(1,3,800,400)
            image = image.to(device)

            outputs = NetWork(image)

            outputs = outputs.to(device)
            outputs = torch.nn.Sigmoid()(outputs)
            binary_outputs = (outputs > 0.5).float()  # binary_outputs(1,1,800,400)

            save_image(binary_outputs, "segment_images_UNet1_video/frame_{}.jpg".format(i))
            i += 1