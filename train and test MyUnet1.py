import weijiangtao_net
from PIL import Image
from torch import nn
import torch
import cv2 as cv
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from skimage import data,filters
from torch.utils.data import Dataset
import logging
# 禁用日志输出
logging.disable(logging.CRITICAL)
#数据集
class myDataset(Dataset):
    def __init__(self, root_dir, transform, transform1):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'labels')
        self.image_filenames = os.listdir(self.image_dir)
        self.transform = transform
        self.transform1 = transform1

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.transform(image)
        mask = self.transform1(mask)

        return image, mask
def readData_image(image_dir):


    img_1 = cv.imread(image_dir, cv.IMREAD_UNCHANGED)
    img_1 = cv.resize(img_1, dsize=(400,800))
    return np.transpose(img_1 / 255, (2, 0, 1))
    # image = Image.open(image_dir).convert('RGB')
    #
    #
    # image = transform(image)
    #
    # return image
def readData_mask(image_dir):
    img_1 = cv.imread(image_dir, 0)
    img_1 = cv.resize(img_1, dsize=(400, 800))
    return img_1/255

    # mask = Image.open(image_dir).convert('L')
    #
    #
    # mask = transform1(mask)
    # return mask
def getPicture_image(folder_path):
    image_list = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):

            image_path = os.path.join(folder_path, filename)
            image = readData_image(image_path)
            image_list.append(np.asarray(image, dtype=float))
    return np.array(image_list)
def getPicture_mask(folder_path):
    image_list = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):

            image_path = os.path.join(folder_path, filename)
            image = readData_mask(image_path)
            image_list.append(np.asarray(image, dtype=float))
    return np.array(image_list)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #define of network
    NetWork = weijiangtao_net.UNet1(1)
    NetWork = NetWork.to(device)
    # NetWork = weijiangtao_net.UNet1(1)
    # NetWork.load_state_dict(torch.load('./FinalWork_detect_UNet1.pth'))
    # NetWork = NetWork.to(device)

    # #transform
    # transform = transforms.Compose(
    #     [
    #
    #         transforms.Resize((128, 128)),
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.ToTensor(),
    #
    #     ]
    # )
    # transform1 = transforms.Compose(
    #     [
    #
    #         transforms.Resize((128, 128)),
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.ToTensor(),
    #
    #     ]
    # )
    # 读取训练集和测试集数据
    # train_images_path = "E:\\data\\test\\people\\train\\source"
    # train_labels_path = "E:\\data\\test\\people\\train\\mask"
    # test_images_path = "E:\\data\\test\\people\\test\\source"
    # test_labels_path = "E:\\data\\test\\people\\test\\mask"
    train_images_path = "E:\\data\\object_detect\\train_new\\source"
    train_labels_path = "E:\\data\\object_detect\\train_new\\mask"
    test_images_path = "E:\\data\\object_detect\\test_new\\source"
    test_labels_path = "E:\\data\\object_detect\\test_new\\mask"
    # train_images_path = "E:\\data\\data_new\\data\\train_data\\source"
    # train_labels_path = "E:\\data\\data_new\\data\\train_data\\label"
    # test_images_path = "E:\\data\\data_new\\data\\test_data\\source"
    # test_labels_path = "E:\\data\\data_new\\data\\test_data\\label"
    # train_images_path = "E:\\UNet\\UNet\\data\\train\\image"
    # train_labels_path = "E:\\UNet\\UNet\\data\\train\\label"
    # test_images_path = "E:\\data\\object_detect\\test_new\\source"
    # test_labels_path = "E:\\data\\object_detect\\test_new\\mask"
    # print(getPicture(train_images_path))
    # train_images = np.array(getPicture(train_images_path))
    # train_labels = np.array(getPicture(train_labels_path))
    # test_images = np.array(getPicture(test_images_path))
    # test_labels =  np.array(getPicture(test_labels_path))
    # train_images =torch.from_numpy(train_images.copy()).unsqueeze(1).float()
    # train_labels = torch.from_numpy(train_labels.copy()).long()
    # test_images = torch.from_numpy(test_images.copy()).unsqueeze(1).float()
    # test_labels = torch.from_numpy(test_labels.copy()).long()
    train_images = torch.from_numpy(getPicture_image(train_images_path)).float()
    train_labels = torch.from_numpy(getPicture_mask(train_labels_path)).float()
    test_images =  torch.from_numpy(getPicture_image(test_images_path)).float()
    test_labels =  torch.from_numpy(getPicture_mask(test_labels_path)).float()
    # # 定义数据集对象
    # train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    # test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    # 定义batch size和num_workers
    # 读取训练集和测试集数据
    # train_images_path = "E:\\data\\object_detect\\train\\images"
    # train_labels_path = "E:\\data\\object_detect\\train\\labels"
    # test_images_path = "E:\\data\\object_detect\\test\\images"
    # test_labels_path = "E:\\data\\object_detect\\test\\labels"
    #
    # train_images = torch.from_numpy(getPicture_train(train_images_path)).float()
    # train_labels = torch.from_numpy(getPicture_test(train_labels_path)).float()
    # test_images = torch.from_numpy(getPicture_train(test_images_path)).float()
    # test_labels = torch.from_numpy(getPicture_test(test_labels_path)).float()
    # 定义数据集对象

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    # train_dataset = myDataset("E:\\data\\object_detect\\train", transform, transform1)
    # test_dataset = myDataset("E:\\data\\object_detect\\test", transform, transform1)
    batch_size = 1
    num_workers = 1

    # 创建DataLoader对象
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    # defined loss function
    loss_function = nn.BCELoss()
    loss_function = loss_function.to(device)

    # define optim

    optim = torch.optim.SGD(NetWork.parameters(), lr=1e-4)
    # optim = torch.optim.Adam(NetWork.parameters(),lr = 1e-3)
    # optim = torch.optim.ASGD(NetWork.parameters(),lr = 0.01)
    # optim = torch.optim.Rprop(NetWork.parameters(),lr = 0.01)
    torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)

    # remember loss and acurracy
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    best_train_loss = 1
    # train

    num_epochs = 15
    # train_process

    NetWork.train()
    best_test_acc = 0
    for epoch in range(num_epochs):
        train_times = 0
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        test_loss = 0.0
        test_correct = 0
        test_total = 0
        # process
        for imgs, labels in train_loader:
            imgs = imgs.float()
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = NetWork(imgs)
            outputs = outputs.to(device)

            outputs = nn.Sigmoid()(outputs)
            # get loss

            loss = loss_function(outputs.squeeze(1), labels)
            # optim
            optim.zero_grad()
            loss.backward()
            optim.step()


            train_loss += loss.item() * labels.size(0)
            # print(loss.item())
            # print(labels.size(0))
            # print(outputs)
            # print(torch.max(outputs.data,1))
            train_total += labels.size(0)


            binary_outputs = (outputs > 0.5).float()

            # 将二进制图像转换为numpy数组
            binary_outputs = binary_outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            # 计算准确率
            accuracy = np.mean(binary_outputs == labels)
            train_correct+=accuracy*batch_size
        # 测试模式
        NetWork.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.float()
                images = images.to(device)
                labels = labels.to(device)
                outputs = NetWork(images)
                outputs = outputs.to(device)
                outputs = nn.Sigmoid()(outputs)
                loss = loss_function(outputs.squeeze(1), labels)

                test_loss += loss.item() * labels.size(0)
                test_total+=labels.size(0)


                binary_outputs = (outputs > 0.5).float()

                # 将二进制图像转换为numpy数组
                binary_outputs = binary_outputs.cpu().numpy()
                labels = labels.cpu().numpy()

                # 计算准确率
                accuracy = np.mean(binary_outputs == labels)

                test_correct += accuracy * batch_size

        # 记录训练过程中的训练精度、测试精度、训练Loss和测试Loss
        train_acc = 100.0 * train_correct / train_total
        test_acc = 100.0 * test_correct / test_total
        train_loss /= train_total
        test_loss /= test_total
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        # 输出训练过程中的训练精度、测试精度、训练Loss和测试Loss

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%,Test Loss{:.4f},Test Acc:{:.2f}%'
              .format(epoch + 1, num_epochs, train_loss, train_acc,test_loss,test_acc))

        # 保存精度最高的模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(NetWork.state_dict(), 'FinalWork_detect_UNet2.pth')
            print("save")

            # 输出最高精度
    print('Best Test Acc: {:.2f}%'.format(best_test_acc))

    # 绘制训练精度和测试精度
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.legend()
    plt.show()
    plt.plot(train_loss_list,label = 'Train Loss')
    plt.plot(test_loss_list,label = 'Test Loss')
    plt.legend()
    plt.show()