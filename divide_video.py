import cv2
import os
# 读取视频文件
video_path = "E:\\data\\dance.mp4"
cap = cv2.VideoCapture(video_path)

# 创建一个用于保存图像帧的目录
output_dir = "E:\\data\\dance_video_divide"
os.makedirs(output_dir, exist_ok=True)

# 循环读取视频帧并保存为图像
frame_count = 0
while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 如果视频帧读取失败，则退出循环
    if not ret:
        break

    # 保存当前帧为图像文件
    frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
    cv2.imwrite(frame_path, frame)

    frame_count += 1

# 释放视频读取对象
cap.release()
