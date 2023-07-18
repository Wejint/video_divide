import cv2
import os

# 设置图像帧目录和输出视频文件名
frames_dir = 'E:\\deep learning\\segment_images_UNet1_video'
# frames_dir = "E:\\data\\dance_video_divide"
output_video = "E:\\data\\output.mp4"

# 获取图像帧列表
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[1].split('.')[0].zfill(4)))
# 读取第一帧以获取图像尺寸信息
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width, channels = first_frame.shape

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 60.0, (width, height))
video_fps = cv2.VideoCapture(os.path.join(frames_dir, frame_files[0])).get(cv2.CAP_PROP_FPS)
# 逐帧写入视频
for frame_file in frame_files:
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    out.write(frame)

# 释放视频写入对象
out.release()
