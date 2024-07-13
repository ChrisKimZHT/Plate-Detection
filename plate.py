# 导入cv相关库
import os.path

import cv2
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
# 导入依赖包
import hyperlpr3 as lpr3
from tqdm import tqdm


def draw_plate_on_image(img, box, text):
    x1, y1, x2, y2 = box
    img_crop = img[y1:y2, x1:x2].copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    # cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (255, 0, 0), -1)
    # cv2.putText(img, text, (x1 + 1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img,img_crop


num_frame = 0
id = 0
imgPlateBox_root = ".\\imgPlateBox"

cap = cv2.VideoCapture('my_test.mp4')
# 实例化识别对象
catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
# 读取图片


frame_num=0
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID' 等其他编码器
# video = cv2.VideoWriter(os.path.join(video_save_root,VideoName), fourcc, fps, (width, height))
images = []
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 使用 tqdm 显示进度条
with open("plate.txt","w") as f:

    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = catcher(frame)
            for code, confidence, type_idx, box in results:
                # 解析数据并绘制
                if(confidence<0.85):
                    continue
                f.write(f'{num_frame}'+'\t'+code+' '+str(box)+" "+str(confidence)+"\n")
                text = f"{code} - {confidence:.2f}"
                img, img_crop = draw_plate_on_image(frame, box, text)
                # cv2.imshow("image", frame)
                cv2.imwrite(os.path.join(imgPlateBox_root,f"{id}"+".jpg"),img)
                cv2.imwrite(os.path.join(imgPlateBox_root, f"{id}_crop" + ".jpg"), img_crop)
                id+=1
            num_frame+=1
            pbar.update(1)
f.close()
# 显示检测结果
# cv2.imwrite("result.jpg", image)