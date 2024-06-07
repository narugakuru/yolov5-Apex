import math

from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.dataloaders import (
    IMG_FORMATS,
    VID_FORMATS,
    LoadImages,
    LoadScreenshots,
    LoadStreams,
)
from models.common import DetectMultiBackend
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import argparse
import csv
import os
import platform
import sys
from utils.augmentations import letterbox
from pathlib import Path
import numpy as np
import torch
import time
from ScreenShot import screenshot
from SendInput import mouse_xy


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def load_mode():
    device = torch.device("cuda:0")
    model = DetectMultiBackend(
        weights="./weights/yolov5n.pt", device=device, dnn=False, data=False, fp16=True
    )
    return device, model


def get_im():
    # 读取图片
    # im = cv2.imread('data/images/bus.jpg')
    im = screenshot()
    # 处理图片
    im = letterbox(im, (640, 640), stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)

    # 将图片转换为tensor
    im = torch.from_numpy(im).to(model.device)
    # 将图片转换为half类型
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im


@smart_inference_mode()
def run(device, model):
    # Load model
    while True:
        # 读取图片
        im = get_im()
        im0 = im
        # 推理
        pred = model(im, augment=False, visualize=False)
        # 只检测0：person
        pred = non_max_suppression(
            pred, conf_thres=0.6, iou_thres=0.45, classes=0, max_det=10
        )

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # 画框
            annotator = Annotator(im0, line_width=1)

            if len(det):
                # Rescale boxes from img_size to im0 size
                distance_list = []
                target_list = []
                # 画框转化为原图
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):  # 处理每个目标的信息
                    # line = (cls, *xywh, conf)  # label format
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    )  # normalized xywh
                    # print(xywh, line)
                    X = int(xywh[0] - 320)
                    Y = int(xywh[1] - 320)
                    distance = math.sqrt(X**2 + Y**2)
                    annotator.box_label(
                        xyxy,
                        label=f'[{int(cls)}Distance:{round(distance,2)}]',
                        color=(99, 88, 201),
                        txt_color=(201, 88, 99),
                    )
                    # 存储每个目标的信息
                    distance_list.append(distance)
                    target_list.append(xywh)

                target_info = target_list[distance_list.index(
                    min(distance_list))]
                x, y = int(target_info[0], target_info[1])
                # print(target_info)
                mouse_xy(x, y)

            im0 = annotator.result()
            cv2.imshow("window", im0)
            cv2.waitKey(1)


if __name__ == "__main__":
    device, model = load_mode()
    run(device, model)
