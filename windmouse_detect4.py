import ctypes
import math
import threading
import time
import pynput.mouse
import numpy as np
import torch

from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import smart_inference_mode
from wind_mouse import wind_mouse
from ScreenShot import screenshot, area, half_area, ScreenX, ScreenY
from SendInput import *
from pynput.mouse import Listener


# 位置变量0~0.5直接调整，越高越接近头 头:0.45(容易封号) 胸0.2
head = 0.3
x1 = False
is_x2_pressed = False


def mouse_listener():
    with Listener(on_click=mouse_click) as listener:
        listener.join()


def mouse_click(x, y, button, pressed):
    """ 单击x1改变值，True启动右键瞄准，False关闭 """
    global is_x2_pressed
    global head
    global x1
    if x1 == True:
        if pressed and button == pynput.mouse.Button.right:
            head = 0.3
            is_x2_pressed = True
            print("yes")
        elif not pressed and button == pynput.mouse.Button.right:
            is_x2_pressed = False

    if x1 == True and pressed and button == pynput.mouse.Button.x1:
        x1 = False
        print("x1:", x1, time.time())
        return
    elif x1 == False and pressed and button == pynput.mouse.Button.x1:
        x1 = True
        print("x1:", x1, time.time())


def process_im(model):
    im = letterbox(im, (area, area), stride=32, auto=True)[
        0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im


def plan_execute(target_X, target_Y):
    """ wind算法规划路径，并执行move_xy """
    points = []
    # wind算法规划移动路径，M步长，G加速度，W风力，D随机风力
    wind_mouse(0, 0, target_X, target_Y, G_0=16, W_0=2,
               M_0=20, D_0=35,
               move_mouse=lambda x, y: points.append([x, y]))
    pre_x = 0
    pre_y = 0
    # 执行移动
    for x, y in points:
        move_x = x - pre_x
        move_y = y - pre_y
        mouse_xy(move_x, move_y)
        pre_x = x
        pre_y = y

    time.sleep(0.02)  # 主动睡眠，防止推理过快,鼠标移动相同的两次


@smart_inference_mode()
def run():
    # Load model
    device = torch.device('cuda:0')
    model = DetectMultiBackend(
        weights='./weights/apex-7000-n.pt', device=device, dnn=False, data=False, fp16=True)

    # 读取图片
    while True:
        # 处理图片
        im = screenshot()
        im0 = im
        im = process_im(model)
        # 推理
        pred = model(im, augment=False, visualize=False)
        # 非极大值抑制
        pred = non_max_suppression(
            pred, conf_thres=0.6, iou_thres=0.45, classes=0, max_det=10)
        # 处理推理内容
        for i, det in enumerate(pred):
            # 画框
            # annotator = Annotator(im0, line_width=2)
            if len(det):
                distance_list = []
                target_list = []
                # 将转换后的图片画框结果转换成原图上的结果
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], im0.shape).round()
                # 处理推理出来每个目标的信息cls是类别的意思
                for *xyxy, conf, cls in reversed(det):
                    # 将xyxy(左上角+右下角bwb)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                    # 坐标是相对与识别框左上角为原点的坐标
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                            ).view(-1).tolist()  # normalized xywh

                    X = xywh[0] - half_area
                    Y = xywh[1] - half_area
                    distance = math.sqrt(X ** 2 + Y ** 2)

                    # annotator.box_label(xyxy, label=f'[{int(cls)}Distance:{round(distance, 2)}]',
                    #                     color=(34, 139, 34),
                    #                     txt_color=(0, 191, 255))
                    distance_list.append(distance)
                    target_list.append(xywh)

                target_info = target_list[distance_list.index(
                    min(distance_list))]
                target_X = int(target_info[0] - half_area)
                # 头部偏移
                target_Y = int(target_info[1] -
                               half_area - target_info[3]*head)

                # 加个距离非零判断 防止抖动
                if is_x2_pressed and (abs(target_X) > 3 or abs(target_Y) > 3):
                    plan_execute(target_X, target_Y)

            # im0 = annotator.result()
            # cv2.imshow('window', im0)
            # cv2.waitKey(1)


if __name__ == "__main__":
    threading.Thread(target=mouse_listener).start()
    run()
