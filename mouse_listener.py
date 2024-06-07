import pynput.mouse
from pynput.mouse import Listener


# 位置变量0~0.5直接调整，越高越接近头 头:0.45(容易封号) 胸0.2
head = 0.3
x1 = False
x2 = False


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
            head = 0.35
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
