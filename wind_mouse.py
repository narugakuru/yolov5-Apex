import numpy as np

sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)


def wind_mouse(start_x, start_y, dest_x, dest_y, G_0=9, W_0=3, M_0=15, D_0=12, move_mouse=lambda x, y: None):
    '''
    WindMouse algorithm. Calls the move_mouse kwarg with each new step.
    Released under the terms of the GPLv3 license.
    G_0 - magnitude of the gravitational fornce
    W_0 - magnitude of the wind force fluctuations
    M_0 - maximum step size (velocity clip threshold)
    D_0 - distance where wind behavior changes from random to damped
    start_x 和 start_y：鼠标的初始坐标。

    dest_x 和 dest_y：鼠标要到达的目标坐标。

    G_0：重力的大小。这个力将鼠标吸s引到目标点。较大的 G_0 值将使鼠标更快地到达目标点。

    W_0：风力的大小。风力给鼠标的移动增加了随机性，使得其运动路径不那么可预测。它有助于鼠标探索不同的路径。

    M_0：最大步长（速度剪切阈值）。当鼠标速度超过 M_0 时，速度将被削减。这样做是为了防止鼠标在移动过程中过快超出目标点，保持移动的平滑性。

    D_0：距离在此值之上时，风力行为从随机变为减震。也就是说，当鼠标与目标点的距离大于等于 D_0 时，风力会具有随机性；而当距离小于 D_0 时，风力会被减弱，以保持移动的稳定性。这有助于使鼠标在接近目标点时更加稳定。
    '''
    current_x, current_y = start_x, start_y
    v_x = v_y = W_x = W_y = 0
    while (dist := np.hypot(dest_x - start_x, dest_y - start_y)) >= 1:
        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x / sqrt3 + (2 * np.random.random() - 1) * W_mag / sqrt5
            W_y = W_y / sqrt3 + (2 * np.random.random() - 1) * W_mag / sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = np.random.random() * 3 + 3
            else:
                M_0 /= sqrt5
        v_x += W_x + G_0 * (dest_x - start_x) / dist
        v_y += W_y + G_0 * (dest_y - start_y) / dist
        v_mag = np.hypot(v_x, v_y)
        if v_mag > M_0:
            v_clip = M_0 / 2 + np.random.random() * M_0 / 2
            v_x = (v_x / v_mag) * v_clip
            v_y = (v_y / v_mag) * v_clip
        start_x += v_x
        start_y += v_y
        move_x = int(np.round(start_x))
        move_y = int(np.round(start_y))
        if current_x != move_x or current_y != move_y:
            # This should wait for the mouse polling interval
            move_mouse(current_x := move_x, current_y := move_y)
    return current_x, current_y
