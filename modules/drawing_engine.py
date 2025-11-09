"""
绘画引擎模块
包含画布校准和三种绘画方法
"""

import time
import random
import numpy as np
import cv2
import pydirectinput
import keyboard
from pynput import mouse
import ctypes

pydirectinput.PAUSE = 0.005


# --- DPI 设置 ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass


def get_dpi_scale():
    """获取Windows DPI缩放比例"""
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        scale = dpi / 96.0
        return scale
    except:
        return 1.0


def human_delay(base_delay, randomness=0.3):
    """添加随机性延迟，模拟人类行为"""
    variation = base_delay * randomness
    actual_delay = base_delay + random.uniform(-variation, variation)
    time.sleep(max(0.001, actual_delay))


def clamp(value, min_val, max_val):
    """限制值在范围内"""
    return max(min_val, min(value, max_val))


def calculate_angle_change(p1, p2, p3):
    """计算三点之间的角度变化"""
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def calculate_path_complexity(points):
    """计算路径复杂度（基于角度变化）"""
    if len(points) < 3:
        return 0
    
    total_angle_change = 0
    for i in range(1, len(points) - 1):
        angle = calculate_angle_change(points[i-1], points[i], points[i+1])
        total_angle_change += angle
    
    return total_angle_change / (len(points) - 2) if len(points) > 2 else 0


def calibrate_and_activate(img_w, img_h):
    """
    画布校准 - 使用鼠标点击选择绘画区域
    返回: (screen_x, screen_y, scale_factor, offset_x, offset_y, 
            safe_x_min, safe_x_max, safe_y_min, safe_y_max)
    """
    print("\n--- 步骤 A: 校准画布 ---")
    print("请用鼠标左键点击小程序绘画区域的 [左上角]...")
    
    top_left = None
    def on_click_top_left(x, y, button, pressed):
        nonlocal top_left
        if pressed and button == mouse.Button.left:
            top_left = (x, y)
            return False
    
    with mouse.Listener(on_click=on_click_top_left) as listener:
        listener.join()
    
    print(f"已记录 [左上角] 坐标: {top_left}")

    print("\n请用鼠标左键点击小程序绘画区域的 [右下角]...")
    
    bottom_right = None
    def on_click_bottom_right(x, y, button, pressed):
        nonlocal bottom_right
        if pressed and button == mouse.Button.left:
            bottom_right = (x, y)
            return False
    
    with mouse.Listener(on_click=on_click_bottom_right) as listener:
        listener.join()
    
    print(f"已记录 [右下角] 坐标: {bottom_right}")
    
    print("\n等待 3 秒后开始绘画...")
    time.sleep(3)

    dpi_scale = get_dpi_scale()
    if dpi_scale != 1.0:
        print(f"检测到显示缩放: {dpi_scale * 100:.0f}% (DPI: {dpi_scale * 96:.0f})")
    
    screen_x, screen_y = top_left[0], top_left[1]
    screen_w, screen_h = bottom_right[0] - screen_x, bottom_right[1] - screen_y
    
    if screen_w <= 0 or screen_h <= 0:
        print("错误: 坐标无效。")
        return None, None, None, None, None, None, None, None, None

    scale_x = screen_w / img_w
    scale_y = screen_h / img_h
    scale_factor = min(scale_x, scale_y)
    
    actual_w = img_w * scale_factor
    actual_h = img_h * scale_factor
    
    offset_x = screen_x + (screen_w - actual_w) / 2
    offset_y = screen_y + (screen_h - actual_h) / 2
    
    safe_x_min = int(offset_x)
    safe_x_max = int(offset_x + actual_w)
    safe_y_min = int(offset_y)
    safe_y_max = int(offset_y + actual_h)
    
    print(f"画布校准完成:")
    print(f"  选定区域: {screen_w}x{screen_h} 像素")
    print(f"  图像尺寸: {img_w}x{img_h} 像素")
    print(f"  缩放比例: {scale_factor:.3f}")
    print(f"  实际绘制: {actual_w:.0f}x{actual_h:.0f} 像素")
    print(f"  绘制范围: ({safe_x_min}, {safe_y_min}) -> ({safe_x_max}, {safe_y_max})")
    
    test_x = int(screen_x + screen_w / 2)
    test_y = int(screen_y + screen_h / 2)
    print(f"  测试: 点击画布中心 ({test_x}, {test_y})")
    pydirectinput.click(test_x, test_y)
    time.sleep(0.5)
    
    return screen_x, screen_y, scale_factor, offset_x, offset_y, safe_x_min, safe_x_max, safe_y_min, safe_y_max


# --- 绘画方法 ---

def start_drawing_method_1(app, contours, img_w, img_h, stroke_len, draw_delay, lift_pause=0.05, speed_mult=1.0):
    """方法1: 短行程绘画"""
    try:
        calib_data = calibrate_and_activate(img_w, img_h)
        if calib_data[0] is None:
            return
        screen_x, screen_y, scale_factor, offset_x, offset_y, safe_x_min, safe_x_max, safe_y_min, safe_y_max = calib_data

        print(f"\n--- 步骤 D: 开始模拟绘画 (方法 1: 短行程) ---")
        print(f"   (笔画长度: {stroke_len} 点, 延迟: {draw_delay:.4f}秒, 速度: {speed_mult:.1f}x)")

        for path in contours:
            if keyboard.is_pressed('q'):
                raise KeyboardInterrupt("用户中止")
            
            path_points = path.reshape(-1, 2)
            
            for i in range(0, len(path_points), stroke_len):
                if keyboard.is_pressed('q'):
                    raise KeyboardInterrupt("用户中止")
                
                sub_path = path_points[i : i + stroke_len + 1]
                if len(sub_path) == 0:
                    continue
                
                start_x = clamp(int(offset_x + sub_path[0][0] * scale_factor), safe_x_min, safe_x_max)
                start_y = clamp(int(offset_y + sub_path[0][1] * scale_factor), safe_y_min, safe_y_max)
                
                pydirectinput.moveTo(start_x, start_y)
                pydirectinput.mouseDown()
                
                for point in sub_path[1:]:
                    draw_x = clamp(int(offset_x + point[0] * scale_factor), safe_x_min, safe_x_max)
                    draw_y = clamp(int(offset_y + point[1] * scale_factor), safe_y_min, safe_y_max)
                    pydirectinput.moveTo(draw_x, draw_y)
                    human_delay(draw_delay)
                
                pydirectinput.mouseUp()
                pause_time = lift_pause + random.uniform(0, lift_pause * 0.3)
                human_delay(pause_time)
        
        print("\n--- 绘画完成！ ---")

    except KeyboardInterrupt:
        print("\n\n检测到中止信号！")
    except Exception as e:
        print(f"\n发生意外错误: {e}")
    finally:
        time.sleep(0.05)
        pydirectinput.mouseUp()
        app.on_drawing_complete()


def start_drawing_method_2(app, contours, img_w, img_h, min_drag_dist, draw_delay, speed_mult=1.0):
    """方法2: 智能拖动"""
    try:
        calib_data = calibrate_and_activate(img_w, img_h)
        if calib_data[0] is None:
            return
        screen_x, screen_y, scale_factor, offset_x, offset_y, safe_x_min, safe_x_max, safe_y_min, safe_y_max = calib_data
            
        print(f"\n--- 步骤 D: 开始模拟绘画 (方法 2: 智能拖动) ---")
        print(f"   (最小距离: {min_drag_dist}px, 延迟: {draw_delay:.4f}秒, 速度: {speed_mult:.1f}x)")
        
        pydirectinput.mouseDown()

        for path in contours:
            if keyboard.is_pressed('q'):
                raise KeyboardInterrupt("用户中止")
            
            path_points = path.reshape(-1, 2)
            if len(path_points) == 0:
                continue

            start_x = clamp(int(offset_x + path_points[0][0] * scale_factor), safe_x_min, safe_x_max)
            start_y = clamp(int(offset_y + path_points[0][1] * scale_factor), safe_y_min, safe_y_max)
            
            pydirectinput.moveTo(start_x, start_y)
            last_drawn_screen_point = (start_x, start_y)
            
            for point in path_points[1:]:
                if keyboard.is_pressed('q'):
                    raise KeyboardInterrupt("用户中止")
                
                new_draw_x = clamp(int(offset_x + point[0] * scale_factor), safe_x_min, safe_x_max)
                new_draw_y = clamp(int(offset_y + point[1] * scale_factor), safe_y_min, safe_y_max)
                
                dist = abs(new_draw_x - last_drawn_screen_point[0]) + abs(new_draw_y - last_drawn_screen_point[1])
                
                if dist >= min_drag_dist:
                    pydirectinput.moveTo(new_draw_x, new_draw_y)
                    last_drawn_screen_point = (new_draw_x, new_draw_y)
                    human_delay(draw_delay)
        
        print("\n--- 绘画完成！ ---")

    except KeyboardInterrupt:
        print("\n\n检测到中止信号！")
    except Exception as e:
        print(f"\n发生意外错误: {e}")
    finally:
        time.sleep(0.05)
        pydirectinput.mouseUp()
        app.on_drawing_complete()


def start_drawing_method_3(app, contours, img_w, img_h, draw_delay, lift_pause, hand_shake, think_pause, speed_mult=1.0):
    """方法3: 仿真人绘画 - 快速移动+停顿"""
    try:
        calib_data = calibrate_and_activate(img_w, img_h)
        if calib_data[0] is None:
            return
        screen_x, screen_y, scale_factor, offset_x, offset_y, safe_x_min, safe_x_max, safe_y_min, safe_y_max = calib_data
            
        print(f"\n--- 步骤 D: 开始模拟绘画 (方法 3: 仿真人绘画) ---")
        print(f"   手部抖动: {hand_shake}px | 思考停顿: {think_pause}x | 速度倍率: {speed_mult:.1f}x")
        
        # 按轮廓面积排序 - 先画大轮廓（主体），再画小轮廓（细节）
        sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
        
        for path_idx, path in enumerate(sorted_contours):
            if keyboard.is_pressed('q'):
                raise KeyboardInterrupt("用户中止")
            
            path_points = path.reshape(-1, 2)
            if len(path_points) < 2:
                continue
            
            # 计算这条路径的复杂度（角度变化大 = 复杂）
            path_complexity = calculate_path_complexity(path_points)
            
            # 起笔位置
            start_point = path_points[0]
            start_x = clamp(int(offset_x + start_point[0] * scale_factor), safe_x_min, safe_x_max)
            start_y = clamp(int(offset_y + start_point[1] * scale_factor), safe_y_min, safe_y_max)
            
            pydirectinput.moveTo(
                start_x + random.randint(-hand_shake, hand_shake),
                start_y + random.randint(-hand_shake, hand_shake)
            )
            human_delay(draw_delay * 2)  # 起笔前思考
            
            pydirectinput.mouseDown()
            human_delay(draw_delay * 0.5)  # 起笔稳定
            
            last_point = start_point
            last_move_time = time.time()
            
            for i, point in enumerate(path_points[1:], 1):
                if keyboard.is_pressed('q'):
                    raise KeyboardInterrupt("用户中止")
                
                target_x_raw = offset_x + point[0] * scale_factor
                target_y_raw = offset_y + point[1] * scale_factor
                
                shake_x = random.randint(-hand_shake, hand_shake)
                shake_y = random.randint(-hand_shake, hand_shake)
                
                target_x = clamp(int(target_x_raw + shake_x), safe_x_min, safe_x_max)
                target_y = clamp(int(target_y_raw + shake_y), safe_y_min, safe_y_max)
                
                # 计算转角角度（判断是直线还是转角）
                angle_change = calculate_angle_change(last_point, point, 
                                                      path_points[min(i+1, len(path_points)-1)])
                
                # 关键改进：快速移动 + 移动后延迟，而非移动中延迟
                # 这样可以保持线条平滑
                current_time = time.time()
                time_since_last = current_time - last_move_time
                
                # 确保移动间隔不要太密集（最少间隔）
                min_interval = draw_delay * 0.3
                if time_since_last < min_interval:
                    time.sleep(min_interval - time_since_last)
                
                # 快速移动到目标点（不等待）
                pydirectinput.moveTo(target_x, target_y)
                
                # 移动后根据角度决定停顿时间
                if angle_change < 20:  # 直线 - 短暂停顿
                    human_delay(draw_delay * 0.3)
                elif angle_change < 60:  # 缓和曲线 - 中等停顿
                    human_delay(draw_delay * 0.6)
                else:  # 急转弯 - 较长停顿
                    human_delay(draw_delay * 1.2)
                    # 复杂转角额外思考停顿
                    if random.random() < 0.2:
                        human_delay(draw_delay * think_pause * 0.5)
                
                last_move_time = time.time()
                last_point = point
            
            # 收笔：轻轻抬起
            human_delay(draw_delay * 0.5)  # 收笔前稍停
            pydirectinput.mouseUp()
            
            # 换线停顿（带随机性）
            pause_time = lift_pause + random.uniform(0, lift_pause * 0.5)
            human_delay(pause_time)
            
            if (path_idx + 1) % 10 == 0:
                print(f"  已完成 {path_idx + 1}/{len(sorted_contours)} 条路径...")
        
        print("\n--- 绘画完成！ ---")

    except KeyboardInterrupt:
        print("\n\n检测到中止信号！")
    except Exception as e:
        print(f"\n发生意外错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        time.sleep(0.05)
        pydirectinput.mouseUp()
        app.on_drawing_complete()
