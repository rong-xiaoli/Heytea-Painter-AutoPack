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
import sys

pydirectinput.PAUSE = 0.005


# --- DPI 感知常量 ---
DPI_AWARENESS_CONTEXT_UNAWARE = -1
DPI_AWARENESS_CONTEXT_SYSTEM_AWARE = -2
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE = -3
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
DPI_AWARENESS_CONTEXT_UNAWARE_GDISCALED = -5


def set_dpi_awareness():
    """
    检测当前 DPI 感知状态
    注意: 实际设置在 heytea_modern.py 启动时完成
    
    返回:
        dict: {'method': 方法名, 'success': 是否成功, 'level': DPI 感知级别}
    """
    # DPI已经在程序启动时设置,这里只是检测状态
    try:
        shcore = ctypes.windll.shcore
        # 如果能成功调用说明至少是V1级别
        return {'method': 'SetProcessDpiAwareness', 'success': True, 'level': 'Per-Monitor V1'}
    except:
        pass
    
    try:
        user32 = ctypes.windll.user32
        return {'method': 'SetProcessDPIAware', 'success': True, 'level': 'System DPI'}
    except:
        pass
    
    return {'method': 'None', 'success': False, 'level': 'Unaware'}


# 检测当前DPI状态
_dpi_result = set_dpi_awareness()


def get_dpi_info():
    """
    获取详细的 DPI 信息和屏幕分辨率
    
    返回:
        dict: {
            'scale': DPI 缩放比例,
            'dpi': 实际 DPI 值,
            'screen_size': (宽, 高) 屏幕分辨率,
            'logical_size': (宽, 高) 逻辑分辨率,
            'awareness_method': DPI 感知方法,
            'awareness_level': DPI 感知级别
        }
    """
    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        
        # 获取主显示器句柄
        hdc = user32.GetDC(0)
        
        # 获取 DPI
        dpi_x = gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        dpi_y = gdi32.GetDeviceCaps(hdc, 90)  # LOGPIXELSY
        
        # 释放 DC
        user32.ReleaseDC(0, hdc)
        
        # 计算缩放比例（96 DPI = 100%）
        scale_x = dpi_x / 96.0
        scale_y = dpi_y / 96.0
        
        # 获取屏幕分辨率
        # SM_CXSCREEN 和 SM_CYSCREEN 返回值取决于 DPI 感知模式
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        
        # 尝试获取真实物理分辨率
        try:
            import pyautogui
            logical_w, logical_h = pyautogui.size()
        except:
            logical_w, logical_h = screen_w, screen_h
        
        return {
            'scale': (scale_x + scale_y) / 2,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'dpi': dpi_x,
            'screen_size': (screen_w, screen_h),
            'logical_size': (logical_w, logical_h),
            'awareness_method': _dpi_result['method'],
            'awareness_level': _dpi_result['level']
        }
    except Exception as e:
        print(f"获取 DPI 信息失败: {e}")
        return {
            'scale': 1.0,
            'scale_x': 1.0,
            'scale_y': 1.0,
            'dpi': 96,
            'screen_size': (1920, 1080),
            'logical_size': (1920, 1080)
        }


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
    
    # 获取系统信息（在等待前显示，方便调试）
    dpi_info = get_dpi_info()
    
    print(f"\n" + "="*60)
    print(f"系统显示信息:")
    print(f"  当前屏幕分辨率: {dpi_info['screen_size'][0]} x {dpi_info['screen_size'][1]} 像素")
    if dpi_info['logical_size'] != dpi_info['screen_size']:
        print(f"  逻辑分辨率: {dpi_info['logical_size'][0]} x {dpi_info['logical_size'][1]} 像素")
    print(f"  当前 DPI 缩放: {dpi_info['scale'] * 100:.0f}% (DPI: {dpi_info['dpi']})")
    print(f"  DPI 感知方法: {dpi_info['awareness_method']}")
    print(f"  DPI 感知级别: {dpi_info['awareness_level']}")
    print(f"  处理后图像尺寸: {img_w} x {img_h} 像素")
    print(f"="*60)
    
    print("\n等待 3 秒后开始绘画...")
    time.sleep(3)
    
    # 计算画布尺寸（使用 pynput 获取的物理坐标）
    screen_x, screen_y = top_left[0], top_left[1]
    bottom_x, bottom_y = bottom_right[0], bottom_right[1]
    screen_w = bottom_x - screen_x
    screen_h = bottom_y - screen_y
    
    print(f"\n画布校准结果:")
    print(f"  画布左上角: ({screen_x}, {screen_y})")
    print(f"  画布右下角: ({bottom_x}, {bottom_y})")
    print(f"  画布尺寸: {screen_w} x {screen_h} 像素")
    
    if screen_w <= 0 or screen_h <= 0:
        print("错误: 坐标无效。")
        return None, None, None, None, None, None, None, None, None

    # 计算等比例缩放（保持宽高比）
    # 使用 min 策略：图片完全适合画布（不裁切、不变形）
    scale_x = screen_w / img_w
    scale_y = screen_h / img_h
    scale_factor = min(scale_x, scale_y)  # 取较小值，确保图片完全在画布内
    
    # 缩放后的实际尺寸
    actual_w = img_w * scale_factor
    actual_h = img_h * scale_factor
    
    # 居中对齐：计算偏移量使图像在画布中居中
    offset_x = screen_x + (screen_w - actual_w) / 2
    offset_y = screen_y + (screen_h - actual_h) / 2
    
    # 安全绘制范围
    safe_x_min = int(offset_x)
    safe_x_max = int(offset_x + actual_w)
    safe_y_min = int(offset_y)
    safe_y_max = int(offset_y + actual_h)
    
    print(f"\n绘画参数计算:")
    print(f"  图像处理尺寸: {img_w} x {img_h} 像素 (已优化)")
    print(f"  画布可用尺寸: {screen_w} x {screen_h} 像素")
    print(f"  X轴缩放比例: {scale_x:.4f}")
    print(f"  Y轴缩放比例: {scale_y:.4f}")
    print(f"  最终缩放比例: {scale_factor:.4f} (取较小值保持比例)")
    print(f"  缩放后图像: {actual_w:.1f} x {actual_h:.1f} 像素")
    print(f"  居中偏移: X={((screen_w - actual_w) / 2):.1f}, Y={((screen_h - actual_h) / 2):.1f}")
    print(f"  绘制区域: ({safe_x_min}, {safe_y_min}) → ({safe_x_max}, {safe_y_max})")
    
    # 🔍 调试信息:检查轮廓坐标范围 (仅在scale_factor<0.5时打印)
    if scale_factor < 0.5:
        print(f"\n🔍 轮廓坐标诊断 (验证数据一致性):")
        print(f"   提示: 轮廓坐标应该在 0-{img_w} (X) 和 0-{img_h} (Y) 范围内")
    
    # 检查画布利用率
    canvas_usage = (actual_w * actual_h) / (screen_w * screen_h) * 100
    print(f"  画布利用率: {canvas_usage:.1f}%")
    
    # 分析利用率低的原因
    if canvas_usage < 80:
        aspect_img = img_w / img_h
        aspect_canvas = screen_w / screen_h
        if abs(aspect_img - aspect_canvas) > 0.2:
            print(f"  💡 提示: 图片宽高比 ({aspect_img:.2f}) 与画布 ({aspect_canvas:.2f}) 差异较大")
            if aspect_img > aspect_canvas:
                print(f"     图片更宽，建议裁剪图片为更接近 {aspect_canvas:.1f}:1 的比例")
            else:
                print(f"     图片更高，建议裁剪图片为更接近 {aspect_canvas:.1f}:1 的比例")
    
    # 精度警告（关键改进）
    if scale_factor < 0.5:
        print(f"\n⚠️ 警告: 缩放比例过小 ({scale_factor:.3f})")
        print(f"   图像尺寸 ({img_w}x{img_h}) 相对画布 ({screen_w}x{screen_h}) 过大")
        print(f"   这会导致:")
        print(f"     • 轮廓精度损失（每 {1/scale_factor:.1f} 个像素才绘制 1 个点）")
        print(f"     • 细节丢失")
        print(f"   解决方案:")
        print(f"     ✅ 推荐: 使用更大的画布（建议至少 {img_w//2}x{img_h//2} 像素）")
        print(f"     ✅ 或者: 在图像编辑器中预先裁剪/缩小图片")
        print(f"     ⚠️  当前画布太小，无法呈现完整细节")
    elif scale_factor < 0.8:
        print(f"\n💡 提示: 缩放比例 {scale_factor:.3f}")
        print(f"   建议使用更大的画布以获得更好的绘画效果")
    
    # 测试点击：点击图像中心
    test_x = int(offset_x + actual_w / 2)
    test_y = int(offset_y + actual_h / 2)
    print(f"\n测试: 点击图像中心 ({test_x}, {test_y})")
    pydirectinput.click(test_x, test_y)
    time.sleep(0.5)
    
    print(f"\n提示: 按 Q 键可随时退出绘画")
    print(f"=" * 60)
    
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


def start_drawing_method_3(app, contours, img_w, img_h, draw_delay, lift_pause, hand_shake, think_pause, corner_sharpness=5, speed_mult=1.0):
    """方法3: 仿真人绘画 - 快速移动+停顿"""
    try:
        calib_data = calibrate_and_activate(img_w, img_h)
        if calib_data[0] is None:
            return
        screen_x, screen_y, scale_factor, offset_x, offset_y, safe_x_min, safe_x_max, safe_y_min, safe_y_max = calib_data
            
        print(f"\n--- 步骤 D: 开始模拟绘画 (方法 3: 仿真人绘画) ---")
        print(f"   手部抖动: {hand_shake}px | 思考停顿: {think_pause}x | 转角锐利度: {corner_sharpness} | 速度倍率: {speed_mult:.1f}x")
        
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
                
                # 转角锐利度处理（核心逻辑改进）
                # 原理：锐利度控制的是"转角停顿时间"，而非抬笔
                # - 圆润（0-3）：快速连续移动，系统自动插值形成圆弧
                # - 锐利（7-10）：转角处长时间停顿，形成明显的顿挫感
                
                is_corner = angle_change > 30  # 30度以上视为转角
                
                # 移动到目标点（始终保持按下状态，不抬笔）
                pydirectinput.moveTo(target_x, target_y)
                
                # 根据锐利度和角度计算停顿时间
                if is_corner:
                    # 转角处的停顿策略
                    if corner_sharpness >= 8:
                        # 极度锐利（8-10）：转角处明显停顿
                        # 停顿时间与角度和锐利度成正比
                        pause_multiplier = 2.0 + (corner_sharpness - 8) * 0.5
                        if angle_change > 90:
                            pause_multiplier *= 1.5  # 大角度额外增强
                        human_delay(draw_delay * pause_multiplier)
                        
                        # 可选：急转弯额外思考
                        if angle_change > 120 and random.random() < 0.3:
                            human_delay(draw_delay * think_pause * 0.3)
                    
                    elif corner_sharpness >= 5:
                        # 中度锐利（5-7）：适度停顿
                        pause_multiplier = 1.0 + (corner_sharpness - 5) * 0.3
                        if angle_change > 90:
                            pause_multiplier *= 1.3
                        human_delay(draw_delay * pause_multiplier)
                    
                    elif corner_sharpness >= 3:
                        # 轻微锐利（3-4）：短暂停顿
                        pause_multiplier = 0.6 + (corner_sharpness - 3) * 0.2
                        if angle_change > 90:
                            pause_multiplier *= 1.2
                        human_delay(draw_delay * pause_multiplier)
                    
                    else:
                        # 圆润（0-2）：几乎不停顿，快速过渡
                        # 角度越大停顿越短（反直觉但符合圆润效果）
                        pause_multiplier = 0.3 - corner_sharpness * 0.05
                        human_delay(draw_delay * pause_multiplier)
                
                else:
                    # 直线段：统一处理，锐利度不影响直线
                    human_delay(draw_delay * 0.3)
                
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
