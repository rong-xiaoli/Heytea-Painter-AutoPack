"""
图像处理模块
包含线条提取算法：Pencil Sketch, Canny, Anime2Sketch
"""

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from modules.smart_resize import adaptive_resize, calculate_optimal_size

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch 未安装，Anime2Sketch 功能将不可用")


# --- Anime2Sketch 模型定义 ---
if TORCH_AVAILABLE:
    class UnetGenerator(nn.Module):
        def __init__(self, input_nc=3, output_nc=1, num_downs=8, ngf=64):
            super(UnetGenerator, self).__init__()
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=nn.InstanceNorm2d, innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=nn.InstanceNorm2d, use_dropout=True)
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=nn.InstanceNorm2d)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=nn.InstanceNorm2d)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=nn.InstanceNorm2d)
            self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=nn.InstanceNorm2d)

        def forward(self, input):
            return self.model(input)

    class UnetSkipConnectionBlock(nn.Module):
        def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
            super(UnetSkipConnectionBlock, self).__init__()
            self.outermost = outermost
            if input_nc is None:
                input_nc = outer_nc
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
            downrelu = nn.LeakyReLU(0.2, True)
            downnorm = norm_layer(inner_nc)
            uprelu = nn.ReLU(True)
            upnorm = norm_layer(outer_nc)

            if outermost:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
                model = down + [submodule] + up
            elif innermost:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                model = down + up
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]
                if use_dropout:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up

            self.model = nn.Sequential(*model)

        def forward(self, x):
            if self.outermost:
                return self.model(x)
            else:
                return torch.cat([x, self.model(x)], 1)


# --- 全局模型缓存 ---
anime2sketch_model = None


def get_anime2sketch_model():
    """加载或获取缓存的 Anime2Sketch 模型"""
    global anime2sketch_model
    
    if anime2sketch_model is not None:
        return anime2sketch_model
    
    if not TORCH_AVAILABLE:
        print("错误: PyTorch 未安装")
        anime2sketch_model = "error"
        return None
    
    if anime2sketch_model == "error":
        return None
    
    # 尝试加载模型
    try:
        import os
        # 模型路径：使用绝对路径，支持中文路径
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(script_dir, "models", "netG.pth")
        
        # 确保路径存在
        if not os.path.exists(model_path):
            # 尝试其他可能的路径
            alternative_paths = [
                os.path.join(os.getcwd(), "models", "netG.pth"),
                os.path.join(script_dir, "netG.pth"),
                "models/netG.pth",
                "netG.pth"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"找到模型文件: {model_path}")
                    break
            else:
                # 所有路径都不存在
                print(f"=" * 60)
                print(f"错误: 找不到 Anime2Sketch 模型文件")
                print(f"尝试的路径:")
                print(f"  1. {os.path.join(script_dir, 'models', 'netG.pth')}")
                for i, p in enumerate(alternative_paths, 2):
                    print(f"  {i}. {p}")
                print(f"")
                print(f"请下载模型文件:")
                print(f"  下载地址: https://github.com/Mukosame/Anime2Sketch/releases/download/1.0/netG.pth")
                print(f"  放置位置: {os.path.join(script_dir, 'models', 'netG.pth')}")
                print(f"=" * 60)
                anime2sketch_model = "error"
                return None
        
        model_path = os.path.abspath(model_path)
        
        if not os.path.exists(model_path):
            print(f"错误: 找不到模型文件: {model_path}")
            print("请下载 Anime2Sketch 模型并放置到 models 文件夹:")
            print("https://github.com/Mukosame/Anime2Sketch/releases/download/1.0/netG.pth")
            anime2sketch_model = "error"
            return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        anime2sketch_model = UnetGenerator(input_nc=3, output_nc=1, num_downs=8, ngf=64)
        
        # 加载权重并处理 'module.' 前缀
        state_dict = torch.load(model_path, map_location=device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        missing_keys, unexpected_keys = anime2sketch_model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"警告: 缺少的键: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"警告: 意外的键: {len(unexpected_keys)} 个 (已忽略)")
        
        anime2sketch_model.to(device)
        anime2sketch_model.eval()
        print(f"Anime2Sketch 模型加载成功 (设备: {device})")
    except Exception as e:
        print(f"错误: 无法加载 Anime2Sketch 模型: {e}")
        import traceback
        traceback.print_exc()
        anime2sketch_model = "error"
        return None
    
    return anime2sketch_model


# --- 图像处理函数 ---

def load_image_with_unicode(file_path, target_size=None):
    """
    支持 Unicode 路径的图片加载并自动调整到合理尺寸
    
    参数:
        file_path: 图片路径（支持中文）
        target_size: 目标尺寸（可选），如果提供则优先使用
    
    返回:
        image: BGR 图像数组
    """
    try:
        # 使用 numpy 读取文件字节，然后用 cv2 解码
        with open(file_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"错误: 无法解码图像文件: {file_path}")
            print(f"  请确认文件是有效的图片格式（jpg, png, bmp 等）")
            return None
        
        # 获取原始尺寸
        h, w = image.shape[:2]
        original_size = f"{w}x{h}"
        
        # 智能缩放策略：使用高质量算法防止精度损失
        MAX_SIZE = 2048       # 绝对上限（防止内存问题）
        MIN_SIZE = 256        # 绝对下限
        DEFAULT_TARGET = 1024 # 默认目标尺寸
        
        # 如果没有指定目标尺寸，使用默认策略
        if target_size is None:
            target_size = DEFAULT_TARGET
        
        max_dim = max(h, w)
        
        # 智能缩放逻辑（使用高质量算法）
        if max_dim > MAX_SIZE:
            # 超大图片：使用智能下采样
            scale = MAX_SIZE / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"🔧 图片过大 ({original_size})，使用智能算法缩小...")
            image = adaptive_resize(image, new_w, new_h, preserve_detail=True)
            print(f"   ✓ 已优化到 {new_w}x{new_h} (保留细节)")
            
        elif max_dim > target_size * 1.5:
            # 图片较大：使用智能下采样
            scale = target_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"🔧 图片较大 ({original_size})，使用智能算法优化...")
            image = adaptive_resize(image, new_w, new_h, preserve_detail=True)
            print(f"   ✓ 已优化到 {new_w}x{new_h} (防止精度损失)")
            
        elif max_dim < MIN_SIZE:
            # 图片太小：使用智能上采样
            scale = MIN_SIZE / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"🔧 图片过小 ({original_size})，使用智能算法放大...")
            image = adaptive_resize(image, new_w, new_h, preserve_detail=True)
            print(f"   ✓ 已放大到 {new_w}x{new_h} (边缘保留)")
        
        else:
            # 尺寸合理：保持原样
            print(f"✅ 图片尺寸合适: {original_size}")
        
        return image
    except FileNotFoundError:
        print(f"错误: 文件不存在: {file_path}")
        return None
    except PermissionError:
        print(f"错误: 没有读取权限: {file_path}")
        return None
    except Exception as e:
        print(f"错误: 加载图像失败: {file_path}")
        print(f"  异常信息: {e}")
        return None


def process_image_pencil(file_path, sigma_s, sigma_r, shade_factor, 
                         simplify_eps, spline_smoothness, preview_thickness):
    """Pencil Sketch 线条提取"""
    try:
        image_rgb = load_image_with_unicode(file_path)
        if image_rgb is None:
            print("错误: 无法加载图像")
            return None, None, 0, 0
        
        img_gray, img_blur = cv2.pencilSketch(
            image_rgb,
            sigma_s=sigma_s,
            sigma_r=sigma_r,
            shade_factor=shade_factor
        )
        
        _, edges = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplified_contours = [cv2.approxPolyDP(c, simplify_eps, True) for c in contours]
        
        final_contours = simplified_contours
        if spline_smoothness > 0:
            final_contours = smooth_contours(simplified_contours, spline_smoothness)
        
        preview_canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for contour in final_contours:
            cv2.polylines(preview_canvas, [contour], False, (0, 0, 255), int(preview_thickness), lineType=cv2.LINE_AA)
        
        total_points = sum(len(c) for c in final_contours)
        print(f"(Pencil) 预览: {len(final_contours)} 条路径, {total_points} 个点。")
        
        return preview_canvas, final_contours, image_rgb.shape[1], image_rgb.shape[0]
    
    except Exception as e:
        print(f"Pencil Sketch 图像处理错误: {e}")
        return None, None, 0, 0


def process_image_canny(file_path, blur_kernel, low_thresh, high_thresh,
                        simplify_eps, spline_smoothness, preview_thickness):
    """Canny 边缘检测"""
    try:
        image_rgb = load_image_with_unicode(file_path)
        if image_rgb is None:
            print("错误: 无法加载图像")
            return None, None, 0, 0
        
        img_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (blur_kernel, blur_kernel), 0)
        edges = cv2.Canny(img_blur, low_thresh, high_thresh)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        simplified_contours = [cv2.approxPolyDP(c, simplify_eps, True) for c in contours]
        
        final_contours = simplified_contours
        if spline_smoothness > 0:
            final_contours = smooth_contours(simplified_contours, spline_smoothness)
        
        preview_canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for contour in final_contours:
            cv2.polylines(preview_canvas, [contour], False, (0, 0, 255), int(preview_thickness), lineType=cv2.LINE_AA)
        
        total_points = sum(len(c) for c in final_contours)
        print(f"(Canny) 预览: {len(final_contours)} 条路径, {total_points} 个点。")
        
        return preview_canvas, final_contours, image_rgb.shape[1], image_rgb.shape[0]
    
    except Exception as e:
        print(f"Canny 图像处理错误: {e}")
        return None, None, 0, 0


def process_image_anime2sketch(file_path, simplify_eps, spline_smoothness, preview_thickness, 
                                threshold_val, morph_size, morph_iter, min_area, contour_mode="外部轮廓 (快速)",
                                pre_blur=0, edge_enhance=0, sigmoid_threshold=0.5, invert=False, adaptive=False):
    """Anime2Sketch AI 线条提取"""
    try:
        if not TORCH_AVAILABLE:
            error_img = np.zeros((300, 500, 3), dtype=np.uint8)
            cv2.putText(error_img, "PyTorch Not Installed", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return error_img, [], 500, 300
        
        image_rgb = load_image_with_unicode(file_path)
        if image_rgb is None:
            print("错误: 无法加载图像")
            return None, None, 0, 0
        
        model = get_anime2sketch_model()
        if model is None:
            error_img = np.zeros((300, 500, 3), dtype=np.uint8)
            cv2.putText(error_img, "Model Failed to Load", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return error_img, [], 500, 300
        
        # 预处理图像
        h, w = image_rgb.shape[:2]
        original_h, original_w = h, w  # 保存原始尺寸
        target_h = ((h - 1) // 256 + 1) * 256
        target_w = ((w - 1) // 256 + 1) * 256
        
        max_size = 1024
        need_restore_size = False  # 🔧 标志:是否需要恢复原始尺寸
        
        if target_h > max_size or target_w > max_size:
            need_restore_size = True  # 标记需要恢复
            scale = min(max_size / target_h, max_size / target_w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
            target_h = ((new_h - 1) // 256 + 1) * 256
            target_w = ((new_w - 1) // 256 + 1) * 256
        else:
            image_resized = image_rgb
        
        pad_h = target_h - image_resized.shape[0]
        pad_w = target_w - image_resized.shape[1]
        image_padded = cv2.copyMakeBorder(image_resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        # 转换为张量
        image = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0).float()
        
        device = next(model.parameters()).device
        image = image.to(device)
        
        # 推理
        with torch.no_grad():
            edge = model(image)
            edge = torch.sigmoid((edge - sigmoid_threshold) * 10)
        
        # 后处理
        edge_map = edge.squeeze().cpu().numpy()
        edge_map = edge_map[:image_resized.shape[0], :image_resized.shape[1]]
        
        # 🔧 关键修复: 先转换到8位再放大(避免轮廓坐标错位)
        edge_map = (edge_map * 255.0).astype(np.uint8)
        
        # 🔧 使用标志变量判断是否需要恢复尺寸
        if need_restore_size:
            print(f"  [缩放恢复] {image_resized.shape[1]}x{image_resized.shape[0]} → {original_w}x{original_h}")
            edge_map = cv2.resize(edge_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # 预处理模糊
        if pre_blur > 0:
            blur_size = pre_blur if pre_blur % 2 == 1 else pre_blur + 1
            edge_map = cv2.GaussianBlur(edge_map, (blur_size, blur_size), 0)
            print(f"  [预处理] 高斯模糊: {blur_size}x{blur_size}")
        
        # 边缘增强
        if edge_enhance > 0:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * edge_enhance / 3.0
            edge_map = cv2.filter2D(edge_map, -1, kernel)
            edge_map = np.clip(edge_map, 0, 255).astype(np.uint8)
            print(f"  [边缘增强] 强度: {edge_enhance:.1f}")
        
        # 二值化
        if adaptive:
            binary = cv2.adaptiveThreshold(edge_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY, 11, 2)
            print(f"  [二值化] 自适应方法")
        else:
            thresh_type = cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY
            _, binary = cv2.threshold(edge_map, threshold_val, 255, thresh_type)
            print(f"  [二值化] 固定阈值: {threshold_val}")
        
        white_pixels_before = np.sum(binary == 255)
        
        # 形态学处理
        kernel = np.ones((morph_size, morph_size), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=morph_iter)
        white_pixels_after = np.sum(binary == 255)
        
        pixel_increase = white_pixels_after - white_pixels_before
        increase_percent = (pixel_increase / white_pixels_before * 100) if white_pixels_before > 0 else 0
        print(f"  [形态学] 核:{morph_size}x{morph_size}, 迭代:{morph_iter} → +{increase_percent:.1f}%")
        
        # 提取轮廓
        if "骨架提取" in contour_mode:
            skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            print(f"  [骨架模式] {len(contours)} 条")
        elif "所有轮廓" in contour_mode:
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            print(f"  [所有轮廓] {len(contours)} 条")
        else:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"  [外部轮廓] {len(contours)} 条")
        
        # 过滤
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        print(f"  过滤后: {len(contours)} 条")
        
        # 简化和平滑
        simplified_contours = [cv2.approxPolyDP(c, simplify_eps, True) for c in contours]
        final_contours = simplified_contours
        if spline_smoothness > 0:
            final_contours = smooth_contours(simplified_contours, spline_smoothness)
        
        # 预览
        preview_canvas = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        for contour in final_contours:
            cv2.polylines(preview_canvas, [contour], False, (0, 0, 255), int(preview_thickness), lineType=cv2.LINE_AA)
        
        total_points = sum(len(c) for c in final_contours)
        print(f"(Anime2Sketch) {len(final_contours)} 条路径, {total_points} 个点")
        
        # 🔍 调试:检查轮廓坐标范围
        if len(final_contours) > 0:
            all_points = np.vstack([c.reshape(-1, 2) for c in final_contours])
            min_x, min_y = np.min(all_points, axis=0)
            max_x, max_y = np.max(all_points, axis=0)
            print(f"  [坐标范围] X: {min_x:.0f}~{max_x:.0f} (应为 0~{image_rgb.shape[1]})")
            print(f"             Y: {min_y:.0f}~{max_y:.0f} (应为 0~{image_rgb.shape[0]})")
            
            # 检查是否只占一小部分
            x_coverage = (max_x - min_x) / image_rgb.shape[1] * 100
            y_coverage = (max_y - min_y) / image_rgb.shape[0] * 100
            if x_coverage < 50 or y_coverage < 50:
                print(f"  ⚠️ 警告: 轮廓只覆盖图像的 {x_coverage:.1f}% (X) × {y_coverage:.1f}% (Y)")
                print(f"           这可能是边缘检测参数问题或图片内容集中在局部")
        
        return preview_canvas, final_contours, image_rgb.shape[1], image_rgb.shape[0]
    
    except Exception as e:
        print(f"Anime2Sketch 错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0, 0


def smooth_contours(contours, smoothness_s, new_points_factor=1.5):
    """B-spline 样条平滑"""
    smoothed_contours = []
    
    for c in contours:
        if len(c) < 4:
            smoothed_contours.append(c)
            continue
        
        try:
            x, y = c[:, 0, 0], c[:, 0, 1]
            tck, u = splprep([x, y], s=smoothness_s, k=3, per=0)
            num_new_points = int(len(c) * new_points_factor)
            if num_new_points < 4:
                num_new_points = 4
            u_new = np.linspace(u.min(), u.max(), num_new_points)
            x_new, y_new = splev(u_new, tck)
            new_contour = np.array([x_new, y_new]).T.reshape(-1, 1, 2).astype(np.int32)
            smoothed_contours.append(new_contour)
        except:
            smoothed_contours.append(c)
    
    return smoothed_contours
