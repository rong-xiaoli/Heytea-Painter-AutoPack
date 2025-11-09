"""
喝茶画画 - 主程序
自动化绘画工具 with AI线条提取
"""

import sys
import os

# 添加modules路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import json
import cv2

# 导入自定义模块
from image_processor import process_image_pencil, process_image_canny, process_image_anime2sketch
from contour_optimizer import thin_contours_to_skeleton, remove_backtracking, apply_point_skipping, apply_jitter_correction
from drawing_engine import start_drawing_method_1, start_drawing_method_2, start_drawing_method_3


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("喝茶画画 - 模块化版本")
        self.root.geometry("1100x800")
        
        self.file_path = None
        self.contours = None
        self.img_w = 0
        self.img_h = 0
        self.preset_file = "heytea_presets.json"
        
        # 预览缩放
        self.preview_zoom = 1.0
        self.preview_original_img = None
        
        self.setup_gui()
        
        # 启动时加载预设
        self.load_presets(silent=True)
        self.on_method_change(None)
    
    def setup_gui(self):
        """创建GUI界面"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板（滚动）
        self.canvas = tk.Canvas(main_frame, borderwidth=0, background="#ffffff")
        self.scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.control_frame = ttk.Frame(self.canvas, padding=10)
        self.canvas.create_window((0, 0), window=self.control_frame, anchor="nw", tags="control_frame")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.Y)
        
        # 右侧预览
        self.preview_label = ttk.Label(main_frame, text="请先加载图片", background="gray")
        self.preview_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.control_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig("control_frame", width=e.width))
        
        self.preview_label.bind("<MouseWheel>", self.on_preview_mousewheel)
        self.preview_label.bind("<Configure>", self.on_preview_resize)
        
        # 创建控件
        self.create_widgets()
    
    def create_widgets(self):
        """创建所有控件"""
        # 加载按钮
        ttk.Button(self.control_frame, text="1. 加载图片", command=self.load_image).pack(fill=tk.X, pady=5)
        
        # 提取方法选择
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(self.control_frame, text="--- 线条提取 ---").pack()
        
        self.extraction_method_var = tk.StringVar(value="Pencil Sketch (V19)")
        method_combo = ttk.Combobox(self.control_frame, textvariable=self.extraction_method_var, 
                                     values=["Pencil Sketch (V19)", "Canny 边缘检测", "Anime2Sketch"],
                                     state="readonly")
        method_combo.pack(fill=tk.X, pady=5)
        method_combo.bind("<<ComboboxSelected>>", self.on_method_change)

        # 动态参数占位符
        self.params_placeholder = ttk.Frame(self.control_frame)
        self.params_placeholder.pack(fill=tk.X)

        # Pencil 参数
        self.pencil_frame = ttk.Frame(self.params_placeholder)
        self.sigma_s = self.create_slider(self.pencil_frame, "1. 平滑度", 1, 200, 60, 1)
        self.sigma_r = self.create_slider(self.pencil_frame, "2. 细节保留", 0.01, 1.0, 0.4, 0.01)
        self.shade_factor = self.create_slider(self.pencil_frame, "3. 阴影强度", 0.0, 1.0, 0.05, 0.01)
        
        # Canny 参数
        self.canny_frame = ttk.Frame(self.params_placeholder)
        self.canny_blur = self.create_slider(self.canny_frame, "1. 高斯模糊", 1, 20, 3, 1)
        self.canny_low = self.create_slider(self.canny_frame, "2. 低阈值", 1, 500, 50, 1)
        self.canny_high = self.create_slider(self.canny_frame, "3. 高阈值", 1, 1000, 150, 1)
        
        # Anime2Sketch 参数
        self.anime_frame = ttk.Frame(self.params_placeholder)
        self.anime_threshold = self.create_slider(self.anime_frame, "1. 二值化阈值", 50, 200, 127, 1)
        self.anime_morph_size = self.create_slider(self.anime_frame, "2. 形态学核大小", 1, 5, 2, 1)
        self.anime_morph_iter = self.create_slider(self.anime_frame, "3. 形态学迭代", 1, 3, 1, 1)
        self.anime_min_area = self.create_slider(self.anime_frame, "4. 最小轮廓面积", 5, 100, 10, 1)
        
        ttk.Separator(self.anime_frame, orient='horizontal').pack(fill=tk.X, pady=8)
        ttk.Label(self.anime_frame, text="--- 高级参数 ---", font=("Arial", 8, "bold")).pack()
        
        self.anime_pre_blur = self.create_slider(self.anime_frame, "5. 预处理模糊", 0, 9, 0, 1)
        self.anime_edge_enhance = self.create_slider(self.anime_frame, "6. 边缘增强", 0, 3.0, 0, 0.1)
        self.anime_sigmoid_threshold = self.create_slider(self.anime_frame, "7. 模型敏感度", 0.3, 0.9, 0.5, 0.05)
        
        self.anime_invert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.anime_frame, text="反转提取", variable=self.anime_invert_var, 
                       command=self.update_preview).pack(fill=tk.X, pady=2)
        
        self.anime_adaptive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.anime_frame, text="自适应二值化", variable=self.anime_adaptive_var, 
                       command=self.update_preview).pack(fill=tk.X, pady=2)
        
        ttk.Label(self.anime_frame, text="8. 轮廓提取模式:").pack(fill=tk.X, pady=(5,0))
        self.anime_contour_mode = tk.StringVar(value="外部轮廓 (快速)")
        ttk.Combobox(self.anime_frame, textvariable=self.anime_contour_mode,
                     values=["外部轮廓 (快速)", "所有轮廓 (详细)", "骨架提取 (推荐)"],
                     state="readonly").pack(fill=tk.X, pady=5)

        # 通用优化
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(self.control_frame, text="--- 通用优化 ---").pack()
        
        self.simplify_epsilon = self.create_slider(self.control_frame, "4. 线条简化度", 0.1, 5.0, 1.0, 0.1)
        self.preview_thickness = self.create_slider(self.control_frame, "5. 预览线条粗细", 1, 20, 1, 1)
        self.spline_smoothness = self.create_slider(self.control_frame, "6. 路径平滑度", 0, 5000, 0, 1)
        self.jitter_correction = self.create_slider(self.control_frame, "7. 抖动修正强度", 0, 10, 0, 1)
        
        # 绘画模拟
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(self.control_frame, text="--- 绘画模拟 ---").pack()

        ttk.Label(self.control_frame, text="选择绘画方法:").pack(fill=tk.X, pady=(5,0))
        self.drawing_method_var = tk.StringVar(value="方法 1: 短行程 (推荐)")
        ttk.Combobox(self.control_frame, textvariable=self.drawing_method_var, 
                     values=["方法 1: 短行程 (推荐)", "方法 2: 智能拖动 (蜘蛛网?)", "方法 3: 仿真人绘画 🎨"],
                     state="readonly").pack(fill=tk.X, pady=5)
        
        self.stroke_len = self.create_slider(self.control_frame, "笔画长度 [方法1]", 5, 100, 15, 5)
        self.min_drag_dist = self.create_slider(self.control_frame, "最小拖动距离 [方法2]", 1, 20, 5, 1)
        self.draw_delay = self.create_slider(self.control_frame, "绘画延迟 (ms)", 1, 100, 5, 1)
        self.lift_pause = self.create_slider(self.control_frame, "换线停顿 (x10ms)", 3, 15, 5, 1)
        self.hand_shake = self.create_slider(self.control_frame, "手部抖动 [方法3]", 0, 5, 1, 1)
        self.think_pause = self.create_slider(self.control_frame, "思考停顿倍率 [方法3]", 1, 10, 3, 1)
        
        # 轮廓优化选项
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        self.thin_contours_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="🔥 边缘细化 (双线变单线)", 
                       variable=self.thin_contours_var, 
                       command=self.update_preview).pack(fill=tk.X, pady=2)
        
        self.skip_points = self.create_slider(self.control_frame, "跳点加速", 1, 5, 1, 1)
        
        # 全局速度控制
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Label(self.control_frame, text="⚡ 全局速度控制", font=("Arial", 9, "bold")).pack(pady=2)
        self.speed_multiplier = self.create_slider(self.control_frame, "速度倍率", 0.1, 5.0, 1.0, 0.1)
        
        # 功能按钮
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        preset_frame = ttk.Frame(self.control_frame)
        ttk.Button(preset_frame, text="保存预设", command=self.save_presets).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(preset_frame, text="加载预设", command=self.load_presets).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(self.control_frame, text="重置参数", command=self.reset_sliders).pack(fill=tk.X, pady=5)
        self.start_button = ttk.Button(self.control_frame, text="2. 开始绘画", command=self.start_drawing_controller, state="disabled")
        self.start_button.pack(fill=tk.X, pady=20)
    
    def create_slider(self, parent, text, from_, to, default, resolution=None):
        """创建滑块控件"""
        frame = ttk.Frame(parent)
        
        label_frame = ttk.Frame(frame)
        label = ttk.Label(label_frame, text=text, wraplength=200)
        label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        var = tk.DoubleVar(value=default)
        
        if resolution is None:
            if (to - from_) < 10 and (from_ != int(from_) or to != int(to) or default != int(default)):
                resolution = 0.01
            else:
                resolution = 1
        
        if resolution >= 1:
            format_str = "{:.0f}"
        elif resolution >= 0.1:
            format_str = "{:.1f}"
        else:
            format_str = "{:.2f}"
        
        value_label = ttk.Label(label_frame, text=format_str.format(var.get()), width=6)
        value_label.pack(side=tk.RIGHT)
        label_frame.pack(fill=tk.X)

        def update_value_label(event=None):
            if resolution:
                rounded_value = round(var.get() / resolution) * resolution
                var.set(rounded_value)
            value_label.config(text=format_str.format(var.get()))

        # 当变量值改变时自动更新标签（修复加载预设时显示不同步的问题）
        var.trace_add('write', lambda *args: value_label.config(text=format_str.format(var.get())))

        slider = ttk.Scale(frame, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL, command=update_value_label)
        slider.bind("<ButtonRelease-1>", self.update_preview_throttled)
        slider.pack(fill=tk.X, pady=2)
        
        update_value_label()
        frame.pack(fill=tk.X, pady=3)
        return var
    
    def on_method_change(self, event):
        """切换提取方法时更新参数面板"""
        for widget in self.params_placeholder.winfo_children():
            widget.pack_forget()

        method = self.extraction_method_var.get()
        if "Pencil Sketch" in method:
            self.pencil_frame.pack(fill=tk.X)
        elif "Canny" in method:
            self.canny_frame.pack(fill=tk.X)
        elif "Anime2Sketch" in method:
            self.anime_frame.pack(fill=tk.X)
        
        if event is not None:
            self.update_preview()
    
    def load_image(self):
        """加载图片"""
        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.path.dirname(__file__), "images"),
            title="选择图片",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if file_path:
            self.file_path = file_path
            print(f"\n已加载图片: {file_path}")
            self.update_preview()
            self.start_button.config(state="normal")
    
    def update_preview_throttled(self, *args):
        """延迟更新预览（避免频繁刷新）"""
        if hasattr(self, '_preview_after_id'):
            self.root.after_cancel(self._preview_after_id)
        self._preview_after_id = self.root.after(300, self.update_preview)
    
    def update_preview(self):
        """更新预览图"""
        if not self.file_path:
            return
            
        method = self.extraction_method_var.get()
        preview_img, contours, img_w, img_h = None, None, 0, 0
        
        simplify_eps = self.simplify_epsilon.get()
        spline_s = self.spline_smoothness.get()
        preview_thick = self.preview_thickness.get()

        try:
            if "Pencil Sketch" in method:
                preview_img, contours, img_w, img_h = process_image_pencil(
                    self.file_path, self.sigma_s.get(), self.sigma_r.get(), self.shade_factor.get(),
                    simplify_eps, spline_s, preview_thick)
            
            elif "Canny" in method:
                blur_k = int(self.canny_blur.get())
                if blur_k % 2 == 0:
                    blur_k += 1
                preview_img, contours, img_w, img_h = process_image_canny(
                    self.file_path, blur_k, self.canny_low.get(), self.canny_high.get(),
                    simplify_eps, spline_s, preview_thick)
            
            elif "Anime2Sketch" in method:
                preview_img, contours, img_w, img_h = process_image_anime2sketch(
                    self.file_path, simplify_eps, spline_s, preview_thick,
                    int(self.anime_threshold.get()), int(self.anime_morph_size.get()),
                    int(self.anime_morph_iter.get()), int(self.anime_min_area.get()),
                    self.anime_contour_mode.get(), int(self.anime_pre_blur.get()),
                    self.anime_edge_enhance.get(), self.anime_sigmoid_threshold.get(),
                    self.anime_invert_var.get(), self.anime_adaptive_var.get())
        
        except Exception as e:
            print(f"Update Preview 失败: {e}")
            return

        if preview_img is None:
            return
        
        # 应用优化
        if contours and len(contours) > 0:
            contours = remove_backtracking(contours)
            
            jitter_strength = int(self.jitter_correction.get())
            if jitter_strength > 0:
                contours = apply_jitter_correction(contours, jitter_strength)
                print(f"  [抖动修正] 强度: {jitter_strength}")
            
            if self.thin_contours_var.get():
                contours = thin_contours_to_skeleton(contours, preview_img.shape)
                preview_img = cv2.cvtColor(cv2.cvtColor(preview_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                for contour in contours:
                    cv2.polylines(preview_img, [contour], False, (0, 0, 255), int(preview_thick), lineType=cv2.LINE_AA)
            
            skip_factor = int(self.skip_points.get())
            if skip_factor > 1:
                contours = apply_point_skipping(contours, skip_factor)
                preview_img = cv2.cvtColor(cv2.cvtColor(preview_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                for contour in contours:
                    cv2.polylines(preview_img, [contour], False, (0, 0, 255), int(preview_thick), lineType=cv2.LINE_AA)
            
        self.contours = contours
        self.img_w = img_w
        self.img_h = img_h

        img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
        self.preview_original_img = Image.fromarray(img)
        self.preview_zoom = 1.0
        self.update_preview_display()
    
    def on_preview_mousewheel(self, event):
        """鼠标滚轮缩放"""
        if self.preview_original_img is None:
            return
        
        if event.delta > 0:
            self.preview_zoom *= 1.1
        else:
            self.preview_zoom /= 1.1
        
        self.preview_zoom = max(0.1, min(self.preview_zoom, 5.0))
        self.update_preview_display()
    
    def on_preview_resize(self, event):
        """窗口调整大小"""
        if self.preview_original_img is not None:
            self.update_preview_display()
    
    def update_preview_display(self):
        """更新预览显示"""
        if self.preview_original_img is None:
            return
        
        img_pil = self.preview_original_img.copy()
        w, h = img_pil.size
        
        max_w = self.preview_label.winfo_width()
        max_h = self.preview_label.winfo_height()
        if max_w < 50 or max_h < 50:
            max_w, max_h = 700, 700
        
        scale = min(max_w / w, max_h / h)
        final_scale = scale * self.preview_zoom
        
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        
        if new_w > 0 and new_h > 0:
            img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(img_pil)
            self.preview_label.config(image=self.img_tk)
            
            if self.preview_zoom != 1.0:
                self.root.title(f"喝茶画画 - 缩放: {self.preview_zoom:.1f}x")
            else:
                self.root.title("喝茶画画")
    
    def start_drawing_controller(self):
        """启动绘画"""
        if not self.contours or len(self.contours) == 0:
            print("错误: 没有可绘制的轮廓")
            return
            
        self.root.iconify()
        print("GUI 已最小化。自动化将在 3 秒后开始...")
        import time
        time.sleep(3)
        
        self.start_button.config(state="disabled")
        method = self.drawing_method_var.get()
        
        speed_mult = self.speed_multiplier.get()
        
        if "方法 1" in method:
            args = (self, self.contours, self.img_w, self.img_h,
                    int(self.stroke_len.get()),
                    self.draw_delay.get() / 1000.0 / speed_mult,
                    self.lift_pause.get() / 100.0 / speed_mult,
                    speed_mult)
            threading.Thread(target=start_drawing_method_1, args=args, daemon=True).start()
        
        elif "方法 2" in method:
            args = (self, self.contours, self.img_w, self.img_h,
                    int(self.min_drag_dist.get()),
                    self.draw_delay.get() / 1000.0 / speed_mult,
                    speed_mult)
            threading.Thread(target=start_drawing_method_2, args=args, daemon=True).start()
        
        elif "方法 3" in method:
            args = (self, self.contours, self.img_w, self.img_h,
                    self.draw_delay.get() / 1000.0 / speed_mult,
                    self.lift_pause.get() / 100.0 / speed_mult,
                    int(self.hand_shake.get()),
                    self.think_pause.get(),
                    speed_mult)
            threading.Thread(target=start_drawing_method_3, args=args, daemon=True).start()
    
    def on_drawing_complete(self):
        """绘画完成回调"""
        def restore_gui():
            print("...绘画完成，恢复 GUI...")
            self.start_button.config(state="normal")
            self.root.deiconify()
        
        self.root.after(0, restore_gui)
    
    def get_all_slider_values(self):
        """获取所有参数值"""
        return {
            "extraction_method": self.extraction_method_var.get(),
            "sigma_s": self.sigma_s.get(),
            "sigma_r": self.sigma_r.get(),
            "shade_factor": self.shade_factor.get(),
            "canny_blur": self.canny_blur.get(),
            "canny_low": self.canny_low.get(),
            "canny_high": self.canny_high.get(),
            "simplify_epsilon": self.simplify_epsilon.get(),
            "preview_thickness": self.preview_thickness.get(),
            "spline_smoothness": self.spline_smoothness.get(),
            "jitter_correction": self.jitter_correction.get(),
            "stroke_len": self.stroke_len.get(),
            "min_drag_dist": self.min_drag_dist.get(),
            "draw_delay": self.draw_delay.get(),
            "drawing_method": self.drawing_method_var.get(),
            "anime_threshold": self.anime_threshold.get(),
            "anime_morph_size": self.anime_morph_size.get(),
            "anime_morph_iter": self.anime_morph_iter.get(),
            "anime_min_area": self.anime_min_area.get(),
            "anime_contour_mode": self.anime_contour_mode.get(),
            "anime_pre_blur": self.anime_pre_blur.get(),
            "anime_edge_enhance": self.anime_edge_enhance.get(),
            "anime_sigmoid_threshold": self.anime_sigmoid_threshold.get(),
            "anime_invert": self.anime_invert_var.get(),
            "anime_adaptive": self.anime_adaptive_var.get(),
            "thin_contours": self.thin_contours_var.get(),
            "skip_points": self.skip_points.get(),
            "hand_shake": self.hand_shake.get(),
            "think_pause": self.think_pause.get(),
            "lift_pause": self.lift_pause.get(),
            "speed_multiplier": self.speed_multiplier.get()
        }
    
    def set_all_slider_values(self, values):
        """设置所有参数值"""
        self.extraction_method_var.set(values.get("extraction_method", "Pencil Sketch (V19)"))
        self.sigma_s.set(values.get("sigma_s", 60))
        self.sigma_r.set(values.get("sigma_r", 0.4))
        self.shade_factor.set(values.get("shade_factor", 0.05))
        self.canny_blur.set(values.get("canny_blur", 3))
        self.canny_low.set(values.get("canny_low", 50))
        self.canny_high.set(values.get("canny_high", 150))
        self.simplify_epsilon.set(values.get("simplify_epsilon", 1.0))
        self.preview_thickness.set(values.get("preview_thickness", 1))
        self.spline_smoothness.set(values.get("spline_smoothness", 0))
        self.jitter_correction.set(values.get("jitter_correction", 0))
        self.stroke_len.set(values.get("stroke_len", 15))
        self.min_drag_dist.set(values.get("min_drag_dist", 5))
        self.draw_delay.set(values.get("draw_delay", 5))
        self.drawing_method_var.set(values.get("drawing_method", "方法 1: 短行程 (推荐)"))
        self.anime_threshold.set(values.get("anime_threshold", 127))
        self.anime_morph_size.set(values.get("anime_morph_size", 2))
        self.anime_morph_iter.set(values.get("anime_morph_iter", 1))
        self.anime_min_area.set(values.get("anime_min_area", 10))
        self.anime_contour_mode.set(values.get("anime_contour_mode", "外部轮廓 (快速)"))
        self.anime_pre_blur.set(values.get("anime_pre_blur", 0))
        self.anime_edge_enhance.set(values.get("anime_edge_enhance", 0))
        self.anime_sigmoid_threshold.set(values.get("anime_sigmoid_threshold", 0.5))
        self.anime_invert_var.set(values.get("anime_invert", False))
        self.anime_adaptive_var.set(values.get("anime_adaptive", False))
        self.thin_contours_var.set(values.get("thin_contours", False))
        self.skip_points.set(values.get("skip_points", 1))
        self.hand_shake.set(values.get("hand_shake", 1))
        self.think_pause.set(values.get("think_pause", 3))
        self.lift_pause.set(values.get("lift_pause", 5))
        self.speed_multiplier.set(values.get("speed_multiplier", 1.0))

    def save_presets(self):
        """保存预设"""
        values = self.get_all_slider_values()
        try:
            with open(self.preset_file, 'w') as f:
                json.dump(values, f, indent=4)
            print(f"预设已保存到 {self.preset_file}")
        except Exception as e:
            print(f"保存预设失败: {e}")

    def load_presets(self, silent=False):
        """加载预设"""
        if not os.path.exists(self.preset_file):
            if not silent:
                print(f"未找到预设文件: {self.preset_file}")
            return
            
        try:
            with open(self.preset_file, 'r') as f:
                values = json.load(f)
            self.set_all_slider_values(values)
            if not silent:
                print(f"预设已从 {self.preset_file} 加载")
        except Exception as e:
            if not silent:
                print(f"加载预设失败: {e}")

    def reset_sliders(self):
        """重置参数"""
        self.set_all_slider_values({})
        self.on_method_change(None)
        self.update_preview()


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
