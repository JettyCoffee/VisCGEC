#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理图像OCR并生成单字级别的可视化和JSON结果
"""

import os
import sys
import cv2
import numpy as np
from copy import deepcopy
import json
import glob
import random
from tqdm import tqdm
import argparse
from pathlib import Path

# 添加PaddleOCR路径
paddle_ocr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/PaddleOCR")
sys.path.insert(0, paddle_ocr_path)

from paddleocr import PaddleOCR

# 为每个字符分配不同颜色的函数
def get_random_color():
    """生成随机颜色 (B,G,R)"""
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )

def generate_distinct_colors(n):
    """生成n个尽可能不同的颜色"""
    colors = []
    for i in range(n):
        # 使用HSV色彩空间，均匀分布色调
        hue = i / n
        saturation = 0.9
        value = 0.9
        
        # 转换到RGB
        h = hue * 360
        s = saturation
        v = value
        
        hi = int(h / 60) % 6
        f = h / 60 - int(h / 60)
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if hi == 0:
            r, g, b = v, t, p
        elif hi == 1:
            r, g, b = q, v, p
        elif hi == 2:
            r, g, b = p, v, t
        elif hi == 3:
            r, g, b = p, q, v
        elif hi == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        # 转换为0-255范围的BGR格式
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    
    return colors

def visualize_char_boxes(image, boxes, texts, output_path):
    """
    可视化单字级别文本检测框，每个字符使用不同颜色
    
    参数:
        image: 原始图像
        boxes: 检测框坐标列表
        texts: 文本内容列表
        output_path: 输出图像路径
    """
    img_copy = deepcopy(image)
    
    # 生成足够多不同的颜色
    colors = generate_distinct_colors(len(texts))
    
    # 绘制检测框
    for i, box in enumerate(boxes):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        
        # 使用不同颜色绘制每个字符的框
        cv2.polylines(img_copy, [box], True, color=colors[i % len(colors)], thickness=1)
        
        # 在框的中心位置添加文本
        if i < len(texts):
            text = texts[i]
            x_center = np.mean(box[:, 0])
            y_center = np.mean(box[:, 1])
            cv2.putText(img_copy, text, 
                        (int(x_center-5), int(y_center+5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i % len(colors)], 1)
    
    cv2.imwrite(output_path, img_copy)
    print(f"已保存可视化结果到: {output_path}")
    return img_copy

def split_text_to_chars(box, text):
    """
    将文本行分解为单个字符，并为每个字符计算其对应的包围盒
    
    参数:
        box: 文本行的包围盒 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text: 文本行内容
        
    返回:
        char_boxes: 每个字符的包围盒列表
        chars: 字符列表
    """
    if not text or len(text) == 0:
        return [], []
    
    # 转换为numpy数组
    box = np.array(box).astype(np.float32)
    
    # 获取框的四个角点
    tl, tr, br, bl = box
    
    # 计算上边和下边
    top_line = tr - tl  # 右上 - 左上 = 上边向量
    bottom_line = br - bl  # 右下 - 左下 = 下边向量
    
    # 字符数量
    num_chars = len(text)
    
    # 结果列表
    char_boxes = []
    chars = []
    
    # 为每个字符分配包围盒
    for i in range(num_chars):
        # 计算上边上的点
        t_ratio = (i / num_chars, (i + 1) / num_chars)
        p_top_left = tl + top_line * t_ratio[0]
        p_top_right = tl + top_line * t_ratio[1]
        
        # 计算下边上的点
        p_bottom_left = bl + bottom_line * t_ratio[0]
        p_bottom_right = bl + bottom_line * t_ratio[1]
        
        # 构建字符包围盒
        char_box = np.array([
            p_top_left, p_top_right, 
            p_bottom_right, p_bottom_left
        ]).astype(np.float32)
        
        char_boxes.append(char_box.tolist())
        chars.append(text[i])
    
    return char_boxes, chars

def extract_char_level_boxes(boxes, texts):
    """
    从行级文本框中提取字符级别的文本框
    
    参数:
        boxes: 行级文本框列表
        texts: 对应的文本内容列表
        
    返回:
        char_boxes: 字符级别的文本框列表
        chars: 对应的字符列表
    """
    all_char_boxes = []
    all_chars = []
    
    for box, text in zip(boxes, texts):
        char_boxes, chars = split_text_to_chars(box, text)
        all_char_boxes.extend(char_boxes)
        all_chars.extend(chars)
        
    return all_char_boxes, all_chars

def process_image(image_path, output_dir, ocr_model):
    """
    处理单个图像的OCR识别
    
    参数:
        image_path: 图像路径
        output_dir: 输出目录
        ocr_model: OCR模型实例
        
    返回:
        结果字典
    """
    # 获取文件名（不含扩展名）作为doc_id
    doc_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 执行OCR识别
    result = ocr_model.ocr(image, cls=False)
    
    if not result or not result[0]:
        print(f"无法识别图像中的文本: {image_path}")
        return None
    
    # 提取行级别框和文本
    boxes = [line[0] for line in result[0]]
    texts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]
    
    # 拼接所有文本
    source_text = "".join(texts)
    
    # 提取字符级别的框和文本
    char_boxes, chars = extract_char_level_boxes(boxes, texts)
    
    # 可视化字符级别OCR结果
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    char_vis_path = os.path.join(vis_dir, f"{doc_id}_char_ocr.jpg")
    visualize_char_boxes(image, char_boxes, chars, char_vis_path)
    
    # 构造输出结果
    char_boxes_result = []
    for i, (box, char) in enumerate(zip(char_boxes, chars)):
        # 转换为[x, y, w, h]格式
        box_np = np.array(box)
        x_min = float(np.min(box_np[:, 0]))
        y_min = float(np.min(box_np[:, 1]))
        x_max = float(np.max(box_np[:, 0]))
        y_max = float(np.max(box_np[:, 1]))
        
        char_boxes_result.append({
            "char": char,
            "bbox": [
                x_min,
                y_min,
                x_max,
                y_max
            ]
        })
    
    # 构造最终结果
    result_data = {
        "doc_id": doc_id,
        "result_count": 1,
        "results": [
            {
                "source_text": source_text,
                "char_count": len(chars),
                "char_boxes": char_boxes_result
            }
        ]
    }
    
    return result_data

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量处理图像OCR并生成单字级别的可视化和JSON结果')
    parser.add_argument('--input_dir', type=str, default='data/test_img_data', 
                        help='输入图像目录路径')
    parser.add_argument('--output_dir', type=str, default='data/paddleocr_version/ocr_output', 
                        help='输出结果目录路径')
    parser.add_argument('--summary_dir', type=str, default='data/paddleocr_version/ocr_summary', 
                        help='汇总JSON文件的输出目录')
    args = parser.parse_args()
    
    # 获取输入目录中的所有图像文件
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.input_dir)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    summary_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.summary_dir)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # 排序以确保结果一致
    image_files.sort()
    
    print(f"找到 {len(image_files)} 个图像文件，开始处理...")
    
    # 创建OCR模型（只创建一次，避免重复加载）
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")
    
    # 批量处理图像
    for image_path in tqdm(image_files):
        try:
            # 处理图像
            result_data = process_image(image_path, output_dir, ocr)
            
            if result_data:
                # 保存JSON结果
                doc_id = result_data["doc_id"]
                summary_path = os.path.join(summary_dir, f"{doc_id}_results.json")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                
                print(f"已处理: {image_path} -> {json_path}")
            else:
                print(f"处理失败: {image_path}")
        except Exception as e:
            print(f"处理 {image_path} 时出错: {str(e)}")
    
    print(f"处理完成。结果保存在: {output_dir} 和 {summary_dir}")

if __name__ == "__main__":
    main()
