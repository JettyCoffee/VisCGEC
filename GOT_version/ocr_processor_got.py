#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
精简版OCR处理程序 - 使用GOT-OCR2.0识别图像中的文字并输出JSON
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List

from transformers import AutoModel, AutoTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """精简版OCR处理器类，只输出文本内容"""
    
    def __init__(self, use_gpu=True, lang='ch', use_angle_cls=False, det_db_thresh=0.3, rec_thresh=0.5):
        """
        初始化OCR处理器
        
        Args:
            use_gpu: 是否使用GPU
            lang: 语言选项，默认中文
            use_angle_cls: 是否使用方向分类器
            det_db_thresh: 检测阈值
            rec_thresh: 识别阈值
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('models/GOT-OCR2_0', trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                'models/GOT-OCR2_0',
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map='cuda' if use_gpu else 'cpu',
                use_safetensors=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            self.model = self.model.eval()
            if use_gpu:
                self.model = self.model.cuda()
            logger.info(f"GOT-OCR2.0模型已加载，运行设备: {'GPU' if use_gpu else 'CPU'}")
        except Exception as e:
            logger.error(f"加载GOT-OCR2.0模型失败: {e}")
            raise
    
    def process_image(self, image_path: str) -> Dict[str, str]:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果字典，只包含文本内容
        """
        logger.info(f"处理图像: {image_path}")
        
        try:
            # 使用GOT-OCR2.0进行OCR识别
            result = self.model.chat(self.tokenizer, image_path, ocr_type='ocr')
            return {
                "source_text": result.strip()
            }
        except Exception as e:
            logger.error(f"OCR处理失败: {e}")
            return {"source_text": ""}


def process_json_data(json_path: str, img_dir: str, ocr_processor: OCRProcessor) -> List[Dict[str, Any]]:
    """
    处理JSON中定义的所有图像
    
    Args:
        json_path: JSON文件路径
        img_dir: 图像目录
        ocr_processor: OCR处理器实例
        
    Returns:
        处理结果列表
    """
    logger.info(f"读取数据文件: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"读取JSON文件失败: {e}")
        return []
    
    results = []
    for i, item in enumerate(data):
        image_filename = item.get('path', '')
        image_path = os.path.join(img_dir, image_filename)
        
        if not os.path.exists(image_path):
            logger.warning(f"图像文件不存在: {image_path}")
            results.append({
                'fk_homework_id': item.get('fk_homework_id', ''),
                'path': image_filename,
                'source_text': ''
            })
            continue
        
        logger.info(f"处理第 {i+1}/{len(data)} 张图像: {image_path}")
        result = ocr_processor.process_image(image_path)
        
        # 更新结果
        updated_item = {
            'fk_homework_id': item.get('fk_homework_id', ''),
            'path': image_filename,
            'source_text': result.get('source_text', '')
        }
        results.append(updated_item)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='中文图像OCR处理（精简版）')
    parser.add_argument('--input', type=str, default='data/test_data.json', help='输入json文件')
    parser.add_argument('--output', type=str, default='data/got_version/ocr_output', help='输出目录')
    parser.add_argument('--img-dir', type=str, default='data/preprocessed_img', help='图像目录')
    parser.add_argument('--no-gpu', action='store_true', help='不使用GPU')
    parser.add_argument('--det-thresh', type=float, default=0.3, help='检测阈值')
    parser.add_argument('--rec-thresh', type=float, default=0.5, help='识别阈值')
    args = parser.parse_args()
    
    # 初始化OCR处理器
    ocr_processor = OCRProcessor(
        use_gpu=not args.no_gpu,
        det_db_thresh=args.det_thresh,
        rec_thresh=args.rec_thresh
    )
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 处理输入
    if args.input.endswith('.json'):
        # 处理JSON中的所有图像
        results = process_json_data(args.input, args.img_dir, ocr_processor)
        
        # 为每个图像创建单独的JSON文件
        for result in results:
            image_filename = result['path']
            output_filename = os.path.join(args.output, f"{image_filename}.json")
            
            # 保存单个结果
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存结果到: {output_filename}")
            
        logger.info(f"OCR处理完成，所有结果已保存到目录: {args.output}")
        
    elif os.path.isfile(args.input):
        # 处理单张图像
        result = ocr_processor.process_image(args.input)
        
        # 生成输出文件名
        image_filename = os.path.basename(args.input)
        output_filename = os.path.join(args.output, f"{image_filename}.json")
        
        # 保存结果
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"OCR处理完成，结果已保存到: {output_filename}")
        
    else:
        logger.error(f"无效的输入: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main() 