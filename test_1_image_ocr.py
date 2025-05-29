import os
import sys
import argparse
from pathlib import Path
import logging
from ocr_processor_paddle import OCRProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser(description='处理单张图片的OCR识别')
    parser.add_argument('--image_path', type=str, required=True, help='输入图片的路径')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用的GPU ID')
    return parser.parse_args()

def main():
    # 设置输入图片路径
    image_path = Path("./data/preprocessed_img/2202.jpg")
    if not image_path.exists():
        logging.error(f"图片文件不存在: {image_path}")
        return
    
    # 设置输出目录
    output_dir = Path("output/ocr_result")
    
    # 创建OCR处理器实例
    processor = OCRProcessor(
        input_dir=image_path.parent,
        output_dir=output_dir,
        max_workers_per_gpu=1,  # 单张图片只需要一个线程
        gpu_ids=[0]  # 使用指定的GPU
    )
    
    # 处理单张图片
    processor.process_single_image((image_path, 0))
    
    logging.info(f"图片处理完成，结果保存在: {output_dir / image_path.stem}")

if __name__ == "__main__":
    main()
