#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的单字级别OCR结果解析器
能够更准确地从表格和图片OCR结果中提取每个字符的坐标框
"""

import json
import os
from typing import Dict, List, Any
from pathlib import Path
import logging
from bs4 import BeautifulSoup

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ImprovedCharParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_table_ocr_result(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析表格OCR结果，提取单字级别的文字框
        
        Args:
            ocr_data: OCR识别结果数据
            
        Returns:
            包含单字文字框的解析结果
        """
        if not ocr_data or 'res' not in ocr_data:
            return {}
        
        res_data = ocr_data['res']
        
        # 提取HTML表格中的文本
        html_content = res_data.get('html', '')
        table_text = self.extract_text_from_html(html_content)
        
        # 提取单元格边界框
        cell_bboxes = res_data.get('cell_bbox', [])
        
        # 分析表格结构并提取字符框
        char_boxes = self.extract_char_boxes_improved(html_content, cell_bboxes)
        
        # 构建结果
        result = {
            'source_text': table_text,
            'char_count': len([c for c in table_text if c.strip()]),
            'char_boxes': char_boxes
        }
        
        return result
    
    def parse_figure_ocr_result(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析图片OCR结果，提取单字级别的文字框
        
        Args:
            ocr_data: OCR识别结果数据
            
        Returns:
            包含单字文字框的解析结果
        """
        if not ocr_data or 'res' not in ocr_data:
            return {}
        
        all_text = ""
        char_boxes = []
        
        # figure类型数据可能直接在res中或在res内的数组中
        if isinstance(ocr_data['res'], list):
            res_data = ocr_data['res']
        else:
            res_data = [ocr_data['res']]
        
        # 处理所有文本区域
        for text_item in res_data:
            if 'text' in text_item:
                text = text_item.get('text', '')
                all_text += text
                
                words = text_item.get('text_word', [])
                word_regions = text_item.get('text_word_region', [])
                
                # 去重处理 - 有些OCR结果中word_regions会重复
                unique_word_regions = []
                processed_indices = []
                
                for i, region in enumerate(word_regions):
                    if i < len(words) and i not in processed_indices:
                        unique_word_regions.append((words[i], region))
                        processed_indices.append(i)
                
                # 确保words和word_regions长度一致且不为空
                for word, region in unique_word_regions:
                    # 计算字符宽度
                    if word and len(region) >= 4:
                        # 从四个点中提取边界框
                        x_coords = [point[0] for point in region]
                        y_coords = [point[1] for point in region]
                        
                        # 计算边界框
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        
                        word_width = x2 - x1
                        
                        # 每个字符分配合适的宽度
                        if len(word) > 0:
                            char_width = word_width / len(word)
                            
                            # 为每个字符创建边界框
                            for i, char in enumerate(word):
                                if char.strip():  # 只处理非空白字符
                                    char_x1 = x1 + i * char_width
                                    char_x2 = char_x1 + char_width
                                    
                                    char_box = {
                                        "char": char,
                                        "bbox": [round(char_x1, 2), round(y1, 2), round(char_x2, 2), round(y2, 2)]
                                    }
                                    char_boxes.append(char_box)
        
        # 构建结果
        result = {
            'source_text': all_text,
            'char_count': len(char_boxes),
            'char_boxes': char_boxes
        }
        
        return result
    
    def extract_text_from_html(self, html_content: str) -> str:
        """
        从HTML表格中提取所有文本内容
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取所有td标签中的文本
            text_parts = []
            for td in soup.find_all('td'):
                text = td.get_text(strip=True)
                if text and text not in ['', ' ']:
                    text_parts.append(text)
            
            # 将文本片段连接
            full_text = ''.join(text_parts)
            return full_text
            
        except Exception as e:
            self.logger.warning(f"HTML解析失败: {e}")
            return ""
    
    def extract_char_boxes_improved(self, html_content: str, cell_bboxes: List[List[float]]) -> List[Dict[str, Any]]:
        """
        改进的字符框提取方法，仅返回char和bbox
        """
        char_boxes = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 获取所有行
            rows = soup.find_all('tr')
            
            cell_index = 0
            
            for row in rows:
                cells = row.find_all('td')
                
                for cell in enumerate(cells):
                    cell_text = cell[1].get_text(strip=True)
                    
                    # 跳过空单元格
                    if not cell_text or cell_index >= len(cell_bboxes):
                        cell_index += 1
                        continue
                    
                    # 获取对应的边界框
                    bbox = cell_bboxes[cell_index]
                    
                    if len(bbox) >= 8:
                        # 将8个坐标点转换为矩形边界框
                        x_coords = [bbox[i] for i in range(0, 8, 2)]
                        y_coords = [bbox[i] for i in range(1, 8, 2)]
                        cell_bbox_rect = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        
                        # 计算单元格的宽度
                        cell_width = cell_bbox_rect[2] - cell_bbox_rect[0]
                        
                        # 为每个字符分配坐标
                        char_count = len(cell_text)
                        if char_count > 0:
                            char_width = cell_width / char_count
                            
                            for i, char in enumerate(cell_text):
                                if char.strip():  # 只处理非空白字符
                                    # 计算字符的边界框
                                    char_x1 = cell_bbox_rect[0] + i * char_width
                                    char_x2 = char_x1 + char_width
                                    char_y1 = cell_bbox_rect[1]
                                    char_y2 = cell_bbox_rect[3]
                                    
                                    char_box = {
                                        "char": char,
                                        "bbox": [round(char_x1, 2), round(char_y1, 2), round(char_x2, 2), round(char_y2, 2)]
                                    }
                                    char_boxes.append(char_box)
                    
                    cell_index += 1
            
        except Exception as e:
            self.logger.error(f"提取字符框失败: {e}")
        
        return char_boxes
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        保存解析结果到JSON文件
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"单字解析结果已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")

def process_ocr_output(input_dir: str, output_dir: str):
    """
    批量处理OCR输出目录下的所有文件
    
    Args:
        input_dir: OCR输出目录路径
        output_dir: 结果保存目录路径
    """
    parser = ImprovedCharParser()
    
    # 遍历所有文档目录
    for doc_id in os.listdir(input_dir):
        doc_path = os.path.join(input_dir, doc_id)
        if not os.path.isdir(doc_path):
            continue
            
        # 修正res_0.txt文件路径
        res_file = os.path.join(doc_path, "structure", doc_id, "res_0.txt")
        if not os.path.exists(res_file):
            logging.warning(f"找不到文件: {res_file}")
            continue
            
        logging.info(f"处理文档 {doc_id}")
        
        try:
            # 读取OCR结果 - 修改为按行读取
            ocr_results = []
            with open(res_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        line = line.strip()
                        if line:  # 跳过空行
                            ocr_data = json.loads(line)
                            ocr_results.append(ocr_data)
                    except json.JSONDecodeError as e:
                        logging.warning(f"解析JSON行失败: {e}, 行内容: {line[:100]}...")
                        continue
            
            # 处理OCR结果
            parsed_results = []
            for result in ocr_results:
                result_type = result.get('type', '')
                if result_type == 'table':
                    parsed_result = parser.parse_table_ocr_result(result)
                    if parsed_result:
                        parsed_results.append(parsed_result)
                elif result_type == 'figure':
                    parsed_result = parser.parse_figure_ocr_result(result)
                    if parsed_result:
                        parsed_results.append(parsed_result)
            
            # 保存结果
            output_path = os.path.join(output_dir, f"{doc_id}_results.json")
            final_results = {
                'doc_id': doc_id,
                'result_count': len(parsed_results),
                'results': parsed_results
            }
            parser.save_results(final_results, output_path)
            
        except Exception as e:
            logging.error(f"处理文档 {doc_id} 时出错: {e}")

def main():
    """主函数"""
    # 修正输入输出目录路径
    input_dir = "data/paddleocr_version/ocr_output"
    output_dir = "data/paddleocr_version/ocr_summary"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有文件
    process_ocr_output(input_dir, output_dir)
    
    print("\n处理完成!")

if __name__ == "__main__":
    main() 