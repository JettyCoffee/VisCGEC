import json
import os
import zipfile
import re

def get_file_id(path):
    """从文件路径中提取ID
    支持两种格式：
    1. 完整文件名（如 "2101.jpg"）
    2. 纯ID格式（如 "2101"）
    """
    # 移除所有文件扩展名
    base_name = os.path.basename(path)
    return base_name

def process_corrected_file(file_path):
    """处理单个纠错文件，合并所有预测句子"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 合并所有预测句子
    if 'corrected_text_list' in data:
        sentences = [item['predict_sentence'] for item in data['corrected_text_list']]
        return ' '.join(sentences)
    return ''

def main():
    # 读取test_data.json
    with open('data/test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 处理每个文件
    for item in test_data:
        # 从path中提取ID
        file_id = get_file_id(item['path'])
        file_path = os.path.join('data/got_version/ocr_corrected', file_id + '.json')
        
        if os.path.exists(file_path):
            # 只更新predict_text，保留原有的source_text
            item['predict_text'] = process_corrected_file(file_path)
            
            # 如果source_text为空，从纠错文件中获取原始文本
            if not item['source_text'] and os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    corrected_data = json.load(f)
                    if 'corrected_text_list' in corrected_data:
                        source_sentences = [item['source_sentence'] for item in corrected_data['corrected_text_list']]
                        item['source_text'] = ' '.join(source_sentences)
            
            # 打印处理进度
            print(f"处理文件: {file_id}, 文本长度: {len(item['predict_text'])}")
    
    # 保存结果
    with open('./output/predict_got.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    print("GOT预测结果已保存到 ./output/predict_got.json")

if __name__ == '__main__':
    main() 