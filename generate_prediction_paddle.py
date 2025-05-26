import json
import os
import zipfile
import re
from difflib import SequenceMatcher

def get_file_id(path):
    """从文件路径中提取ID
    支持两种格式：
    1. 完整文件名（如 "2101.jpg"）
    2. 纯ID格式（如 "2101"）
    """
    # 移除所有文件扩展名，只保留ID部分
    base_name = os.path.basename(path)
    # 移除扩展名，提取ID
    file_id = re.sub(r'\.[^.]*$', '', base_name)
    return file_id

def find_modified_chars(source, target):
    """找出从source到target的转换中所有修改的字符及其位置，并评估错误的典型性"""
    modified = []
    matcher = SequenceMatcher(None, source, target)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ['replace']:  # 包含替换和删除的情况
            for pos in range(i1, i2):
                if pos < len(source):
                    original_char = source[pos]
                    # 如果有对应的替换字符
                    if pos - i1 + j1 < len(target):
                        corrected_char = target[pos - i1 + j1] if j1 < len(target) else ''
                        error_score = calculate_error_typicality(original_char, corrected_char, source, pos)
                        modified.append((pos, original_char, corrected_char, error_score))
                    else:
                        # 删除的情况
                        error_score = calculate_error_typicality(original_char, '', source, pos)
                        modified.append((pos, original_char, '', error_score))
    
    # 按错误典型性得分排序，得分越高越典型
    modified.sort(key=lambda x: x[3], reverse=True)
    return modified

def calculate_error_typicality(original_char, corrected_char, context, pos):
    """计算错误的典型性得分，得分越高越可能是真正的错误"""
    score = 0
    
    # 1. 形似字错误（OCR常见错误）- 高分
    similar_pairs = {
        '已': '己', '己': '已', '日': '目', '目': '日', 
        '天': '夭', '夭': '天', '人': '入', '入': '人',
        '木': '未', '未': '木', '土': '士', '士': '土',
        '刀': '力', '力': '刀', '几': '儿', '儿': '几',
        '马': '鸟', '鸟': '马', '千': '干', '干': '千',
        '王': '主', '主': '王', '由': '田', '田': '由'
    }
    if original_char in similar_pairs and corrected_char == similar_pairs[original_char]:
        score += 100
    
    # 2. 音近字错误 - 中高分
    phonetic_pairs = {
        '的': '得', '得': '的', '在': '再', '再': '在',
        '做': '作', '作': '做', '像': '象', '象': '像',
        '以': '已', '已': '以', '那': '哪', '哪': '那'
    }
    if original_char in phonetic_pairs and corrected_char == phonetic_pairs[original_char]:
        score += 80
    
    # 3. 繁简字错误 - 中分
    traditional_simplified = {
        '実': '实', '発': '发', '図': '图', '対': '对',
        '説': '说', '経': '经', '學': '学', '師': '师'
    }
    if original_char in traditional_simplified and corrected_char == traditional_simplified[original_char]:
        score += 60
    
    # 4. 标点符号错误 - 低分
    if original_char in '，。！？；：' and corrected_char in '，。！？；：':
        score += 20
    
    # 5. 数字字母错误 - 中低分
    if (original_char.isdigit() and corrected_char.isdigit()) or \
       (original_char.isalpha() and corrected_char.isalpha()):
        score += 30
    
    # 6. 位置权重：句子中间的错误比开头结尾更典型
    if 0.2 * len(context) < pos < 0.8 * len(context):
        score += 10
    
    # 7. 如果原字符是常见错字，增加得分
    common_errors = '己日目天人入木未土士刀力几儿马鸟千干王主由田'
    if original_char in common_errors:
        score += 15
    
    return score

def get_char_bbox(char_info):
    """从字符信息中提取bbox并转换为指定格式"""
    bbox = char_info['bbox']
    return {
        "start_x": bbox[0],
        "end_x": bbox[2],
        "start_y": bbox[1],
        "end_y": bbox[3]
    }

def process_corrected_file(file_path, bbox_path):
    """处理单个纠错文件，合并所有预测句子，并收集修改字符的bbox信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(bbox_path, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)
        
    sentences = []
    bounding_box_list = []
    
    if 'corrected_text_list' in data:
        for item in data['corrected_text_list']:
            source = item['source_sentence']
            target = item['predict_sentence']
            sentences.append(target)
            
            # 找出修改的字符并按典型性排序
            modified_chars = find_modified_chars(source, target)
            
            # 在bbox数据中找到对应句子
            sentence_id = item.get('sentence_id')
            if sentence_id is not None:
                for sentence in bbox_data['sentences']:
                    if sentence['sentence_id'] == sentence_id:
                        # 对于每个修改的字符，找到其bbox
                        for char_pos, original_char, corrected_char, error_score in modified_chars:
                            if char_pos < len(sentence['chars']):
                                char_info = sentence['chars'][char_pos]
                                if char_info['char'] == original_char:  # 确保字符匹配
                                    bbox = get_char_bbox(char_info)
                                    # 添加错误得分信息到bbox中
                                    bbox['error_score'] = error_score
                                    bbox['original_char'] = original_char
                                    bbox['corrected_char'] = corrected_char
                                    bounding_box_list.append(bbox)
                        break
    
    # 按错误典型性得分排序，只保留前三个最典型的错误
    bounding_box_list.sort(key=lambda x: x.get('error_score', 0), reverse=True)
    # 清理辅助信息，只保留bbox坐标
    top_bboxes = []
    for bbox in bounding_box_list[:3]:
        clean_bbox = {
            "start_x": bbox["start_x"],
            "end_x": bbox["end_x"],
            "start_y": bbox["start_y"],
            "end_y": bbox["end_y"]
        }
        top_bboxes.append(clean_bbox)
    
    return ' '.join(sentences), top_bboxes

def main():
    # 读取test_data.json
    with open('data/test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    processed_count = 0
    
    # 处理每个文件
    for item in test_data:
        # 从path中提取ID
        file_id = get_file_id(item['path'])
        corrected_file_path = os.path.join('data/paddleocr_version/ocr_corrected', file_id + '.json')
        bbox_file_path = os.path.join('data/paddleocr_version/bbox_washed', file_id + '.json')
        
        if os.path.exists(corrected_file_path) and os.path.exists(bbox_file_path):
            # 处理文件并获取预测文本和bbox列表
            predict_text, bounding_box_list = process_corrected_file(corrected_file_path, bbox_file_path)
            item['predict_text'] = predict_text
            item['bounding_box_list'] = bounding_box_list
            
            # 如果source_text为空，从纠错文件中获取原始文本
            if not item['source_text'] and os.path.exists(corrected_file_path):
                with open(corrected_file_path, 'r', encoding='utf-8') as f:
                    corrected_data = json.load(f)
                    if 'corrected_text_list' in corrected_data:
                        source_sentences = [item['source_sentence'] for item in corrected_data['corrected_text_list']]
                        item['source_text'] = ' '.join(source_sentences)
            
            processed_count += 1
            print(f"处理文件: {file_id}, 文本长度: {len(item['predict_text'])}, bbox数量: {len(item['bounding_box_list'])}")
    
    # 保存结果
    with open('./output/predict_paddle.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    print("Paddle预测结果已保存到 ./output/predict_paddle.json")

if __name__ == '__main__':
    main() 