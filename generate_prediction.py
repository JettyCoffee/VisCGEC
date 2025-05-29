import json
import os
import zipfile
import re
from difflib import SequenceMatcher
import numpy as np

# 全局配置：bbox数量限制
MAX_BBOX_LIMIT = 1  # 可以通过修改这个数值统一控制所有函数的bbox数量限制

def get_file_id(path):
    """从文件路径中提取ID"""
    base_name = os.path.basename(path)
    return re.sub(r'\.[^.]*$', '', base_name)

def calculate_bbox_iou(box1, box2):
    """
    计算两个bbox的IOU（Intersection over Union）
    Args:
        box1, box2: 字典格式 {start_x, start_y, end_x, end_y}
    Returns:
        float: IOU值，范围[0, 1]
    """
    x_left = max(box1["start_x"], box2["start_x"])
    y_top = max(box1["start_y"], box2["start_y"])
    x_right = min(box1["end_x"], box2["end_x"])
    y_bottom = min(box1["end_y"], box2["end_y"])

    # 如果没有重叠
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1["end_x"] - box1["start_x"]) * (box1["end_y"] - box1["start_y"])
    area2 = (box2["end_x"] - box2["start_x"]) * (box2["end_y"] - box2["start_y"])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def enhanced_bbox_selection(bbox_candidates, corrected_char_positions, strategy="confidence_weighted"):
    """
    增强的bbox选择策略
    Args:
        bbox_candidates: 候选bbox列表
        corrected_char_positions: 纠错字符位置列表
        strategy: 选择策略类型
    Returns:
        list: 选择的bbox列表
    """
    if not bbox_candidates or not corrected_char_positions:
        return []
    
    selected_bboxes = []
    
    if strategy == "confidence_weighted":
        # 基于置信度和位置匹配的加权选择
        for char_pos in corrected_char_positions:
            best_bbox = None
            best_score = 0.0
            
            for bbox in bbox_candidates:
                # 计算位置匹配度
                position_score = calculate_position_match(char_pos, bbox)
                # 如果有置信度信息，加入加权
                confidence = bbox.get('confidence', 0.8)
                total_score = position_score * confidence
                
                if total_score > best_score:
                    best_score = total_score
                    best_bbox = bbox
            
            if best_bbox and best_score > 0.3:  # 阈值过滤
                selected_bboxes.append(best_bbox)
    
    elif strategy == "distance_based":
        # 基于距离的选择策略
        for char_pos in corrected_char_positions:
            min_distance = float('inf')
            best_bbox = None
            
            for bbox in bbox_candidates:
                distance = calculate_bbox_distance(char_pos, bbox)
                if distance < min_distance:
                    min_distance = distance
                    best_bbox = bbox
            
            if best_bbox and min_distance < 50:  # 距离阈值
                selected_bboxes.append(best_bbox)
    
    return selected_bboxes[:MAX_BBOX_LIMIT]  # 使用全局变量控制bbox数量

def multi_strategy_bbox_selection(bbox_candidates, corrected_char_positions):
    """
    多策略bbox选择，结合多种策略提高准确性
    Args:
        bbox_candidates: 候选bbox列表
        corrected_char_positions: 纠错字符位置列表
    Returns:
        list: 最终选择的bbox列表
    """
    if not bbox_candidates or not corrected_char_positions:
        return []
    
    # 策略1: 置信度加权选择
    strategy1_results = enhanced_bbox_selection(bbox_candidates, corrected_char_positions, "confidence_weighted")
    
    # 策略2: 距离基础选择
    strategy2_results = enhanced_bbox_selection(bbox_candidates, corrected_char_positions, "distance_based")
    
    # 策略3: IOU基础的重叠检测
    strategy3_results = []
    for char_pos in corrected_char_positions:
        candidate_bboxes = []
        for bbox in bbox_candidates:
            iou_score = calculate_position_iou(char_pos, bbox)
            if iou_score > 0.1:  # IOU阈值
                candidate_bboxes.append((bbox, iou_score))
        
        if candidate_bboxes:
            # 选择IOU最高的bbox
            best_bbox = max(candidate_bboxes, key=lambda x: x[1])[0]
            strategy3_results.append(best_bbox)
    
    # 融合多个策略的结果
    final_bboxes = []
    all_strategies = [strategy1_results, strategy2_results, strategy3_results]
    
    for i, char_pos in enumerate(corrected_char_positions):
        bbox_votes = {}
        
        # 收集各策略的投票
        for strategy_results in all_strategies:
            if i < len(strategy_results):
                bbox = strategy_results[i]
                bbox_key = f"{bbox['start_x']},{bbox['start_y']},{bbox['end_x']},{bbox['end_y']}"
                bbox_votes[bbox_key] = bbox_votes.get(bbox_key, 0) + 1
        
        # 选择得票最多的bbox
        if bbox_votes:
            best_bbox_key = max(bbox_votes.keys(), key=lambda k: bbox_votes[k])
            # 找到对应的bbox对象
            for strategy_results in all_strategies:
                for bbox in strategy_results:
                    if f"{bbox['start_x']},{bbox['start_y']},{bbox['end_x']},{bbox['end_y']}" == best_bbox_key:
                        final_bboxes.append(bbox)
                        break
                else:
                    continue
                break
    
    return final_bboxes[:MAX_BBOX_LIMIT]  # 使用全局变量控制bbox数量

def calculate_position_match(char_pos, bbox):
    """计算字符位置与bbox的匹配度"""
    if 'x' in char_pos and 'y' in char_pos:
        char_x, char_y = char_pos['x'], char_pos['y']
        bbox_center_x = (bbox['start_x'] + bbox['end_x']) / 2
        bbox_center_y = (bbox['start_y'] + bbox['end_y']) / 2
        
        # 计算归一化距离匹配度
        bbox_width = bbox['end_x'] - bbox['start_x']
        bbox_height = bbox['end_y'] - bbox['start_y']
        
        x_match = 1.0 - min(abs(char_x - bbox_center_x) / max(bbox_width, 1), 1.0)
        y_match = 1.0 - min(abs(char_y - bbox_center_y) / max(bbox_height, 1), 1.0)
        
        return (x_match + y_match) / 2
    return 0.5  # 默认中等匹配度

def calculate_bbox_distance(char_pos, bbox):
    """计算字符位置到bbox中心的距离"""
    if 'x' in char_pos and 'y' in char_pos:
        char_x, char_y = char_pos['x'], char_pos['y']
        bbox_center_x = (bbox['start_x'] + bbox['end_x']) / 2
        bbox_center_y = (bbox['start_y'] + bbox['end_y']) / 2
        
        return np.sqrt((char_x - bbox_center_x)**2 + (char_y - bbox_center_y)**2)
    return float('inf')

def calculate_position_iou(char_pos, bbox):
    """计算字符位置与bbox的IOU（将字符位置扩展为小bbox）"""
    if 'x' in char_pos and 'y' in char_pos:
        # 将字符位置扩展为小的bbox（假设字符大小为10x10）
        char_size = 10
        char_bbox = {
            'start_x': char_pos['x'] - char_size/2,
            'start_y': char_pos['y'] - char_size/2,
            'end_x': char_pos['x'] + char_size/2,
            'end_y': char_pos['y'] + char_size/2
        }
        return calculate_bbox_iou(char_bbox, bbox)
    return 0.0

def process_corrected_file(corrected_file_path, bbox_file_path):
    """
    处理纠错文件和bbox文件，生成预测文本和bbox列表
    Args:
        corrected_file_path: 纠错结果文件路径
        bbox_file_path: bbox文件路径
    Returns:
        tuple: (predict_text, bounding_box_list)
    """
    try:
        # 读取纠错文件
        with open(corrected_file_path, 'r', encoding='utf-8') as f:
            corrected_data = json.load(f)
        
        # 读取bbox文件
        with open(bbox_file_path, 'r', encoding='utf-8') as f:
            bbox_data = json.load(f)
        
        # 提取纠错后的文本和变化信息
        predict_text = ""
        correction_chars = []  # 存储发生变化的字符信息
        
        if 'corrected_text_list' in corrected_data:
            corrected_sentences = []
            for item in corrected_data['corrected_text_list']:
                source_sentence = item.get('source_sentence', '')
                predict_sentence = item.get('predict_sentence', '')
                corrected_sentences.append(predict_sentence)
                
                # 找出发生变化的字符位置
                changes = find_text_changes(source_sentence, predict_sentence)
                correction_chars.extend(changes)
            
            predict_text = ' '.join(corrected_sentences)
        
        # 提取所有字符的bbox信息
        all_char_bboxes = []
        if 'sentences' in bbox_data:
            for sentence in bbox_data['sentences']:
                if 'chars' in sentence:
                    for char_info in sentence['chars']:
                        if 'bbox' in char_info and len(char_info['bbox']) >= 4:
                            bbox_info = {
                                'start_x': float(char_info['bbox'][0]),
                                'start_y': float(char_info['bbox'][1]),
                                'end_x': float(char_info['bbox'][2]),
                                'end_y': float(char_info['bbox'][3]),
                                'char': char_info.get('char', ''),
                                'sentence_id': sentence.get('sentence_id', 0),
                                'char_index': len(all_char_bboxes)
                            }
                            all_char_bboxes.append(bbox_info)
        
        # 选择错误相关的bbox
        selected_bboxes = []
        
        if correction_chars:
            # 基于纠错信息选择bbox
            selected_bboxes = select_correction_bboxes(correction_chars, all_char_bboxes)
        else:
            # 使用启发式方法选择可能有错误的bbox
            selected_bboxes = select_error_bboxes_enhanced(all_char_bboxes, predict_text)
        
        # 格式化bbox输出
        formatted_bboxes = []
        for bbox in selected_bboxes:
            formatted_bbox = {
                'start_x': bbox['start_x'],
                'start_y': bbox['start_y'],
                'end_x': bbox['end_x'],
                'end_y': bbox['end_y']
            }
            formatted_bboxes.append(formatted_bbox)
        
        return predict_text, formatted_bboxes
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return "", []

def find_text_changes(source_text, predict_text):
    """
    找出两个文本之间的变化位置
    Returns: 变化字符的位置信息列表
    """
    changes = []
    
    # 使用简单的字符对比找出不同之处
    min_len = min(len(source_text), len(predict_text))
    
    for i in range(min_len):
        if source_text[i] != predict_text[i]:
            changes.append({
                'position': i,
                'source_char': source_text[i],
                'predict_char': predict_text[i],
                'change_type': 'substitution'
            })
    
    # 处理长度不同的情况
    if len(source_text) != len(predict_text):
        if len(source_text) > len(predict_text):
            # 删除操作
            for i in range(min_len, len(source_text)):
                changes.append({
                    'position': i,
                    'source_char': source_text[i],
                    'predict_char': '',
                    'change_type': 'deletion'
                })
        else:
            # 插入操作
            for i in range(min_len, len(predict_text)):
                changes.append({
                    'position': i,
                    'source_char': '',
                    'predict_char': predict_text[i],
                    'change_type': 'insertion'
                })
    
    return changes

def select_correction_bboxes(correction_chars, all_char_bboxes):
    """
    根据纠错信息选择相关的bbox
    """
    selected_bboxes = []
    
    for change in correction_chars:
        position = change['position']
        
        # 找到对应位置的bbox
        if position < len(all_char_bboxes):
            target_bbox = all_char_bboxes[position]
            selected_bboxes.append(target_bbox)
            
            # 同时选择周围的字符bbox (增加上下文)
            for offset in [-1, 1]:
                neighbor_pos = position + offset
                if 0 <= neighbor_pos < len(all_char_bboxes):
                    neighbor_bbox = all_char_bboxes[neighbor_pos]
                    if neighbor_bbox not in selected_bboxes:
                        selected_bboxes.append(neighbor_bbox)
    
    # 使用全局变量控制bbox数量限制
    return selected_bboxes[:MAX_BBOX_LIMIT]

def select_error_bboxes_enhanced(all_char_bboxes, predict_text):
    """
    增强的启发式bbox选择方法
    """
    if not all_char_bboxes:
        return []
    
    selected_bboxes = []
    
    # 策略1: 基于字符类型选择
    error_prone_chars = {
        '的', '得', '地', '在', '再', '做', '作', '它', '他', '她', 
        '这', '哪', '那', '问', '文', '测', '测', '道', '到', '知', '只',
        '曾', '增', '长', '常', '出', '初', '未', '来', '观', '好'
    }
    
    for bbox in all_char_bboxes:
        char = bbox.get('char', '')
        if char in error_prone_chars:
            selected_bboxes.append(bbox)
    
    # 策略2: 基于位置分布选择（选择一些分散的位置）
    if len(all_char_bboxes) > 10:
        step = len(all_char_bboxes) // 8  # 选择8个分散的位置
        for i in range(0, len(all_char_bboxes), step):
            if all_char_bboxes[i] not in selected_bboxes:
                selected_bboxes.append(all_char_bboxes[i])
    
    # 策略3: 随机选择一些bbox
    import random
    random.seed(42)  # 固定随机种子
    remaining_bboxes = [bbox for bbox in all_char_bboxes if bbox not in selected_bboxes]
    if remaining_bboxes:
        random_count = min(3, len(remaining_bboxes))
        random_bboxes = random.sample(remaining_bboxes, random_count)
        selected_bboxes.extend(random_bboxes)
    
    # 限制返回数量并去重
    unique_bboxes = []
    seen_positions = set()
    
    for bbox in selected_bboxes:
        pos_key = f"{bbox['start_x']},{bbox['start_y']},{bbox['end_x']},{bbox['end_y']}"
        if pos_key not in seen_positions:
            seen_positions.add(pos_key)
            unique_bboxes.append(bbox)
    
    return unique_bboxes[:MAX_BBOX_LIMIT]  # 使用全局变量控制bbox数量

def select_error_bboxes_heuristic(bbox_candidates, predict_text):
    """
    启发式方法选择可能包含错误的bbox
    基于字符频率、位置分布等特征
    """
    if not bbox_candidates:
        return []
    
    selected_bboxes = []
    
    # 策略1: 选择置信度较低的bbox
    low_confidence_bboxes = [bbox for bbox in bbox_candidates if bbox.get('confidence', 1.0) < 0.7]
    
    # 策略2: 选择特殊字符或常见错误字符的bbox
    error_prone_chars = ['的', '得', '地', '在', '再', '做', '作', '它', '他', '她']
    error_char_bboxes = [bbox for bbox in bbox_candidates if bbox.get('char', '') in error_prone_chars]
    
    # 策略3: 随机选择一些bbox作为错误候选（模拟真实错误分布）
    import random
    random.seed(42)  # 固定随机种子保证可重现性
    random_bboxes = random.sample(bbox_candidates, min(len(bbox_candidates)//10, 5))
    
    # 合并策略结果，去重
    all_selected = low_confidence_bboxes + error_char_bboxes + random_bboxes
    seen_positions = set()
    
    for bbox in all_selected:
        pos_key = f"{bbox['start_x']},{bbox['start_y']},{bbox['end_x']},{bbox['end_y']}"
        if pos_key not in seen_positions:
            seen_positions.add(pos_key)
            selected_bboxes.append(bbox)
    
    return selected_bboxes[:MAX_BBOX_LIMIT]  # 使用全局变量控制bbox数量

def main():
    os.makedirs('./output', exist_ok=True)
    
    with open('data/test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    processed_count = 0
    total_bbox_count = 0
    
    print(f"开始处理 {len(test_data)} 个文件...")
    
    for i, item in enumerate(test_data):
        file_id = get_file_id(item['path'])
        corrected_file_path = f'data/paddleocr_version/ocr_corrected/{file_id}.json'
        bbox_file_path = f'data/paddleocr_version/bbox_washed/{file_id}.json'
        
        if os.path.exists(corrected_file_path) and os.path.exists(bbox_file_path):
            try:
                predict_text, bounding_box_list = process_corrected_file(corrected_file_path, bbox_file_path)
                item['predict_text'] = predict_text
                item['bounding_box_list'] = bounding_box_list
                
                # 如果source_text为空，从纠错文件中获取
                if not item.get('source_text'):
                    with open(corrected_file_path, 'r', encoding='utf-8') as f:
                        corrected_data = json.load(f)
                        if 'corrected_text_list' in corrected_data:
                            source_sentences = [item['source_sentence'] for item in corrected_data['corrected_text_list']]
                            item['source_text'] = ' '.join(source_sentences)
                
                processed_count += 1
                total_bbox_count += len(bounding_box_list)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(test_data):
                    print(f"进度: {i + 1}/{len(test_data)} - 文件: {file_id}, bbox数量: {len(bounding_box_list)}")
                    
            except Exception as e:
                print(f"处理文件 {file_id} 时出错: {str(e)}")
        else:
            print(f"跳过文件 {file_id}: 缺少必要文件")
    
    # 保存结果
    with open('./output/predict.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    with zipfile.ZipFile('./output/prediction.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write('./output/predict.json', arcname='predict.json')
        
    print(f"\n处理完成! 成功处理: {processed_count}/{len(test_data)}, 总bbox数: {total_bbox_count}")

if __name__ == '__main__':
    main() 