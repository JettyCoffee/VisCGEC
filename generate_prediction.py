import json
import os
import zipfile
import re
from difflib import SequenceMatcher

def get_file_id(path):
    """从文件路径中提取ID"""
    base_name = os.path.basename(path)
    return re.sub(r'\.[^.]*$', '', base_name)

def find_modified_chars(source, target):
    """找出从source到target的转换中所有修改的字符及其位置"""
    modified = []
    matcher = SequenceMatcher(None, source, target)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            source_len = i2 - i1
            target_len = j2 - j1
            max_len = max(source_len, target_len)
            
            for k in range(max_len):
                source_idx = i1 + k
                target_idx = j1 + k
                
                if k < source_len and source_idx < len(source):
                    original_char = source[source_idx]
                    corrected_char = target[target_idx] if k < target_len and target_idx < len(target) else ''
                    error_score = calculate_error_score(original_char, corrected_char, source, source_idx)
                    modified.append((source_idx, original_char, corrected_char, error_score, 'replace'))
                    
        elif tag == 'delete':
            for pos in range(i1, i2):
                if pos < len(source):
                    original_char = source[pos]
                    error_score = calculate_error_score(original_char, '', source, pos)
                    modified.append((pos, original_char, '', error_score, 'delete'))
                    
        elif tag == 'insert':
            inserted_text = target[j1:j2]
            context_pos = i1 - 1 if i1 > 0 else (i1 if i1 < len(source) else None)
            
            if context_pos is not None and context_pos < len(source):
                context_char = source[context_pos]
                error_score = calculate_error_score('', inserted_text, source, context_pos, is_insert=True)
                modified.append((context_pos, context_char, f"[+{inserted_text}]", error_score, 'insert'))
    
    return modified

def calculate_error_score(original_char, corrected_char, context, pos, is_insert=False):
    """重新设计的错误评分系统，更注重实际错误检测"""
    score = 50  # 基础分数，所有检测到的修改都有一定价值
    
    # 处理插入操作
    if is_insert:
        # 常见遗漏词汇的重要性更平衡
        common_omissions = ['了', '的', '地', '得', '着', '呢', '吗', '吧', '过', '来', '去', '与', '和']
        if corrected_char in common_omissions:
            score += 30  # 降低权重，避免过度集中
        
        # 标点符号遗漏
        if corrected_char in '，。！？；：""''（）【】':
            score += 25
        
        # 字符长度影响
        if len(corrected_char) == 1:
            score += 20
        else:
            score += 15
            
        return score
    
    # 对所有类型的错误给予更平衡的评分
    # 形似字错误
    similar_pairs = {
        '已': '己', '己': '已', '日': '目', '目': '日', '天': '夭', '夭': '天',
        '人': '入', '入': '人', '木': '未', '未': '木', '土': '士', '士': '土',
        '刀': '力', '力': '刀', '几': '儿', '儿': '几', '马': '鸟', '鸟': '马',
        '千': '干', '干': '千', '王': '主', '主': '王', '由': '田', '田': '由'
    }
    if original_char in similar_pairs and corrected_char == similar_pairs[original_char]:
        score += 40  # 降低权重
    
    # 音近字错误
    phonetic_pairs = {
        '的': '得', '得': '的', '在': '再', '再': '在', '做': '作', '作': '做',
        '像': '象', '象': '像', '以': '已', '已': '以', '那': '哪', '哪': '那'
    }
    if original_char in phonetic_pairs and corrected_char == phonetic_pairs[original_char]:
        score += 35
    
    # 繁简字错误
    traditional_simplified = {
        '実': '实', '発': '发', '図': '图', '対': '对',
        '説': '说', '経': '经', '學': '学', '師': '师'
    }
    if original_char in traditional_simplified and corrected_char == traditional_simplified[original_char]:
        score += 30
    
    # 标点符号错误
    if original_char in '，。！？；：' and corrected_char in '，。！？；：':
        score += 15
    
    # 数字字母错误
    if (original_char.isdigit() and corrected_char.isdigit()) or \
       (original_char.isalpha() and corrected_char.isalpha()):
        score += 20
    
    # 位置权重 - 减少影响
    if 0.2 * len(context) < pos < 0.8 * len(context):
        score += 5
    
    # 添加随机性因子，避免相同得分的bbox被完全过滤
    import random
    score += random.uniform(-2, 2)
    
    return score

def get_char_bbox(char_info):
    """从字符信息中提取bbox"""
    bbox = char_info['bbox']
    return {
        "start_x": bbox[0], "end_x": bbox[2],
        "start_y": bbox[1], "end_y": bbox[3]
    }

def find_char_bbox(sentence_chars, char_pos, target_char, sentence_id):
    """鲁棒的字符bbox查找"""
    if not sentence_chars or char_pos >= len(sentence_chars):
        return None
    
    # 直接位置匹配
    if char_pos < len(sentence_chars) and sentence_chars[char_pos]['char'] == target_char:
        return sentence_chars[char_pos]
    
    # 小范围窗口搜索
    for offset in range(-3, 4):
        check_idx = char_pos + offset
        if 0 <= check_idx < len(sentence_chars) and sentence_chars[check_idx]['char'] == target_char:
            return sentence_chars[check_idx]
    
    return None

def process_corrected_file(file_path, bbox_path):
    """处理单个纠错文件，使用改进的bbox选择策略"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(bbox_path, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)
        
    sentences = []
    all_error_bboxes = []  # 先收集所有bbox
    
    if 'corrected_text_list' in data:
        for item in data['corrected_text_list']:
            source = item['source_sentence']
            target = item['predict_sentence']
            sentences.append(target)
            
            # 找出修改的字符
            modified_chars = find_modified_chars(source, target)
            
            # 在bbox数据中找到对应句子
            sentence_id = item.get('sentence_id')
            if sentence_id is not None:
                sentence_chars = None
                for sentence in bbox_data['sentences']:
                    if sentence['sentence_id'] == sentence_id:
                        sentence_chars = sentence['chars']
                        break
                
                if sentence_chars:
                    # 为每个修改的字符找bbox
                    for char_pos, original_char, corrected_char, error_score, mod_type in modified_chars:
                        char_info = find_char_bbox(sentence_chars, char_pos, original_char, sentence_id)
                        
                        if char_info:
                            bbox = get_char_bbox(char_info)
                            bbox['error_score'] = error_score
                            bbox['original_char'] = original_char
                            bbox['corrected_char'] = corrected_char
                            bbox['mod_type'] = mod_type
                            bbox['sentence_id'] = sentence_id
                            bbox['char_pos'] = char_pos  # 添加位置信息用于多样性选择
                            all_error_bboxes.append(bbox)
    
    # 使用增强的bbox选择策略
    selected_bboxes = enhanced_bbox_selection(all_error_bboxes)
    
    # 清理辅助信息
    top_bboxes = [{
        "start_x": bbox["start_x"], "end_x": bbox["end_x"],
        "start_y": bbox["start_y"], "end_y": bbox["end_y"]
    } for bbox in selected_bboxes]
    
    return ' '.join(sentences), top_bboxes

def calculate_bbox_iou(bbox1, bbox2):
    """计算两个bbox的IoU"""
    x1 = max(bbox1["start_x"], bbox2["start_x"])
    y1 = max(bbox1["start_y"], bbox2["start_y"])
    x2 = min(bbox1["end_x"], bbox2["end_x"])
    y2 = min(bbox1["end_y"], bbox2["end_y"])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1["end_x"] - bbox1["start_x"]) * (bbox1["end_y"] - bbox1["start_y"])
    area2 = (bbox2["end_x"] - bbox2["start_x"]) * (bbox2["end_y"] - bbox2["start_y"])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_position_diversity_score(bbox, selected_bboxes):
    """计算位置多样性得分，鼓励选择分布更广的bbox"""
    if not selected_bboxes:
        return 1.0
    
    min_distance = float('inf')
    bbox_center_x = (bbox['start_x'] + bbox['end_x']) / 2
    bbox_center_y = (bbox['start_y'] + bbox['end_y']) / 2
    
    for selected in selected_bboxes:
        selected_center_x = (selected['start_x'] + selected['end_x']) / 2
        selected_center_y = (selected['start_y'] + selected['end_y']) / 2
        
        distance = ((bbox_center_x - selected_center_x) ** 2 + 
                   (bbox_center_y - selected_center_y) ** 2) ** 0.5
        min_distance = min(min_distance, distance)
    
    # 归一化距离得分
    return min(1.0, min_distance / 100.0)  # 假设100像素为合理距离单位

def enhanced_bbox_selection(all_error_bboxes, max_count=16):
    """增强的bbox选择策略"""
    if not all_error_bboxes:
        return []
    
    # 第一步：质量过滤，去除明显的低质量检测
    scores = [bbox.get('error_score', 0) for bbox in all_error_bboxes]
    if scores:
        mean_score = sum(scores) / len(scores)
        min_threshold = max(40, mean_score * 0.6)  # 动态阈值
        
        # 过滤低分bbox
        quality_filtered = [bbox for bbox in all_error_bboxes 
                          if bbox.get('error_score', 0) >= min_threshold]
    else:
        quality_filtered = all_error_bboxes
    
    if not quality_filtered:
        quality_filtered = all_error_bboxes[:max_count]  # 回退策略
    
    # 第二步：多维度选择
    selected = []
    candidates = quality_filtered.copy()
    
    while len(selected) < max_count and candidates:
        best_candidate = None
        best_score = -1
        
        for candidate in candidates:
            # 综合评分：错误得分 + 位置多样性 + 类型平衡
            error_score = candidate.get('error_score', 0)
            
            # 位置多样性得分
            diversity_score = calculate_position_diversity_score(candidate, selected)
            
            # 类型平衡得分
            mod_type = candidate.get('mod_type', 'unknown')
            type_count = sum(1 for s in selected if s.get('mod_type') == mod_type)
            type_balance_score = 1.0 / (1.0 + type_count * 0.5)  # 减少同类型的偏好
            
            # 综合得分
            combined_score = (error_score * 0.5 + 
                            diversity_score * 30 + 
                            type_balance_score * 20)
            
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        else:
            break
    
    return selected

def multi_strategy_bbox_selection(all_error_bboxes, max_count=16):
    """多策略bbox选择，平衡质量与多样性"""
    if not all_error_bboxes:
        return []
    
    # 策略1: 按错误类型分组选择
    type_groups = {}
    for bbox in all_error_bboxes:
        mod_type = bbox.get('mod_type', 'unknown')
        if mod_type not in type_groups:
            type_groups[mod_type] = []
        type_groups[mod_type].append(bbox)
    
    # 为每种类型排序
    for mod_type in type_groups:
        type_groups[mod_type].sort(key=lambda x: x.get('error_score', 0), reverse=True)
    
    selected = []
    
    # 策略2: 轮询选择，确保每种错误类型都有代表
    max_per_type = max(1, max_count // len(type_groups)) if type_groups else max_count
    
    for mod_type, bboxes in type_groups.items():
        type_selected = 0
        for bbox in bboxes:
            if type_selected >= max_per_type:
                break
                
            # 检查与已选择的bbox是否重叠过多
            should_add = True
            for selected_bbox in selected:
                if calculate_bbox_iou(bbox, selected_bbox) > 0.8:
                    should_add = False
                    break
            
            if should_add:
                selected.append(bbox)
                type_selected += 1
                
            if len(selected) >= max_count:
                break
        
        if len(selected) >= max_count:
            break
    
    # 策略3: 如果还有剩余槽位，按分数填充
    if len(selected) < max_count:
        remaining_bboxes = [bbox for bbox in all_error_bboxes if bbox not in selected]
        remaining_bboxes.sort(key=lambda x: x.get('error_score', 0), reverse=True)
        
        for bbox in remaining_bboxes:
            if len(selected) >= max_count:
                break
                
            should_add = True
            for selected_bbox in selected:
                if calculate_bbox_iou(bbox, selected_bbox) > 0.6:
                    should_add = False
                    break
            
            if should_add:
                selected.append(bbox)
    
    # 最终按分数排序
    selected.sort(key=lambda x: x.get('error_score', 0), reverse=True)
    return selected[:max_count]

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