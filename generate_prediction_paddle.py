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
    """统一计算错误的典型性得分"""
    score = 0
    
    # 处理插入操作
    if is_insert:
        common_omissions = ['了', '的', '地', '得', '着', '呢', '吗', '吧', '过', '来', '去', '与', '和', 
                           '就', '也', '还', '都', '才', '又', '再', '只', '已', '便']
        if corrected_char in common_omissions:
            score += 85
        
        if corrected_char in '，。！？；：""''（）【】':
            score += 50
        
        if len(corrected_char) == 1:
            score += 40
        elif len(corrected_char) == 2:
            score += 20
        else:
            score += 10
            
        return score
    
    # 形似字错误（OCR常见错误）
    similar_pairs = {
        '已': '己', '己': '已', '日': '目', '目': '日', '天': '夭', '夭': '天',
        '人': '入', '入': '人', '木': '未', '未': '木', '土': '士', '士': '土',
        '刀': '力', '力': '刀', '几': '儿', '儿': '几', '马': '鸟', '鸟': '马',
        '千': '干', '干': '千', '王': '主', '主': '王', '由': '田', '田': '由'
    }
    if original_char in similar_pairs and corrected_char == similar_pairs[original_char]:
        score += 100
    
    # 音近字错误
    phonetic_pairs = {
        '的': '得', '得': '的', '在': '再', '再': '在', '做': '作', '作': '做',
        '像': '象', '象': '像', '以': '已', '已': '以', '那': '哪', '哪': '那'
    }
    if original_char in phonetic_pairs and corrected_char == phonetic_pairs[original_char]:
        score += 80
    
    # 繁简字错误
    traditional_simplified = {
        '実': '实', '発': '发', '図': '图', '対': '对',
        '説': '说', '経': '经', '學': '学', '師': '师'
    }
    if original_char in traditional_simplified and corrected_char == traditional_simplified[original_char]:
        score += 60
    
    # 标点符号错误
    if original_char in '，。！？；：' and corrected_char in '，。！？；：':
        score += 20
    
    # 数字字母错误
    if (original_char.isdigit() and corrected_char.isdigit()) or \
       (original_char.isalpha() and corrected_char.isalpha()):
        score += 30
    
    # 位置权重
    if 0.2 * len(context) < pos < 0.8 * len(context):
        score += 10
    
    # 常见错字
    common_errors = '己日目天人入木未土士刀力几儿马鸟千干王主由田'
    if original_char in common_errors:
        score += 15
    
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
    """处理单个纠错文件，计算所有bbox后统一截断"""
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
                            all_error_bboxes.append(bbox)
    
    # 全局排序并应用NMS
    all_error_bboxes.sort(key=lambda x: x.get('error_score', 0), reverse=True)
    filtered_bboxes = apply_bbox_nms(all_error_bboxes, iou_threshold=0.7)
    
    # 截断到前8个并清理辅助信息
    top_bboxes = [{
        "start_x": bbox["start_x"], "end_x": bbox["end_x"],
        "start_y": bbox["start_y"], "end_y": bbox["end_y"]
    } for bbox in filtered_bboxes[:8]]
    
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

def apply_bbox_nms(bboxes_with_scores, iou_threshold=0.7):
    """应用NMS过滤重叠的bbox"""
    if not bboxes_with_scores:
        return []
    
    filtered = []
    for current_bbox in bboxes_with_scores:
        should_keep = True
        for kept_bbox in filtered:
            if calculate_bbox_iou(current_bbox, kept_bbox) > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            filtered.append(current_bbox)
    
    return filtered

def main():
    # 确保输出目录存在
    os.makedirs('./output', exist_ok=True)
    
    # 读取test_data.json
    with open('data/test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    processed_count = 0
    total_bbox_count = 0
    
    print(f"开始处理 {len(test_data)} 个文件...")
    
    # 处理每个文件
    for i, item in enumerate(test_data):
        # 从path中提取ID
        file_id = get_file_id(item['path'])
        corrected_file_path = os.path.join('data/paddleocr_version/ocr_corrected', file_id + '.json')
        bbox_file_path = os.path.join('data/paddleocr_version/bbox_washed', file_id + '.json')
        
        if os.path.exists(corrected_file_path) and os.path.exists(bbox_file_path):
            try:
                # 处理文件并获取预测文本和bbox列表
                predict_text, bounding_box_list = process_corrected_file(corrected_file_path, bbox_file_path)
                item['predict_text'] = predict_text
                item['bounding_box_list'] = bounding_box_list
                
                # 如果source_text为空，从纠错文件中获取原始文本
                if not item.get('source_text') and os.path.exists(corrected_file_path):
                    with open(corrected_file_path, 'r', encoding='utf-8') as f:
                        corrected_data = json.load(f)
                        if 'corrected_text_list' in corrected_data:
                            source_sentences = [item['source_sentence'] for item in corrected_data['corrected_text_list']]
                            item['source_text'] = ' '.join(source_sentences)
                
                processed_count += 1
                total_bbox_count += len(item['bounding_box_list'])
                
                if (i + 1) % 10 == 0 or (i + 1) == len(test_data):
                    print(f"进度: {i + 1}/{len(test_data)} - 处理文件: {file_id}, 文本长度: {len(item['predict_text'])}, bbox数量: {len(item['bounding_box_list'])}")
                    
            except Exception as e:
                print(f"处理文件 {file_id} 时出错: {str(e)}")
                continue
        else:
            missing_files = []
            if not os.path.exists(corrected_file_path):
                missing_files.append("纠错文件")
            if not os.path.exists(bbox_file_path):
                missing_files.append("bbox文件")
            print(f"跳过文件 {file_id}: 缺少{', '.join(missing_files)}")
    
    # 保存结果
    with open('./output/predict.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 创建ZIP文件
    with zipfile.ZipFile('./output/prediction.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write('./output/predict.json', arcname='predict.json')
        
    print(f"\n处理完成!")
    print(f"- 成功处理文件数: {processed_count}/{len(test_data)}")
    print(f"- 总bbox数量: {total_bbox_count}")
    print(f"- 平均每文件bbox数: {total_bbox_count/max(processed_count, 1):.2f}")

if __name__ == '__main__':
    main() 