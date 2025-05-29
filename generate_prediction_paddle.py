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
    """找出从source到target的转换中所有修改的字符及其位置，并评估错误的典型性
    
    返回格式: [(pos, original_char, corrected_char, error_score, mod_type), ...]
    mod_type可以是: 'replace', 'delete', 'insert'
    """
    modified = []
    matcher = SequenceMatcher(None, source, target)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # 处理替换操作
            source_len = i2 - i1
            target_len = j2 - j1
            max_len = max(source_len, target_len)
            
            for k in range(max_len):
                source_idx = i1 + k
                target_idx = j1 + k
                
                if k < source_len and source_idx < len(source):
                    original_char = source[source_idx]
                    corrected_char = target[target_idx] if k < target_len and target_idx < len(target) else ''
                    error_score = calculate_error_typicality(original_char, corrected_char, source, source_idx)
                    modified.append((source_idx, original_char, corrected_char, error_score, 'replace'))
                    
        elif tag == 'delete':
            # 处理删除操作
            for pos in range(i1, i2):
                if pos < len(source):
                    original_char = source[pos]
                    error_score = calculate_error_typicality(original_char, '', source, pos)
                    modified.append((pos, original_char, '', error_score, 'delete'))
                    
        elif tag == 'insert':
            # 处理插入操作（源文本中的遗漏）
            inserted_text = target[j1:j2]
            # 确定上下文字符位置用于获取bbox
            context_pos = None
            context_char = ''
            
            if i1 > 0:  # 插入点不在开头，使用前一个字符作为上下文
                context_pos = i1 - 1
                context_char = source[context_pos] if context_pos < len(source) else ''
            elif i1 < len(source):  # 插入在开头，使用后一个字符作为上下文
                context_pos = i1
                context_char = source[context_pos]
            
            if context_pos is not None and context_char:
                error_score = calculate_omission_typicality(inserted_text, source, context_pos)
                # 对于插入，original_char是上下文字符，corrected_char标记插入的内容
                modified.append((context_pos, context_char, f"[+{inserted_text}]", error_score, 'insert'))
    
    # 按错误典型性得分排序，得分越高越典型
    modified.sort(key=lambda x: x[1], reverse=True)
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

def calculate_omission_typicality(inserted_text, context, pos):
    """计算遗漏错误的典型性得分，得分越高越可能是真正的遗漏错误"""
    score = 0
    
    # 1. 常见遗漏词汇 - 高分
    common_omissions = ['了', '的', '地', '得', '着', '呢', '吗', '吧', '过', '来', '去', '与', '和']
    if inserted_text in common_omissions:
        score += 85
    
    # 2. 助词和语气词遗漏 - 中高分
    auxiliary_words = ['就', '也', '还', '都', '才', '又', '再', '只', '已', '便']
    if inserted_text in auxiliary_words:
        score += 70
    
    # 3. 标点符号遗漏 - 中分
    if inserted_text in '，。！？；：""''（）【】':
        score += 50
    
    # 4. 单字符遗漏比多字符更常见 - 单字符加分
    if len(inserted_text) == 1:
        score += 40
    elif len(inserted_text) == 2:
        score += 20
    else:
        score += 10
    
    # 5. 位置权重：句子中间的遗漏比开头结尾更典型
    if 0.1 * len(context) < pos < 0.9 * len(context):
        score += 15
    
    # 6. 常见词组遗漏
    common_phrases = ['一些', '一个', '一下', '可以', '应该', '能够', '需要', '必须']
    if inserted_text in common_phrases:
        score += 60
        
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

def find_char_bbox_robust(sentence_chars, char_pos, target_char, sentence_id):
    """更鲁棒的字符bbox查找，包含回退策略
    
    Args:
        sentence_chars: bbox数据中的字符列表
        char_pos: 目标字符在源文本中的位置
        target_char: 要查找的字符
        sentence_id: 句子ID（用于调试）
    
    Returns:
        char_info dict 或 None
    """
    if not sentence_chars or char_pos >= len(sentence_chars):
        return None
    
    # 策略1: 直接位置匹配
    if char_pos < len(sentence_chars):
        char_info = sentence_chars[char_pos]
        if char_info['char'] == target_char:
            return char_info
    
    # 策略2: 在小范围窗口内搜索
    search_window = 3  # 搜索窗口大小
    for offset in range(-search_window, search_window + 1):
        if offset == 0:
            continue  # 已经在策略1中检查过
        check_idx = char_pos + offset
        if 0 <= check_idx < len(sentence_chars):
            char_info = sentence_chars[check_idx]
            if char_info['char'] == target_char:
                print(f"Warning: Found char '{target_char}' at offset {offset} from expected pos {char_pos} in sentence {sentence_id}")
                return char_info
    
    # 策略3: 在整个字符列表中搜索（作为最后的回退）
    for idx, char_info in enumerate(sentence_chars):
        if char_info['char'] == target_char and abs(idx - char_pos) <= 5:  # 限制在合理范围内
            print(f"Warning: Found char '{target_char}' at distant pos {idx} instead of {char_pos} in sentence {sentence_id}")
            return char_info
    
    print(f"Error: Could not find bbox for char '{target_char}' at pos {char_pos} in sentence {sentence_id}")
    return None

def process_corrected_file(file_path, bbox_path):
    """处理单个纠错文件，合并所有预测句子，并收集修改字符的bbox信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(bbox_path, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)
        
    sentences = []
    all_error_bboxes_with_scores = []  # 收集所有句子的错误bbox
    
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
                sentence_chars = None
                for sentence in bbox_data['sentences']:
                    if sentence['sentence_id'] == sentence_id:
                        sentence_chars = sentence['chars']
                        break
                
                if sentence_chars:
                    # 对于每个修改的字符，找到其bbox
                    for char_pos, original_char, corrected_char, error_score, mod_type in modified_chars:
                        # 使用鲁棒的bbox查找方法
                        char_info = find_char_bbox_robust(sentence_chars, char_pos, original_char, sentence_id)
                        
                        if char_info:
                            bbox = get_char_bbox(char_info)
                            # 添加错误得分和其他信息
                            bbox['error_score'] = error_score
                            bbox['original_char'] = original_char
                            bbox['corrected_char'] = corrected_char
                            bbox['mod_type'] = mod_type
                            bbox['sentence_id'] = sentence_id
                            all_error_bboxes_with_scores.append(bbox)
                        else:
                            print(f"Skipping bbox for char '{original_char}' at pos {char_pos} in sentence {sentence_id} - bbox not found")
    
    # 全局排序所有识别到的错误，然后选择前3个最重要的
    all_error_bboxes_with_scores.sort(key=lambda x: x.get('error_score', 0), reverse=True)
    
    # 应用非最大抑制类似的逻辑，避免重复高亮相同区域
    filtered_bboxes = apply_bbox_nms(all_error_bboxes_with_scores, iou_threshold=0.7)
    
    # 清理辅助信息，只保留bbox坐标，取前3个
    top_bboxes = []
    for bbox in filtered_bboxes[:1]:
        clean_bbox = {
            "start_x": bbox["start_x"],
            "end_x": bbox["end_x"],
            "start_y": bbox["start_y"],
            "end_y": bbox["end_y"]
        }
        top_bboxes.append(clean_bbox)
    
    return ' '.join(sentences), top_bboxes

def calculate_bbox_iou(bbox1, bbox2):
    """计算两个bbox的IoU"""
    # 计算交集区域
    x1 = max(bbox1["start_x"], bbox2["start_x"])
    y1 = max(bbox1["start_y"], bbox2["start_y"])
    x2 = min(bbox1["end_x"], bbox2["end_x"])
    y2 = min(bbox1["end_y"], bbox2["end_y"])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集区域
    area1 = (bbox1["end_x"] - bbox1["start_x"]) * (bbox1["end_y"] - bbox1["start_y"])
    area2 = (bbox2["end_x"] - bbox2["start_x"]) * (bbox2["end_y"] - bbox2["start_y"])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

def apply_bbox_nms(bboxes_with_scores, iou_threshold=0.7):
    """应用类似非最大抑制的逻辑，过滤重叠的bbox"""
    if not bboxes_with_scores:
        return []
    
    # 按分数排序（已经排序过了，但确保一下）
    sorted_bboxes = sorted(bboxes_with_scores, key=lambda x: x.get('error_score', 0), reverse=True)
    
    filtered = []
    for current_bbox in sorted_bboxes:
        should_keep = True
        for kept_bbox in filtered:
            iou = calculate_bbox_iou(current_bbox, kept_bbox)
            if iou > iou_threshold:
                # 如果IoU超过阈值，比较分数
                if current_bbox.get('error_score', 0) <= kept_bbox.get('error_score', 0):
                    should_keep = False
                    break
                else:
                    # 当前bbox分数更高，移除之前的bbox
                    filtered.remove(kept_bbox)
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