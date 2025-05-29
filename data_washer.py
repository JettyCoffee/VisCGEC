import json
import os
from transformers import AutoTokenizer
import re

class PaddleTextWasher:
    def __init__(self):
        # 使用本地的chinese-roberta-wwm-ext模型
        self.tokenizer = AutoTokenizer.from_pretrained("models/chinese-roberta-wwm-ext")
        # 设置句子最大长度
        self.max_sentence_length = 50
        
    def clean_text_with_bbox(self, text, chars):
        """
        清理文本开头的非正文内容，同时保持bbox信息的一致性
        """
        if not text or not chars:
            return text, chars
            
        # 清理模式列表
        patterns = [
            # 清理所有空格（包括中英文之间的空格）
            (r'\s+', ''),
            # 清理学号（8-12位数字）及其前面的文字
            (r'[^\d]*?\d{8,12}(?=\D|$)', ''),
            # 清理"学号"及其后面的数字
            (r'学号[：:\-~]*\d+', ''),
            # 清理题号（一、二、三...）
            (r'^[一二三四五六七八九十]+、', ''),
            # 清理数字题号
            (r'^\d+[、\.]', ''),
            # 清理写作题型说明
            (r'^写作[\(\（]\d+分[\)\）]', ''),
            # 清理题目说明
            (r'题目[：:\-~]+\d*', ''),
            # 清理姓名
            (r'姓名[：:\-~]+\w+', ''),
            # 清理班级
            (r'班级[：:\-~]+\w+', ''),
            # 清理常见标记符号
            (r'^[\-~\d]+', ''),
            # 清理括号内容
            (r'[\(\（].*?[\)\）]', ''),
            # 清理学校名称+学号组合
            (r'^[^\s\d]*?\d{8,}', ''),
            # 清理带书名号的标题
            (r'《[^》]*》', ''),
            # 清理学校简称（如：朴社）
            (r'^[一-龥]{1,4}社', ''),
            # 清理常见学校缩写+数字组合
            (r'^[一-龥]{1,4}[校社院系所]\d+', ''),
            # 清理标题形式（带书名号和破折号的组合）
            (r'^.*?《.*?》.*?[—\-]+', ''),
            # 清理开头的非中文字符和数字组合
            (r'^[^\u4e00-\u9fa5]+', ''),
        ]
        
        # 记录需要删除的字符位置
        to_delete = set()
        cleaned_text = text
        
        for pattern, replacement in patterns:
            matches = list(re.finditer(pattern, cleaned_text))
            for match in matches:
                start, end = match.span()
                to_delete.update(range(start, end))
        
        # 过滤掉需要删除的字符和对应的bbox
        filtered_chars = []
        filtered_text = ""
        current_pos = 0
        
        for char_info in chars:
            if current_pos not in to_delete:
                filtered_chars.append(char_info)
                filtered_text += char_info["char"]
            current_pos += 1
            
        return filtered_text, filtered_chars

    def semantic_split(self, text, chars):
        """
        基于语义进行句子切分，同时处理bbox信息
        """
        if len(text) <= self.max_sentence_length:
            return [(text, chars)]

        # 常见的语义分割词
        semantic_markers = ['但是', '因为', '所以', '而且', '不过', '然后', '接着', '并且', '如果', '虽然', '尽管', '否则', '要是']
        
        # 使用tokenizer进行分词
        tokens = self.tokenizer.tokenize(text)
        
        # 初始化结果和当前片段
        result = []
        current_text = ''
        current_chars = []
        current_length = 0
        char_index = 0
        
        for token in tokens:
            # 去除tokenizer添加的特殊标记
            clean_token = token.replace('##', '')
            token_length = len(clean_token)
            
            # 获取当前token对应的字符和bbox
            token_chars = chars[char_index:char_index + token_length]
            char_index += token_length
            
            # 如果当前片段加上新token超过最大长度
            if current_length + token_length > self.max_sentence_length:
                if current_text:
                    result.append((current_text.strip(), current_chars))
                current_text = clean_token
                current_chars = token_chars
                current_length = token_length
            else:
                # 检查是否遇到语义分割词
                next_text = current_text + clean_token
                should_split = False
                
                for marker in semantic_markers:
                    if next_text.endswith(marker):
                        result.append((next_text.strip(), current_chars + token_chars))
                        current_text = ''
                        current_chars = []
                        current_length = 0
                        should_split = True
                        break
                
                if not should_split:
                    current_text = next_text
                    current_chars.extend(token_chars)
                    current_length += token_length
        
        # 添加最后一个片段
        if current_text:
            result.append((current_text.strip(), current_chars))
        
        return result

    def split_by_comma(self, text, chars):
        """
        使用逗号分割过长的句子，同时处理bbox信息
        """
        if len(text) <= self.max_sentence_length:
            return [(text, chars)]
            
        # 找到所有逗号的位置
        comma_positions = []
        for i, char in enumerate(text):
            if char in '，,':
                comma_positions.append(i)
                
        if not comma_positions:
            return self.semantic_split(text, chars)
            
        # 根据逗号位置分割
        result = []
        start = 0
        current_text = ''
        current_chars = []
        
        for pos in comma_positions:
            if len(current_text) + (pos - start + 1) <= self.max_sentence_length:
                current_text += text[start:pos + 1]
                current_chars.extend(chars[start:pos + 1])
            else:
                if current_text:
                    result.append((current_text.strip(), current_chars))
                current_text = text[start:pos + 1]
                current_chars = chars[start:pos + 1]
            start = pos + 1
            
        # 处理最后一段
        if start < len(text):
            if len(current_text) + (len(text) - start) <= self.max_sentence_length:
                current_text += text[start:]
                current_chars.extend(chars[start:])
                result.append((current_text.strip(), current_chars))
            else:
                if current_text:
                    result.append((current_text.strip(), current_chars))
                result.extend(self.semantic_split(text[start:], chars[start:]))
                
        return result

    def split_sentences(self, text, chars):
        """
        将文本分成句子，同时保持bbox信息的对应关系
        """
        if not text or not isinstance(text, str):
            return []
            
        # 预处理：清理文本和对应的bbox
        text, chars = self.clean_text_with_bbox(text, chars)
            
        # 预处理：统一全角半角符号
        punctuation_map = {
            '．': '。',
            '!': '！',
            '?': '？',
            ';': '；'
        }
        
        # 找到所有句子结束符的位置
        end_positions = []
        for i, char in enumerate(text):
            if char in '。！？；':
                end_positions.append(i)
                
        if not end_positions:
            return self.split_by_comma(text, chars)
            
        # 根据句子结束符分割
        result = []
        start = 0
        
        for pos in end_positions:
            sentence = text[start:pos + 1]
            sentence_chars = chars[start:pos + 1]
            
            # 对长句子进行进一步处理
            if len(sentence) > self.max_sentence_length:
                sub_sentences = self.split_by_comma(sentence, sentence_chars)
                result.extend(sub_sentences)
            else:
                result.append((sentence, sentence_chars))
            start = pos + 1
            
        # 处理最后一段
        if start < len(text):
            last_sentence = text[start:]
            last_chars = chars[start:]
            if len(last_sentence) > self.max_sentence_length:
                sub_sentences = self.split_by_comma(last_sentence, last_chars)
                result.extend(sub_sentences)
            else:
                result.append((last_sentence, last_chars))
                
        return [(sent.strip(), sent_chars) for sent, sent_chars in result if sent.strip()]

    def process_file(self, input_file):
        """处理单个JSON文件"""
        try:
            # 从文件名中提取图片ID（去掉_results.json后缀）
            img_id = os.path.basename(input_file).replace('_results.json', '')
            img_path = f"{img_id}"  # 默认使用jpg格式
            
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取所有文本和对应的bbox信息
            all_text = ""
            all_chars = []
            
            # 遍历所有结果
            for result in data.get('results', []):
                # 获取文本和字符框信息
                text = result.get('source_text', '')
                chars = result.get('char_boxes', [])
                
                if text and chars:
                    # 清理文本和bbox
                    cleaned_text, cleaned_chars = self.clean_text_with_bbox(text, chars)
                    if cleaned_text and cleaned_chars:
                        all_text += cleaned_text
                        all_chars.extend(cleaned_chars)
            
            # 分句处理
            sentences = self.split_sentences(all_text, all_chars)
            
            # 构建bbox_washed输出（包含分句信息）
            bbox_washed = {
                "path": img_path,
                "text": all_text,
                "sentences": []
            }
            
            # 为每个句子添加其对应的字符和bbox信息
            for idx, (sentence_text, sentence_chars) in enumerate(sentences):
                sentence_info = {
                    "sentence_id": idx,
                    "sentence": sentence_text,
                    "chars": sentence_chars
                }
                bbox_washed["sentences"].append(sentence_info)
            
            # 构建ocr_washed输出
            washed_text_list = [
                {
                    "sentence_id": idx,
                    "sentence": sent
                } for idx, (sent, _) in enumerate(sentences)
            ]
            
            ocr_washed = {
                "path": img_path,
                "washed_text_list": washed_text_list
            }
            
            return img_id, bbox_washed, ocr_washed
            
        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {str(e)}")
            return None, None, None

def main():
    # 创建输出目录
    bbox_washed_dir = "data/paddleocr_version/bbox_washed"
    ocr_washed_dir = "data/paddleocr_version/ocr_washed"
    os.makedirs(bbox_washed_dir, exist_ok=True)
    os.makedirs(ocr_washed_dir, exist_ok=True)
    
    # 初始化清洗器
    washer = PaddleTextWasher()
    
    # 处理输入目录中的所有JSON文件
    input_dir = "data/paddleocr_version/ocr_summary"
    
    for filename in os.listdir(input_dir):
        if filename.endswith('_results.json'):
            input_file = os.path.join(input_dir, filename)
            
            # 处理文件
            img_id, bbox_result, ocr_result = washer.process_file(input_file)
            
            if img_id and bbox_result and ocr_result:
                # 使用图片ID作为输出文件名
                bbox_output_file = os.path.join(bbox_washed_dir, f"{img_id}.json")
                ocr_output_file = os.path.join(ocr_washed_dir, f"{img_id}.json")
                
                # 保存bbox_washed结果
                with open(bbox_output_file, 'w', encoding='utf-8') as f:
                    json.dump(bbox_result, f, ensure_ascii=False, indent=2)
                    
                # 保存ocr_washed结果
                with open(ocr_output_file, 'w', encoding='utf-8') as f:
                    json.dump(ocr_result, f, ensure_ascii=False, indent=2)
                    
                print(f"已处理并保存文件: {img_id}")
            else:
                print(f"处理文件失败: {filename}")

if __name__ == "__main__":
    main() 