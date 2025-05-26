import json
import os
from transformers import AutoTokenizer
import re

class TextWasher:
    def __init__(self):
        # 使用本地的chinese-roberta-wwm-ext模型
        self.tokenizer = AutoTokenizer.from_pretrained("models/chinese-roberta-wwm-ext")
        # 设置句子最大长度
        self.max_sentence_length = 50
        
    def clean_text(self, text):
        """
        清理文本开头的非正文内容
        """
        if not text:
            return text
            
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
        
        # 应用所有清理模式
        cleaned_text = text
        for pattern, replacement in patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
            
        # 如果清理后是空的，返回原文本
        if not cleaned_text and text:
            return text
            
        return cleaned_text

    def semantic_split(self, text):
        """
        基于语义进行句子切分
        使用tokenizer的分词结果和常见语义分割词来辅助切分
        """
        if len(text) <= self.max_sentence_length:
            return [text]

        # 常见的语义分割词
        semantic_markers = ['但是', '因为', '所以', '而且', '不过', '然后', '接着', '并且', '如果', '虽然', '尽管', '否则', '要是']
        
        # 使用tokenizer进行分词
        tokens = self.tokenizer.tokenize(text)
        
        # 初始化结果和当前片段
        result = []
        current_part = ''
        current_length = 0
        
        for token in tokens:
            # 去除tokenizer添加的特殊标记
            clean_token = token.replace('##', '')
            token_length = len(clean_token)
            
            # 如果当前片段加上新token超过最大长度
            if current_length + token_length > self.max_sentence_length:
                if current_part:
                    result.append(current_part.strip())
                current_part = clean_token
                current_length = token_length
            else:
                # 检查是否遇到语义分割词
                next_part = current_part + clean_token
                should_split = False
                
                for marker in semantic_markers:
                    if next_part.endswith(marker):
                        result.append(next_part.strip())
                        current_part = ''
                        current_length = 0
                        should_split = True
                        break
                
                if not should_split:
                    current_part = next_part
                    current_length += token_length
        
        # 添加最后一个片段
        if current_part:
            result.append(current_part.strip())
        
        # 合并过短的片段
        final_result = []
        current = ''
        
        for part in result:
            if len(current) == 0:
                current = part
            elif len(current) + len(part) <= self.max_sentence_length:
                current += part
            else:
                final_result.append(current)
                current = part
        
        if current:
            final_result.append(current)
        
        return final_result
        
    def split_by_comma(self, text):
        """
        使用逗号分割过长的句子
        """
        if len(text) <= self.max_sentence_length:
            return [text]
            
        # 将中英文逗号统一
        text = text.replace(',', '，')
        parts = text.split('，')
        
        # 合并过短的片段
        result = []
        current_part = ''
        
        for part in parts:
            if not part.strip():
                continue
                
            if not current_part:
                current_part = part
            elif len(current_part) + len(part) + 1 <= self.max_sentence_length:
                current_part += '，' + part
            else:
                result.append(current_part.strip())
                current_part = part
        
        if current_part:
            result.append(current_part.strip())
            
        # 如果还有超长的片段，使用语义切分
        final_result = []
        for part in result:
            if len(part) > self.max_sentence_length:
                final_result.extend(self.semantic_split(part))
            else:
                final_result.append(part)
            
        return final_result
        
    def split_sentences(self, text):
        """
        将文本分成句子
        使用多种标点符号作为分隔符：。！？；
        对于过长的句子，使用逗号进行二次分割
        对于没有标点的长句子，使用语义分割
        """
        if not text or not isinstance(text, str):
            return []
            
        # 预处理：清理文本
        text = self.clean_text(text)
            
        # 预处理：统一全角半角符号，移除多余空格
        text = text.replace('．', '。').strip()
        text = text.replace('!', '！').replace('?', '？').replace(';', '；')
        
        # 使用正则表达式分句
        pattern = r'([。！？；])+(?![0-9])'
        sentences = re.split(pattern, text)
        
        # 清理分句结果
        cleaned_sentences = []
        current_sentence = ""
        
        for s in sentences:
            if s in ['。', '！', '？', '；']:
                current_sentence += s
                if current_sentence.strip():
                    # 对长句子进行处理
                    sub_sentences = self.split_by_comma(current_sentence.strip())
                    cleaned_sentences.extend(sub_sentences)
                current_sentence = ""
            else:
                current_sentence += s
                
        # 处理最后一个可能没有标点的句子
        if current_sentence.strip():
            # 如果最后一个句子没有标点且过长，使用语义切分
            if len(current_sentence) > self.max_sentence_length and not any(p in current_sentence for p in '。！？；，'):
                sub_sentences = self.semantic_split(current_sentence.strip())
            else:
                sub_sentences = self.split_by_comma(current_sentence.strip())
            cleaned_sentences.extend(sub_sentences)
            
        return [sent for sent in cleaned_sentences if sent.strip()]

    def process_file(self, input_file):
        """处理单个JSON文件"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取source_text
            source_text = data.get('source_text', '')
            path = data.get('path', '')
            
            # 分句处理
            sentences = self.split_sentences(source_text)
            
            # 构建输出格式
            washed_text_list = [
                {
                    "sentence_id": idx,
                    "sentence": sent
                } for idx, sent in enumerate(sentences)
            ]
            
            return {
                "path": path,
                "washed_text_list": washed_text_list
            }
            
        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {str(e)}")
            return None

def main():
    # 创建输出目录
    output_dir = "data/got_version/ocr_washed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化清洗器
    washer = TextWasher()
    
    # 处理输入目录中的所有JSON文件
    input_dir = "data/got_version/ocr_output"
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            
            # 处理文件
            result = washer.process_file(input_file)
            
            if result:
                # 保存处理结果
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"已处理并保存文件: {filename}")
            else:
                print(f"处理文件失败: {filename}")

if __name__ == "__main__":
    main() 