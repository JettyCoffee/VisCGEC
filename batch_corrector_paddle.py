import json
import os
import time
from datetime import datetime
from chinese_error_corrector import ChineseErrorCorrector

class BatchCorrector:
    def __init__(self):
        self.corrector = ChineseErrorCorrector()
        self.input_dir = "data/paddleocr_version/ocr_washed"
        self.output_dir = "data/paddleocr_version/ocr_corrected"
        self.time_log_file = "correction_time_log.json"
        self.time_records = {
            "start_time": "",
            "end_time": "",
            "total_duration": 0,
            "files": []
        }
        
    def process_single_file(self, input_file):
        """处理单个文件"""
        file_start_time = time.time()
        
        try:
            # 读取输入文件
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 准备新的数据结构
            corrected_data = {
                "path": data["path"],
                "corrected_text_list": []
            }
            
            # 处理每个句子
            for item in data["washed_text_list"]:
                source_sentence = item["sentence"]
                # 纠错处理
                corrected = self.corrector.correct(source_sentence)
                
                # 添加到结果列表
                corrected_data["corrected_text_list"].append({
                    "sentence_id": item["sentence_id"],
                    "source_sentence": source_sentence,
                    "predict_sentence": corrected
                })
                
            # 计算处理时间
            file_end_time = time.time()
            duration = file_end_time - file_start_time
            
            # 记录文件处理时间
            self.time_records["files"].append({
                "filename": os.path.basename(input_file),
                "start_time": datetime.fromtimestamp(file_start_time).strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.fromtimestamp(file_end_time).strftime('%Y-%m-%d %H:%M:%S'),
                "duration_seconds": duration
            })
            
            return corrected_data
            
        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {str(e)}")
            return None
            
    def process_all_files(self):
        """处理所有文件"""
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 记录开始时间
        self.time_records["start_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_start_time = time.time()
        count = 0

        # 处理每个文件
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.json'):
                print(f"正在处理文件: {filename}")
                input_file = os.path.join(self.input_dir, filename)
                output_file = os.path.join(self.output_dir, filename)
                
                # 处理文件
                result = self.process_single_file(input_file)
                
                if result:
                    # 保存处理结果
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    count += 1
                    print(f"已完成文件处理: {count} & {filename}")

                else:
                    print(f"文件处理失败: {filename}")
                    
        # 记录结束时间和总时长
        total_end_time = time.time()
        self.time_records["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.time_records["total_duration"] = total_end_time - total_start_time
        
        # 保存时间记录
        with open(self.time_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.time_records, f, ensure_ascii=False, indent=2)
            
        print(f"\n处理完成！")
        print(f"总处理时间: {self.time_records['total_duration']:.2f} 秒")
        print(f"详细时间记录已保存到: {self.time_log_file}")

def main():
    corrector = BatchCorrector()
    corrector.process_all_files()

if __name__ == "__main__":
    main() 