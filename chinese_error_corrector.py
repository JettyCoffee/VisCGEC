import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ChineseErrorCorrector:
    def __init__(self, model_path="models/ChineseErrorCorrector2-7B"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.prompt = "你是一个文本纠错专家，纠正输入句子中的语法、拼写、标点错误，并输出语义通顺的句子，输入句子为："

    def correct(self, text):
        messages = [
            {"role": "user", "content": self.prompt + text}
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

def main():
    # 初始化纠错器
    corrector = ChineseErrorCorrector()
    
    # 测试用例
    test_sentences = [
        "妆到的在服比我想的过要好，从那以后，程开始经常在网上笑乐西。",
    ]
    
    print("开始文本纠错测试：")
    print("-" * 50)
    
    for sentence in test_sentences:
        print(f"原始文本：{sentence}")
        corrected = corrector.correct(sentence)
        print(f"纠正后的文本：{corrected}")
        print("-" * 50)

if __name__ == "__main__":
    main() 