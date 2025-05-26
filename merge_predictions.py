import json
import os
import zipfile

def main():
    # 读取两个预测文件
    with open('./output/predict_got.json', 'r', encoding='utf-8') as f:
        got_data = json.load(f)
    
    with open('./output/predict_paddle.json', 'r', encoding='utf-8') as f:
        paddle_data = json.load(f)
    
    # 创建一个ID到数据的映射，方便查找
    got_map = {}
    for item in got_data:
        if 'fk_homework_id' in item:
            got_map[item['fk_homework_id']] = item
    
    replaced_count = 0
    
    # 合并结果
    for item in paddle_data:
        if 'fk_homework_id' not in item:
            continue
            
        fk_id = item['fk_homework_id']
        if fk_id in got_map:
            # 获取got对应的item
            got_item = got_map[fk_id]
            
            # 获取文本长度
            paddle_text_len = len(item.get('predict_text', ''))
            got_text_len = len(got_item.get('predict_text', ''))
            
            print(f"ID {fk_id}: paddle={paddle_text_len}, got={got_text_len}")
            
            # 如果paddle方法比got的少超过20个字，就使用got的预测文本
            if got_text_len > 0 and (paddle_text_len == 0 or got_text_len - paddle_text_len > 20):
                item['source_text'] = got_item.get('source_text', '')
                item['predict_text'] = got_item.get('predict_text', '')
                replaced_count += 1
                print(f"替换ID {fk_id}：paddle文本长度={paddle_text_len}，got文本长度={got_text_len}")
    
    print(f"合计替换了 {replaced_count} 项内容")
    
    # 保存结果
    with open('./output/predict.json', 'w', encoding='utf-8') as f:
        json.dump(paddle_data, f, ensure_ascii=False, indent=2)
    
    # 创建ZIP文件
    with zipfile.ZipFile('./output/prediction.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write('./output/predict.json', arcname='predict.json')
    
    print("合并完成，结果已保存到 ./output/predict.json 和 ./output/prediction.zip")

if __name__ == '__main__':
    main()
