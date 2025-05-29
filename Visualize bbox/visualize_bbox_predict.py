import json
from PIL import Image, ImageDraw, ImageFont
import os

def visualize_bbox(json_path, img_dir, output_path):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 遍历所有预测结果
    for item in data:
        img_id = str(item['fk_homework_id'])  # 获取图片ID
        img_path = os.path.join(img_dir, f"{img_id}.jpg")  # 尝试jpg格式
        
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{img_id}.png")  # 尝试png格式
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, f"{img_id}.JPG")  # 尝试JPG格式
                if not os.path.exists(img_path):
                    print(f"找不到图片: {img_id}")
                    continue
        
        # 读取图片
        img = Image.open(img_path)
        # 如果是RGBA格式，转换为RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # 设置字体（使用默认字体）
        try:
            font = ImageFont.truetype("/font/HarmonyOS_Sans_SC_Medium.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 遍历所有边界框
        if 'bounding_box_list' in item:
            for bbox in item['bounding_box_list']:
                # 获取边界框坐标
                x1 = bbox['start_x']
                y1 = bbox['start_y']
                x2 = bbox['end_x']
                y2 = bbox['end_y']
                
                # 绘制矩形边界框
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 保存图片
        output_file = os.path.join(output_path, f"{img_id}_bbox.jpg")
        img.save(output_file)
        print(f"已保存可视化结果到: {output_file}")

if __name__ == "__main__":
    # 设置路径
    json_path = "../output/predict.json"
    img_dir = "../data/test_img_data"
    output_path = "../output/vis_img_predict"
    
    # 执行可视化
    visualize_bbox(json_path, img_dir, output_path) 