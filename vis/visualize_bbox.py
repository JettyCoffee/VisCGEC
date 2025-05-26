import json
from PIL import Image, ImageDraw, ImageFont
import os

def visualize_bbox(json_path, img_path, output_path):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图片ID
    img_id = data['path']
    
    # 读取图片
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    # 设置字体（使用默认字体）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 遍历所有句子和字符
    for sentence in data['sentences']:
        for char_data in sentence['chars']:
            # 获取边界框坐标
            bbox = char_data['bbox']
            x1, y1, x2, y2 = bbox
            
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
    json_path = "data/paddleocr_version/bbox_washed/2365.json"
    img_path = "data/test_img_data/2365.jpg"
    output_path = "data/vis_img"
    
    # 执行可视化
    visualize_bbox(json_path, img_path, output_path) 