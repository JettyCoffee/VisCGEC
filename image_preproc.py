from PIL import Image
import os
from pathlib import Path

def preprocess_image(image_path):
    # 打开图片文件
    img = Image.open(image_path)
    
    # 确保图像是RGB模式
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 转换为黑白图像
    bw_img = img.convert('L')
    
    return bw_img

def process_directory(input_dir, output_dir):
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.JPG')
    
    # 遍历输入目录中的所有文件
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                # 构建完整的输入路径
                input_path = os.path.join(root, file)
                
                # 构建相对路径，保持目录结构
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # 确保输出文件的目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    # 处理图片
                    processed_img = preprocess_image(input_path)
                    
                    # 保存处理后的图片
                    processed_img.save(output_path)
                    print(f"已处理: {rel_path}")
                except Exception as e:
                    print(f"处理 {rel_path} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = "data/test_img_data"
    output_dir = "data/preprocessed_img"
    
    # 处理图片
    process_directory(input_dir, output_dir)
    print("图像预处理完成！") 