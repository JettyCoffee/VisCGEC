#!/bin/bash

echo "=== 开始执行自动化处理流程 ==="

echo "=== 开始 OCR 处理 ==="
echo "执行 OCR 处理..."
# python image_preprocessor.py
echo "图像预处理完成"
python ocr_processor_paddle.py
echo "OCR 处理完成"
python ocr_char_parser.py
echo "OCR 字符解析完成"

echo "=== 开始数据清洗 ==="
echo "执行数据清洗..."
python data_washer_paddle.py
echo "数据清洗完成"

echo "=== 开始文本纠错 ==="
echo "执行批量文本纠错..."
python batch_corrector_paddle.py
echo "文本纠错完成" 

echo "=== 开始生成预测结果 ==="
echo "执行生成预测结果..."
python generate_prediction_paddle.py
echo "预测结果生成完成"

echo "=== 处理流程完成 ==="

