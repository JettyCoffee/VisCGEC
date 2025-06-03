#!/bin/bash

echo "=== 开始执行自动化处理流程 ==="

echo "=== 开始 OCR 处理阶段 ==="
python image_preproc.py
echo "图像预处理完成"
python ocr_processor.py
echo "OCR 处理完成"
python ocr_char_parser.py
echo "OCR 字符解析完成"

echo "=== 开始数据清洗 ==="
python data_washer.py
echo "数据清洗完成"

echo "=== 开始文本纠错 ==="
python batch_corrector.py
echo "文本纠错完成" 

echo "=== 开始生成预测结果 ==="
python generate_prediction.py
echo "预测结果生成完成"

echo "=== 处理流程完成 ==="

