# 中文OCR错误纠正系统 (VisCGEC)

本项目旨在对OCR识别后的中文文本进行自动纠错，提升文本质量。系统集成了深度学习模型，支持多种OCR方案，适用于学术研究和实际应用场景。

## 项目介绍

VisCGEC（Visual Chinese Grammatical Error Correction）系统采用流水线处理方式，包含图像预处理、OCR识别、字符解析、数据清洗、文本纠错和结果生成等阶段。系统可以自动检测并纠正OCR识别过程中产生的错误，提高文本的准确性和可读性。

## 流水线处理流程

1. 图像预处理：优化图像质量
2. OCR处理：识别图像中的文本（基于PaddleOCR或其他OCR引擎）
3. 字符解析：解析OCR输出的结构化数据
4. 数据清洗：过滤和规范化OCR输出
5. 文本纠错：应用深度学习模型进行语法和拼写纠错
6. 生成预测结果：整合纠错后的文本，生成最终输出

## 环境依赖

推荐使用Python 3.8+，Linux系统，建议GPU环境。

```bash
pip install -r requirements.txt
```

主要依赖项包括：
- transformers
- paddlepaddle (推荐GPU版本)
- paddleocr
- opencv-python
- numpy
- beautifulsoup4
- Pillow

## 预训练模型

**系统需要以下预训练模型：**

1. **ChineseErrorCorrector2-7B** (文本纠错模型)
   - 下载地址: https://huggingface.co/twnlp/ChineseErrorCorrector2-7B
   - 存放位置: `models/ChineseErrorCorrector2-7B/`

2. **chinese-roberta-wwm-ext** (分句与文本处理模型)
   - 下载地址: https://huggingface.co/hfl/chinese-roberta-wwm-ext
   - 存放位置: `models/chinese-roberta-wwm-ext/`

3. **PaddleOCR v2.10.0** (OCR识别模型)
   - 下载地址: https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.10
   - 安装方式: `pip install paddleocr==2.10.0`
   - 或克隆仓库: `git clone -b release/2.10 https://github.com/PaddlePaddle/PaddleOCR.git models/PaddleOCR`

## 快速开始

### 完整流水线处理

使用自动化脚本一键执行全流程：

```bash
bash pipeline.sh
```

### 单独执行各阶段

1. 图像预处理：
   ```bash
   python image_preproc.py
   ```

2. OCR处理：
   ```bash
   python ocr_processor.py
   ```

3. 字符解析：
   ```bash
   python ocr_char_parser.py
   ```

4. 数据清洗：
   ```bash
   python data_washer.py
   ```

5. 文本纠错：
   ```bash
   python batch_corrector.py
   ```

6. 生成预测结果：
   ```bash
   python generate_prediction.py
   ```

## 目录结构

- `pipeline.sh` - 自动化流水线处理脚本
- `image_preproc.py` - 图像预处理
- `ocr_processor.py` - PaddleOCR处理
- `ocr_char_parser.py` - OCR字符解析
- `data_washer.py` - 数据清洗
- `batch_corrector.py` - 批量文本纠错
- `chinese_error_corrector.py` - 纠错模型调用
- `generate_prediction.py` - 生成预测结果
- `models/` - 存放预训练模型
- `data/` - 存放数据及中间结果
- `output/` - 输出结果
- `evaluation_scores/` - 评测结果

## 数据说明

- `data/test_data.json` - 测试数据集
- `data/preprocessed_img/` - 预处理后的图像
- `data/paddleocr_version/` - PaddleOCR处理的中间结果
  - `ocr_output/` - OCR原始输出
  - `ocr_washed/` - 清洗后的OCR结果
  - `ocr_corrected/` - 纠错后的结果
- `output/` - 最终预测结果

## 项目特点

- **多阶段流水线**: 从图像到文本纠错的完整处理流程
- **高精度纠错**: 基于大型预训练模型的文本纠错
- **模块化设计**: 各阶段可独立执行和优化
- **多OCR引擎支持**: 支持PaddleOCR等多种OCR引擎
- **完整评测**: 内置评测工具，支持多种评测指标

## 注意事项

- 首次使用需下载所有预训练模型
- 推荐在GPU环境运行，尤其是大模型推理阶段
- 预处理图像质量会显著影响OCR和纠错效果
- 详细参数设置请查看各脚本的注释说明

## 许可证

请参阅 LICENSE 文件获取详细信息。