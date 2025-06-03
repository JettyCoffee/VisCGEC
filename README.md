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
   
   **重要**：本项目使用了经过修改的PaddleOCR模型配置以提升OCR性能，需要下载以下特定模型并进行相应配置：
   
   ```bash
   # 进入PaddleOCR目录
   cd models/PaddleOCR/ppstructure
   
   # 创建推理目录
   mkdir -p inference && cd inference
   
   # 下载PP-OCRv5服务器端检测模型
   wget https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/ch_PP-OCRv5_det_server_infer.tar && tar xf ch_PP-OCRv5_det_server_infer.tar
   # 重命名模型目录
   mv ch_PP-OCRv5_det_server_infer PP-OCRv5_server_det_infer
   
   # 下载PP-OCRv5服务器端识别模型
   wget https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/ch_PP-OCRv5_rec_server_infer.tar && tar xf ch_PP-OCRv5_rec_server_infer.tar
   # 重命名模型目录
   mv ch_PP-OCRv5_rec_server_infer PP-OCRv5_server_rec_infer
   
   # 下载中文表格结构模型
   wget https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar && tar xf ch_ppstructure_mobile_v2.0_SLANet_infer.tar
   
   # 下载版面分析模型
   wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar && tar xf picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar
   
   # 返回上级目录
   cd ../..
   
   # 确保字典文件存在于正确位置
   mkdir -p ppocr/utils
   cp PaddleOCR_changedfiles/ppocrv5_dict.txt ppocr/utils/
   
   # 复制修改后的预测系统文件
   cp PaddleOCR_changedfiles/predict_system_enhanced.py ppstructure/
   ```
   
   注意：
   - 必须确保词典文件`ppocrv5_dict.txt`位于`models/PaddleOCR/ppocr/utils/`目录下
   - 必须确保修改后的`predict_system_enhanced.py`文件位于`models/PaddleOCR/ppstructure/`目录下

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

MIT License

Copyright (c) 2025 Jetty Coffee