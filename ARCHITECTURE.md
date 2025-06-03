# VisCGEC 系统架构文档

## 1. 系统概述

VisCGEC（Visual Chinese Grammatical Error Correction）是一个端到端的中文OCR文本纠错系统。系统旨在解决OCR识别过程中产生的文本错误问题，通过多阶段处理流程，实现从图像输入到纠错输出的全流程自动化。

## 2. 系统架构

### 2.1 整体架构

系统采用流水线（Pipeline）架构，分为以下主要模块：

```
图像预处理 → OCR处理 → 字符解析 → 数据清洗 → 文本纠错 → 结果生成
```

每个模块可以独立运行，也可以通过自动化脚本（pipeline.sh）串联执行。

### 2.2 核心模块说明

#### 2.2.1 图像预处理模块

**核心文件**: `image_preproc.py`

**功能**:
- 图像缩放和规范化
- 色彩调整和对比度优化
- 噪点消除和图像增强

**输入**: 原始图像文件
**输出**: 预处理后的图像（存放于 `data/preprocessed_img/`）

#### 2.2.2 OCR处理模块

**核心文件**: `ocr_processor.py`

**功能**:
- 调用PaddleOCR引擎进行文本识别
- 提取文本位置信息和识别结果
- 生成结构化OCR输出数据

**模型配置**:
- 使用PP-OCRv5服务器端模型提升识别精度
- 自定义配置的模型路径:
  ```python
  det_model_dir="models/PaddleOCR/ppstructure/inference/PP-OCRv5_server_det_infer"
  rec_model_dir="models/PaddleOCR/ppstructure/inference/PP-OCRv5_server_rec_infer"
  rec_char_dict_path="models/PaddleOCR/ppocr/utils/ppocrv5_dict.txt"
  table_model_dir="models/PaddleOCR/ppstructure/inference/ch_ppstructure_mobile_v2.0_SLANet_infer"
  ```
- 修改的PaddleOCR预测系统:
  - 使用自定义的`predict_system_enhanced.py`
  - 该文件位于`models/PaddleOCR/ppstructure/`目录
  - 主要修改：将表格区域统一识别为图像区域，避免表格结构识别错误

**性能优化**:
- 多GPU并行处理
- 线程池优化批处理性能
- 自定义词典扩展中文识别能力

**输入**: 预处理后的图像
**输出**: OCR识别结果（存放于 `data/paddleocr_version/ocr_output/`）

#### 2.2.3 字符解析模块

**核心文件**: `ocr_char_parser.py`

**功能**:
- 解析OCR输出的JSON/XML格式数据
- 提取文本内容、位置和置信度信息
- 生成标准化的字符级数据

**输入**: OCR原始输出
**输出**: 解析后的字符级数据

#### 2.2.4 数据清洗模块

**核心文件**: `data_washer.py`

**功能**:
- 过滤低置信度文本
- 合并分散的文本片段
- 规范化标点符号和特殊字符
- 初步纠正明显错误

**输入**: 解析后的OCR数据
**输出**: 清洗后的文本数据（存放于 `data/paddleocr_version/ocr_washed/`）

#### 2.2.5 文本纠错模块

**核心文件**: 
- `batch_corrector.py` (批量处理入口)
- `chinese_error_corrector.py` (纠错模型调用)

**功能**:
- 调用ChineseErrorCorrector2-7B大模型进行文本纠错
- 检测并修正语法、拼写和上下文错误
- 优化文本的语义连贯性

**使用模型**: 
- ChineseErrorCorrector2-7B (基于LLaMA-2的中文纠错大模型)
- chinese-roberta-wwm-ext (用于文本分句和特征提取)

**输入**: 清洗后的文本数据
**输出**: 纠错后的文本（存放于 `data/paddleocr_version/ocr_corrected/`）

#### 2.2.6 结果生成模块

**核心文件**: `generate_prediction_paddle.py`

**功能**:
- 整合纠错结果
- 生成最终输出文件
- 可选生成可视化结果

**输入**: 纠错后的文本
**输出**: 最终预测结果（存放于 `output/`）

### 2.3 数据流向

```
原始图像 → 预处理图像 → OCR结果 → 解析后数据 → 清洗数据 → 纠错文本 → 最终结果
```

## 3. 技术栈

### 3.1 核心技术

- **OCR引擎**: PaddleOCR v2.10.0
- **纠错模型**: ChineseErrorCorrector2-7B (基于LLaMA-2)
- **特征提取**: chinese-roberta-wwm-ext
- **图像处理**: OpenCV, PIL
- **深度学习框架**: PyTorch, PaddlePaddle

### 3.2 依赖项

详细依赖项见 `requirements.txt` 文件，主要包括：
- transformers
- paddlepaddle
- paddleocr
- opencv-python
- numpy
- beautifulsoup4
- Pillow

## 4. 部署架构

### 4.1 推荐硬件配置

- **CPU**: 8核及以上
- **内存**: 16GB及以上
- **GPU**: NVIDIA GPU，显存20GB及以上（推荐用于大模型推理）
- **存储**: 50GB及以上（预训练模型较大）

### 4.2 部署流程

1. 环境准备
   - 安装Python 3.8+
   - 安装CUDA和cuDNN（GPU版本）

2. 依赖安装
   ```bash
   pip install -r requirements.txt
   ```

3. 模型下载
   - 下载ChineseErrorCorrector2-7B模型
   - 下载chinese-roberta-wwm-ext模型
   - 安装PaddleOCR

4. 配置验证
   ```bash
   python test_1_image_ocr.py
   ```

5. 运行系统
   ```bash
   bash pipeline.sh
   ```

## 5. 系统性能

### 5.1 性能指标

- **处理速度**: 平均每张图像处理时间（取决于硬件配置）
  - CPU模式: 10-30秒/张
  - GPU模式: 1-5秒/张
  
- **纠错准确率**: 取决于OCR识别质量和文本复杂度
  - 简单文本: 90%以上
  - 复杂文本: 75-85%

### 5.2 资源消耗

- **GPU内存**: 7-8GB (ChineseErrorCorrector2-7B模型推理)
- **CPU内存**: 4-8GB
- **存储空间**: 
  - 模型: 约15-20GB
  - 运行数据: 根据处理数据量动态变化

## 6. 扩展性设计

### 6.1 支持的扩展点

- **OCR引擎**: 可替换为其他OCR引擎（见GOT版本示例）
- **纠错模型**: 可替换为其他文本纠错模型
- **预处理算法**: 可定制图像预处理流程
- **数据格式**: 支持多种输入/输出格式

### 6.2 自定义配置

系统的大部分参数可通过配置文件或命令行参数进行自定义，包括：
- OCR置信度阈值
- 纠错模型参数
- 数据清洗规则
- 输出格式设置

## 7. 维护与更新

### 7.1 日志系统

系统在各处理阶段会生成详细日志，便于问题诊断和性能优化。

### 7.2 更新路线

- 支持更多OCR引擎
- 优化纠错模型性能
- 增加更多语言支持
- 提升处理速度和资源利用率

## 8. 最佳实践

### 8.1 优化建议

- 使用GPU加速模型推理
- 对大量图像进行批处理
- 根据具体应用场景调整置信度阈值
- 定期更新模型以获取更好的性能

### 8.2 常见问题解决

- OCR质量不佳：调整图像预处理参数，提高图像质量
- 纠错不准确：检查清洗后的文本质量，必要时调整模型参数
- 系统内存不足：减小批处理大小，或增加系统内存
- 处理速度慢：使用GPU加速，或优化处理流程
