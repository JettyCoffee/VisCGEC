# 中文错误纠正系统 (VisCGEC)

本项目旨在对OCR识别后的中文文本进行自动纠错，提升文本质量。系统集成了多种深度学习模型，支持多种OCR方案，适用于学术研究和实际应用。

## 目录结构

- `chinese_error_corrector.py`         主纠错模型推理脚本
- `ocr_processor_paddle.py`/`ocr_processor_got.py`  OCR输出处理（PaddleOCR/GOT-OCR）
- `batch_corrector_paddle.py`/`batch_corrector_got.py`  批量纠错入口
- `data_washer_paddle.py`/`data_washer_got.py`      数据清洗工具
- `image_preprocessor.py`              图像预处理
- `evaluation.py`                      评测脚本
- `generate_prediction.py`/`generate_prediction_paddle.py`  生成预测结果
- `merge_predictions.py`               合并多模型预测
- `test.py`/`test_single_image.py`     测试脚本
- `visualize_bbox.py`/`visualize_bbox_predict.py`   可视化工具
- `models/`                            存放各类模型目录（见下文）
- `data/`                              训练、测试及中间数据
- `output/`                            预测输出与结果

## 环境依赖

推荐使用Python 3.8+，Linux系统，建议GPU环境。

```bash
pip install -r requirements.txt
```

## 预训练模型下载与目录说明

**注意：所有模型文件需手动下载，目录结构如下：**

```
models/
  ChineseErrorCorrector2-7B/        # 主纠错大模型（需下载）
  chinese-roberta-wwm-ext/          # RoBERTa中文预训练模型（需下载）
  GOT-OCR2_0/                       # GOT-OCR模型（需下载）
  PaddleOCR/                        # PaddleOCR模型（需下载）
```

### 下载方式

- ChineseErrorCorrector2-7B
  - 下载链接: [模型下载地址]
  - 解压到 `models/ChineseErrorCorrector2-7B/`
- chinese-roberta-wwm-ext
  - 下载链接: [模型下载地址]
  - 解压到 `models/chinese-roberta-wwm-ext/`
- GOT-OCR2_0
  - 下载链接: [模型下载地址]
  - 解压到 `models/GOT-OCR2_0/`
- PaddleOCR
  - 下载链接: [模型下载地址]
  - 或通过 `git clone https://github.com/PaddlePaddle/PaddleOCR.git models/PaddleOCR` 获取
- PaddleOCR完整版
  - 仓库中仅包含精简版核心代码
  - 完整版可通过 `git clone https://github.com/PaddlePaddle/PaddleOCR.git models/PaddleOCR` 获取

> 若目录为空，仅保留`.gitkeep`文件用于结构占位。

## 快速开始

1. 数据清洗（以PaddleOCR为例）：
   ```bash
   python data_washer_paddle.py
   ```
2. 执行批量纠错：
   ```bash
   python batch_corrector_paddle.py
   ```
3. 评测结果：
   ```bash
   python evaluation.py
   ```

## 数据说明

- `data/` 目录下包含原始图片、OCR输出、纠错结果等。
- 结构示例：
  - `test_data.json`/`train_data.json`：主数据集
  - `paddleocr_version/`、`got_version/`：不同OCR方案下的中间结果
  - `preprocessed_img/`：预处理图片

## 主要模型说明

- **ChineseErrorCorrector2-7B**：基于LLaMA-2的中文纠错大模型
- **chinese-roberta-wwm-ext**：RoBERTa中文预训练模型，特征提取与初步检测
- **GOT-OCR2_0**：GOT-OCR识别模型
- **PaddleOCR**：PaddleOCR识别模型

## 注意事项

- 所有大模型需单独下载，见上文链接
- 仓库中PaddleOCR仅包含精简版核心代码，不包含完整模型
- 推荐GPU环境，CPU推理速度较慢
- `models/`目录下仅保留结构，内容需用户自行下载
- 详细用法请参考各脚本内注释

## GitHub同步说明

本项目使用Git LFS（Large File Storage）管理大文件。但由于模型文件体积巨大，我们在`.gitignore`中排除了所有模型文件，仅保留目录结构。首次克隆后请按照上述指南下载所需模型。

```bash
# 克隆仓库
git clone https://github.com/your-username/VisCGEC.git
# 进入目录
cd VisCGEC
# 安装依赖
pip install -r requirements.txt
# 下载必要模型
# ...按照上述指南下载模型文件...
```

## 许可证

请参阅 LICENSE 文件获取详细信息。