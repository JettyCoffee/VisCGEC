import os
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
from pathlib import Path
import logging
import math

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OCRProcessor:
    def __init__(
        self,
        input_dir,
        output_dir,
        det_model_dir="models/PaddleOCR/ppstructure/inference/PP-OCRv5_server_det_infer",
        rec_model_dir="models/PaddleOCR/ppstructure/inference/PP-OCRv5_server_rec_infer",
        rec_char_dict_path="models/PaddleOCR/ppocr/utils/ppocrv5_dict.txt",
        table_model_dir="models/PaddleOCR/ppstructure/inference/ch_ppstructure_mobile_v2.0_SLANet_infer",
        table_char_dict_path="models/PaddleOCR/ppocr/utils/dict/table_structure_dict_ch.txt",
        layout_model_dir="models/PaddleOCR/ppstructure/inference/picodet_lcnet_x1_0_fgd_layout_cdla_infer",
        layout_dict_path="models/PaddleOCR/ppocr/utils/dict/layout_dict/layout_cdla_dict.txt",
        vis_font_path="models/PaddleOCR/doc/fonts/chinese_cht.ttf",
        max_workers_per_gpu=4,
        gpu_ids=[0, 1]  # 指定要使用的GPU ID
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.rec_char_dict_path = rec_char_dict_path
        self.table_model_dir = table_model_dir
        self.table_char_dict_path = table_char_dict_path
        self.layout_model_dir = layout_model_dir
        self.layout_dict_path = layout_dict_path
        self.vis_font_path = vis_font_path
        self.max_workers_per_gpu = max_workers_per_gpu
        self.gpu_ids = gpu_ids

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_single_image(self, args):
        """处理单张图片"""
        image_path, gpu_id = args
        try:
            output_dir = self.output_dir / image_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            # 设置环境变量指定GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            cmd = [
                "python", "models/PaddleOCR/ppstructure/predict_system_enhanced.py",
                f"--image_dir={str(image_path)}",
                f"--det_model_dir={self.det_model_dir}",
                f"--rec_model_dir={self.rec_model_dir}",
                f"--rec_char_dict_path={self.rec_char_dict_path}",
                f"--table_model_dir={self.table_model_dir}",
                f"--table_char_dict_path={self.table_char_dict_path}",
                f"--layout_model_dir={self.layout_model_dir}",
                f"--layout_dict_path={self.layout_dict_path}",
                f"--vis_font_path={self.vis_font_path}",
                f"--output={str(output_dir)}",
                "--return_word_box=True",
                "--ocr=True",
                "--table=False",
                "--layout=True",
                "--use_angle_cls=False",
                "--use_mp=True",
            ]

            logging.info(f"Processing image: {image_path} on GPU {gpu_id}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                logging.info(f"Successfully processed {image_path} on GPU {gpu_id}")
            else:
                logging.error(f"Error processing {image_path} on GPU {gpu_id}: {result.stderr}")

        except Exception as e:
            logging.error(f"Error processing {image_path} on GPU {gpu_id}: {str(e)}")

    def process_all_images(self):
        """多GPU并行处理所有图片"""
        image_files = [
            f for f in self.input_dir.glob("*")
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.JPG'}
        ]

        total_images = len(image_files)
        logging.info(f"Found {total_images} images to process")

        # 为每个图片分配GPU
        tasks = []
        for i, image_file in enumerate(image_files):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            tasks.append((image_file, gpu_id))

        # 按GPU分组任务
        gpu_tasks = {}
        for gpu_id in self.gpu_ids:
            gpu_tasks[gpu_id] = [task for task in tasks if task[1] == gpu_id]

        # 创建进程池，每个GPU一个进程
        with ProcessPoolExecutor(max_workers=len(self.gpu_ids)) as process_executor:
            # 提交每个GPU的任务批次
            futures = []
            for gpu_id, gpu_batch in gpu_tasks.items():
                future = process_executor.submit(self._process_gpu_batch, gpu_batch)
                futures.append(future)

            # 等待所有任务完成
            for future in futures:
                future.result()

        logging.info("所有图片处理完成")

    def _process_gpu_batch(self, gpu_tasks):
        """处理单个GPU上的任务批次"""
        with ThreadPoolExecutor(max_workers=self.max_workers_per_gpu) as thread_executor:
            list(thread_executor.map(self.process_single_image, gpu_tasks))

if __name__ == "__main__":
    # 设置输入输出目录
    input_dir = "data/preprocessed_img"
    output_dir = "data/paddleocr_version/ocr_output"

    # 创建OCR处理器实例
    processor = OCRProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers_per_gpu=6,
        gpu_ids=[0, 1]
    )

    # 处理所有图片
    processor.process_all_images()
