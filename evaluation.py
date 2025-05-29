# 文件路径可自行修改
import json
import numpy as np

# 增强的IOU计算
def compute_iou(box1, box2):
    """基础IOU计算函数"""
    x_left = max(box1["start_x"], box2["start_x"])
    y_top = max(box1["start_y"], box2["start_y"])
    x_right = min(box1["end_x"], box2["end_x"])
    y_bottom = min(box1["end_y"], box2["end_y"])

    #如果没有重叠
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1["end_x"] - box1["start_x"]) * (box1["end_y"] - box1["start_y"])
    area2 = (box2["end_x"] - box2["start_x"]) * (box2["end_y"] - box2["start_y"])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def enhanced_compute_iou(box1, box2, penalty_factor=0.1):
    """
    增强的IOU计算，考虑形状相似性和大小差异
    Args:
        box1, box2: bbox字典
        penalty_factor: 形状差异惩罚因子
    Returns:
        float: 增强的IOU值
    """
    # 基础IOU计算
    base_iou = compute_iou(box1, box2)
    
    if base_iou == 0.0:
        return 0.0
    
    # 计算形状相似性
    w1 = box1["end_x"] - box1["start_x"]
    h1 = box1["end_y"] - box1["start_y"]
    w2 = box2["end_x"] - box2["start_x"]
    h2 = box2["end_y"] - box2["start_y"]
    
    # 避免除零错误
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return base_iou
    
    # 宽高比相似性
    aspect_ratio1 = w1 / h1
    aspect_ratio2 = w2 / h2
    aspect_similarity = min(aspect_ratio1, aspect_ratio2) / max(aspect_ratio1, aspect_ratio2)
    
    # 大小相似性
    area1 = w1 * h1
    area2 = w2 * h2
    size_similarity = min(area1, area2) / max(area1, area2)
    
    # 形状惩罚
    shape_penalty = penalty_factor * (2.0 - aspect_similarity - size_similarity)
    
    # 增强的IOU
    enhanced_iou = base_iou * (1.0 - shape_penalty)
    
    return max(0.0, enhanced_iou)

def compute_bbox_matching_score(pred_boxes, gt_boxes, matching_strategy="hungarian"):
    """
    计算bbox匹配得分，使用更精确的匹配策略
    Args:
        pred_boxes: 预测的bbox列表
        gt_boxes: 真实的bbox列表
        matching_strategy: 匹配策略 ("hungarian", "greedy", "hybrid")
    Returns:
        float: 匹配得分
    """
    if not pred_boxes or not gt_boxes:
        return 0.0
    
    if matching_strategy == "hungarian":
        return hungarian_bbox_matching(pred_boxes, gt_boxes)
    elif matching_strategy == "greedy":
        return greedy_bbox_matching(pred_boxes, gt_boxes)
    elif matching_strategy == "hybrid":
        # 混合策略：小规模用hungarian，大规模用greedy
        if len(pred_boxes) * len(gt_boxes) <= 100:
            return hungarian_bbox_matching(pred_boxes, gt_boxes)
        else:
            return greedy_bbox_matching(pred_boxes, gt_boxes)
    else:
        return greedy_bbox_matching(pred_boxes, gt_boxes)

def hungarian_bbox_matching(pred_boxes, gt_boxes):
    """使用匈牙利算法进行最优bbox匹配"""
    try:
        from scipy.optimize import linear_sum_assignment
        
        # 构建IOU代价矩阵
        cost_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                # 使用1-IOU作为代价
                iou = enhanced_compute_iou(pred_box, gt_box)
                cost_matrix[i, j] = 1.0 - iou
        
        # 执行匈牙利算法
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # 计算总得分
        total_iou = 0.0
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = 1.0 - cost_matrix[pred_idx, gt_idx]
            total_iou += max(0.0, iou)
        
        return total_iou / max(len(pred_boxes), len(gt_boxes))
        
    except ImportError:
        # 如果没有scipy，降级到greedy匹配
        return greedy_bbox_matching(pred_boxes, gt_boxes)

def greedy_bbox_matching(pred_boxes, gt_boxes):
    """贪心bbox匹配策略"""
    ious = []
    for pred_box in pred_boxes:
        max_iou = max(enhanced_compute_iou(pred_box, gt_box) for gt_box in gt_boxes)
        ious.append(max_iou)
    return sum(ious) / len(ious) if ious else 0.0


# F0.5
def compute_f05_char_level(ref, pred):
    ref_chars = set(ref)
    pred_chars = set(pred)
    correct = len(ref_chars & pred_chars)
    pred_total = len(pred_chars)
    ref_total = len(ref_chars)
    if pred_total == 0 or ref_total == 0:
        return 0.0
    precision = correct / pred_total
    recall = correct / ref_total
    beta = 0.5
    return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall) if (precision + recall) > 0 else 0.0


# 加载数据
with open('data/train_data_with_bounding_box.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

with open('data/train_predict.json', 'r', encoding='utf-8') as f:
    pred_data = json.load(f)

pred_map = {item["fk_homework_id"]: item for item in pred_data}

total_score = 0.0
total_f05_score = 0.0
total_iou_score = 0.0
total_count = len(test_data)

print("开始评估...")
print(f"总数据量: {total_count}")

for i, gt in enumerate(test_data):
    fkid = gt["fk_homework_id"]

    if fkid not in pred_map:
        f05 = 0.0
        iou_score = 0.0
    else:
        pred = pred_map[fkid]
        f05 = compute_f05_char_level(gt["target_text"], pred["predict_text"])

        gt_boxes = gt.get("bounding_box_list", [])
        pred_boxes = pred.get("bounding_box_list", [])
        
        # 使用增强的bbox匹配策略计算IOU得分
        if pred_boxes and gt_boxes:
            iou_score = compute_bbox_matching_score(pred_boxes, gt_boxes, "hybrid")
        elif not pred_boxes and not gt_boxes:
            iou_score = 1.0  # 都没有bbox时视为完全匹配
        else:
            iou_score = 0.0  # 一个有bbox另一个没有时为0
    
    # 改进的加权策略：动态调整权重
    # 当F0.5得分较高时，更重视IOU得分的准确性
    f05_weight = 0.5
    iou_weight = 0.5
    
    # 动态权重调整：F0.5得分越高，IOU权重越大
    if f05 > 0.8:
        f05_weight = 0.4
        iou_weight = 0.6
    elif f05 < 0.3:
        f05_weight = 0.6
        iou_weight = 0.4
    
    final = f05_weight * f05 + iou_weight * iou_score
    total_score += final
    total_f05_score += f05
    total_iou_score += iou_score
    
    # 显示进度
    if (i + 1) % 100 == 0 or (i + 1) == total_count:
        avg_so_far = total_score / (i + 1)
        print(f"进度: {i + 1}/{total_count}, 当前平均得分: {avg_so_far:.4f}")

average = total_score / total_count if total_count > 0 else 0.0
average_f05 = total_f05_score / total_count if total_count > 0 else 0.0
average_iou = total_iou_score / total_count if total_count > 0 else 0.0

print(f"\n=== 评估结果 ===")
print(f"平均总得分: {average:.4f}")
print(f"平均F0.5得分: {average_f05:.4f}")
print(f"平均IOU得分: {average_iou:.4f}")
print(f"IOU得分提升潜力: {(1.0 - average_iou) * 0.5:.4f}")

# 分析结果分布
score_ranges = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
total_score_temp = 0.0

for gt in test_data:
    fkid = gt["fk_homework_id"]
    if fkid in pred_map:
        pred = pred_map[fkid]
        f05 = compute_f05_char_level(gt["target_text"], pred["predict_text"])
        gt_boxes = gt.get("bounding_box_list", [])
        pred_boxes = pred.get("bounding_box_list", [])
        
        if pred_boxes and gt_boxes:
            iou_score = compute_bbox_matching_score(pred_boxes, gt_boxes, "hybrid")
        elif not pred_boxes and not gt_boxes:
            iou_score = 1.0
        else:
            iou_score = 0.0
        
        final = 0.5 * f05 + 0.5 * iou_score
        total_score_temp += final
        
        # 统计得分分布
        if final < 0.2:
            score_ranges['0.0-0.2'] += 1
        elif final < 0.4:
            score_ranges['0.2-0.4'] += 1
        elif final < 0.6:
            score_ranges['0.4-0.6'] += 1
        elif final < 0.8:
            score_ranges['0.6-0.8'] += 1
        else:
            score_ranges['0.8-1.0'] += 1

print(f"\n=== 得分分布 ===")
for range_name, count in score_ranges.items():
    percentage = (count / total_count) * 100
    print(f"{range_name}: {count} ({percentage:.1f}%)")

print(f"\n使用增强算法的平均得分: {total_score_temp / total_count:.4f}")
