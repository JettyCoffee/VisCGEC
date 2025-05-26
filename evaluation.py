# 文件路径可自行修改
import json

# 计算IOU
def compute_iou(box1, box2):
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
total_count = len(test_data)

for gt in test_data:
    fkid = gt["fk_homework_id"]

    if fkid not in pred_map:
        f05 = 0.0
        iou_score = 0.0
    else:
        pred = pred_map[fkid]
        f05 = compute_f05_char_level(gt["target_text"], pred["predict_text"])

        gt_boxes = gt.get("bounding_box_list", [])
        pred_boxes = pred.get("bounding_box_list", [])
        iou_score = 0.0
        if pred_boxes:
            ious = []
            for pb in pred_boxes:
                #一对一计算IOU
                max_iou = max(compute_iou(pb, gb) for gb in gt_boxes)
                ious.append(max_iou)
            iou_score = sum(ious) / len(ious) if ious else 0.0
    #加权求和
    final = 0.5 * f05 + 0.5 * iou_score
    total_score += final

average = total_score / total_count if total_count > 0 else 0.0
print(f"平均得分: {average:.4f}")
