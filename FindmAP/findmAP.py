import os
import json
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

# Function to convert YOLO format to COCO format
def yolo_to_coco(yolo_path, coco_output_path, images_path, is_ground_truth=True):
    categories = [{'id': 1, 'name': 'pest'}]
    coco_data = {
        'images': [],
        'annotations': [] if is_ground_truth else None,
        'categories': categories
    }
    detections = []

    image_id = 0
    annotation_id = 0
    for filename in os.listdir(yolo_path):
        if filename.endswith('.txt'):
            image_filename = filename.replace('.txt', '.jpg')
            image_path = os.path.join(images_path, image_filename)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            coco_data['images'].append({
                'id': image_id,
                'file_name': image_filename,
                'height': height,
                'width': width
            })

            with open(os.path.join(yolo_path, filename), 'r') as file:
                for line in file.readlines():
                    class_id, x_center, y_center, bbox_width, bbox_height, *conf = map(float, line.strip().split())
                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    bbox_width = bbox_width * width
                    bbox_height = bbox_height * height

                    if is_ground_truth:
                        coco_data['annotations'].append({
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': int(class_id) + 1,
                            'bbox': [x_min, y_min, bbox_width, bbox_height],
                            'area': bbox_width * bbox_height,
                            'iscrowd': 0
                        })
                        annotation_id += 1
                    else:
                        detection = {
                            'image_id': image_id,
                            'category_id': int(class_id) + 1,
                            'bbox': [x_min, y_min, bbox_width, bbox_height],
                            'score': conf[0] if conf else 1.0  # confidence score for predictions
                        }
                        detections.append(detection)
            image_id += 1

    if is_ground_truth:
        with open(coco_output_path, 'w') as outfile:
            json.dump(coco_data, outfile)
    else:
        with open(coco_output_path, 'w') as outfile:
            json.dump(detections, outfile)

# Paths
ground_truth_path = r'D:/git/New folder/data/val/labels'
predictions_path = r'D:/git/New folder/data/val/detectionswitheval1class'
images_path = r'D:/git/New folder/data/val/images'
ground_truth_coco_path = r'D:/git/New folder/data/val/ground_truth_coco.json'
predictions_coco_path = r'D:/git/New folder/data/val/predictions_coco.json'

# Convert YOLO annotations to COCO format
yolo_to_coco(ground_truth_path, ground_truth_coco_path, images_path, is_ground_truth=True)
yolo_to_coco(predictions_path, predictions_coco_path, images_path, is_ground_truth=False)

# Load COCO ground truth and predictions
coco_gt = COCO(ground_truth_coco_path)
coco_dt = coco_gt.loadRes(predictions_coco_path)

# Evaluate
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Extracting detailed results
def print_detailed_results(coco_eval):
    precision = coco_eval.eval['precision']
    recall = coco_eval.eval['recall']
    IoU_thresholds = coco_eval.params.iouThrs
    cat_ids = coco_eval.params.catIds

    iou_50_index = np.where(IoU_thresholds == 0.5)[0][0]

    for cat_id in cat_ids:
        ap = precision[iou_50_index, :, cat_id, 0, 2]
        ap = np.mean(ap[ap > -1])
        print(f"class_id = {cat_id - 1}, name = pest, ap = {ap * 100:.2f}%")

        # Precision, recall, and F1-score at IoU=0.50 and conf_thresh=0.25
        conf_thresh_index = 1  # Adjust this index based on actual thresholds used in your model
        tp = np.sum(coco_eval.eval['dtMatches'][iou_50_index, :, cat_id, conf_thresh_index])
        fp = np.sum(coco_eval.eval['dtScores'][iou_50_index, :, cat_id, conf_thresh_index] == 0)
        fn = np.sum(coco_eval.eval['gtIgnore'][iou_50_index, :, cat_id] == 0) - tp

        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

        print(f" for conf_thresh = 0.25, precision = {precision_val:.2f}, recall = {recall_val:.2f}, F1-score = {f1_score:.2f}")
        print(f" for conf_thresh = 0.25, TP = {tp}, FP = {fp}, FN = {fn}")

        ious = coco_eval.eval['ious'][iou_50_index, :, cat_id, conf_thresh_index]
        avg_iou = np.mean(ious[ious > -1])
        print(f" average IoU = {avg_iou * 100:.2f} %\n")

print_detailed_results(coco_eval)
