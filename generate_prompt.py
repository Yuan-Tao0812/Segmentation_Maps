import os
import json
import csv
from collections import defaultdict, Counter
import cv2
from tqdm import tqdm

# === 路径配置（请根据你的目录结构修改） ===
YOLO_LABELS_DIR = "/content/drive/MyDrive/VisDrone2019-YOLO/VisDrone2019-YOLO-train/labels/"         # YOLO格式标签文件夹
SEMANTIC_MASKS_DIR = "/content/drive/MyDrive/VisDrone2019-DET/VisDrone2019-DET-train/segmaps/"   # 语义分割单通道PNG文件夹
CSV_PATH = "/content/drive/MyDrive/ControlNet-train/object150_info.csv"           # 类别信息CSV路径
OUTPUT_JSON = "/content/drive/MyDrive/ControlNet-train/prompt.json"

# 这里假设你的图片文件都在同一个文件夹，source和target路径可以拼出来，自己根据实际路径修改
SOURCE_IMG_DIR = "/content/drive/MyDrive/ControlNet-train/source/"
TARGET_IMG_DIR = "/content/drive/MyDrive/ControlNet-train/target/"

# === 加载语义类别ID到名称的映射 ===
def load_semantic_classes(csv_path):
    id_to_name = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['id'])
            name = row['name'].strip()
            id_to_name[sid] = name
    return id_to_name

# === YOLO标签加载器 ===
def load_yolo_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    objects = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        objects.append((cls_id, cx, cy, w, h))
    return objects

# === 获取实例对应的语义背景 ===
def get_dominant_semantic_label(mask, bbox):
    H, W = mask.shape
    cx, cy, w, h = bbox
    x1 = max(0, int((cx - w / 2) * W))
    y1 = max(0, int((cy - h / 2) * H))
    x2 = min(W, int((cx + w / 2) * W))
    y2 = min(H, int((cy + h / 2) * H))
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return None
    label_counts = Counter(region.flatten())
    most_common = label_counts.most_common(1)
    return most_common[0][0] if most_common else None

# === 主函数 ===
def generate_prompt_json(yolo_dir, mask_dir, csv_path, output_path):
    semantic_id_to_name = load_semantic_classes(csv_path)
    results = []

    for label_file in tqdm(os.listdir(yolo_dir)):
        if not label_file.endswith(".txt"):
            continue

        image_id = os.path.splitext(label_file)[0]
        label_path = os.path.join(yolo_dir, label_file)
        mask_path = os.path.join(mask_dir, image_id + ".png")
        if not os.path.exists(mask_path):
            print(f"Missing mask for {image_id}, skipping.")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask for {image_id}, skipping.")
            continue

        objects = load_yolo_labels(label_path)

        class_region_map = defaultdict(list)  # {cls_id_str: [semantic_names]}
        for cls_id, cx, cy, w, h in objects:
            semantic_id = get_dominant_semantic_label(mask, (cx, cy, w, h))
            if semantic_id is None:
                continue
            semantic_name = semantic_id_to_name.get(semantic_id, f"class_{semantic_id}")
            class_region_map[str(cls_id)].append(semantic_name)

        # 构造自然语言句子
        object_descriptions = []
        for obj_class, regions in class_region_map.items():
            region_counter = Counter(regions)
            most_common_region = region_counter.most_common(1)[0][0]
            count = len(regions)
            # 这里用你csv里面的类别名称代替class_id
            class_name = None
            try:
                # 尝试转回int找名字
                class_name = semantic_id_to_name.get(int(obj_class))
            except:
                class_name = None
            if class_name is None:
                class_name = f"class {obj_class}"

            # 复数简单加s，建议你根据实际类别英文调整，或者你可以扩展这里的复数规则
            description = f"{count} {class_name}{'s' if count > 1 else ''} mainly on {most_common_region}"
            object_descriptions.append(description)

        if object_descriptions:
            if len(object_descriptions) == 1:
                sentence = "There are " + object_descriptions[0] + "."
            else:
                sentence = "There are " + ", ".join(object_descriptions[:-1])
                sentence += ", and " + object_descriptions[-1] + "."
        else:
            sentence = "No detectable objects in the image."

        results.append({
            "source": os.path.join(SOURCE_IMG_DIR, image_id + ".png"),
            "target": os.path.join(TARGET_IMG_DIR, image_id + ".png"),
            "prompt": sentence
        })

    # 写入JSON文件（格式化缩进方便查看）
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved descriptions to {output_path}")

# === 执行脚本 ===
if __name__ == "__main__":
    generate_prompt_json(YOLO_LABELS_DIR, SEMANTIC_MASKS_DIR, CSV_PATH, OUTPUT_JSON)
