# predict_val_create_csv.py
import os, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import yaml
import cv2
import pandas as pd
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', default='val_preds.csv')
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.data) as f:
        data = yaml.safe_load(f)
    val_dir = data['val']
    names = data.get('names', {})
    model = YOLO(args.weights)
    rows = []
    for img_path in Path(val_dir).glob('*'):
        if not img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            continue
        res = model.predict(str(img_path), imgsz=640, save=False, verbose=False)
        # res is sequence; take first prediction for image
        boxes = res[0].boxes if hasattr(res[0], 'boxes') else []
        # choose top predicted class (or -1 if none)
        pred_class = -1
        if len(boxes) > 0:
            pred_class = int(boxes.data[0, 5].item())  # index may vary by ultralytics version
        # true class: we assume corresponding label file contains first class id
        label_file = Path(str(img_path).replace('/images/', '/labels/')).with_suffix('.txt')
        true_class = -1
        if label_file.exists():
            with open(label_file) as f:
                l = f.readline().strip().split()
                if len(l) > 0:
                    true_class = int(l[0])
        rows.append({'image':str(img_path), 'true_class':true_class, 'pred_class':pred_class})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Saved", args.out)

if __name__ == '__main__':
    main()
