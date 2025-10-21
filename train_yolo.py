import os
# Permet de contourner le conflit OpenMP (libiomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from ultralytics import YOLO
import wandb
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='path to data.yaml')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--model', type=str, default='yolov8n.pt')
    p.add_argument('--project', type=str, default='yolo_streamlit')
    p.add_argument('--name', type=str, default='run1')
    p.add_argument('--device', type=str, default='0')  # 'cpu' or '0'
    p.add_argument('--save_examples', type=str, default='outputs/examples')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_examples, exist_ok=True)

    # init wandb
    wandb.init(project=args.project, name=args.name, config={
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'model': args.model
    })

    # create YOLO model instance (pretrained)
    model = YOLO(args.model)

    # Train - ultralytics integrates with wandb automatically if wandb.init() is active
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=args.project, name=args.name, exist_ok=True)
    
    
    # After training, run validation and save some example images
    print("Running validation and saving example detections...")
    # running val (will produce metrics and optionally save visuals)
    val_results = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, save=True)
    # ultralytics saves visualized predictions in runs/detect/exp or runs/segment/exp etc.
    # We'll find the latest run folder and copy 3 example images to save_examples
    runs_dir = os.path.join('runs', 'detect')
    if not os.path.exists(runs_dir):
        # some versions may save to 'runs/detect'
        runs_dir = os.path.join('runs')
    # find latest folder with name matching project/name
    candidate = None
    runs = []
    for root, dirs, files in os.walk('runs'):
        for d in dirs:
            if args.name in d:
                runs.append(os.path.join(root, d))
    # fallback: pick newest folder under runs
    if runs:
        candidate = sorted(runs, key=os.path.getmtime)[-1]
    else:
        # pick any runs/detect/exp* latest
        for root, dirs, files in os.walk('runs'):
            for d in dirs:
                runs.append(os.path.join(root, d))
        if runs:
            candidate = sorted(runs, key=os.path.getmtime)[-1]

    if candidate:
        # copy up to 3 images from candidate
        imgs = []
        for root, dirs, files in os.walk(candidate):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    imgs.append(os.path.join(root, f))
        imgs = sorted(imgs, key=os.path.getmtime, reverse=True)[:3]
        for i, img in enumerate(imgs):
            shutil.copy(img, os.path.join(args.save_examples, f'example_{i+1}.jpg'))

    # Log final metrics to wandb
    # val_results contains metrics summary in ultralytics; we can log a few things:
    try:
        # val_results is a list/dict, adapt per ultralytics version
        metrics = getattr(val_results, 'metrics', None) or val_results
        # Here we attempt safe extraction
        if isinstance(metrics, dict):
            wandb.log(metrics)
    except Exception as e:
        print("Could not log detailed metrics to wandb:", e)

    wandb.finish()
    print("Training complete.")

if __name__ == '__main__':
    main()
   
