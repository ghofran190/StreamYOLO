# eval_and_artifacts.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import glob
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import yaml

def load_names_from_yaml(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get('names', {})

def plot_confusion(cm, class_names, outpath):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title='Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_preds_csv', type=str, help='CSV with columns image, true_class, pred_class (optional)')
    parser.add_argument('--data_yaml', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='outputs/eval')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    class_names = load_names_from_yaml(args.data_yaml)
    if args.val_preds_csv and os.path.exists(args.val_preds_csv):
        import pandas as pd
        df = pd.read_csv(args.val_preds_csv)
        y_true = df['true_class'].values
        y_pred = df['pred_class'].values
    else:
        print("No CSV predictions provided — can't compute confusion automatically. Exiting.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Accuracy: {acc:.4f}, F1 (macro): {f1:.4f}')
    with open(os.path.join(args.outdir, 'metrics.txt'), 'w') as f:
        f.write(f'accuracy: {acc}\nf1_macro: {f1}\n')
    plot_confusion(cm, class_names, os.path.join(args.outdir, 'confusion.png'))

if __name__ == '__main__':
    main()
