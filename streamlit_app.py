# streamlit_app.py
import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import zipfile
import subprocess
import time
import wandb
from pathlib import Path
from tempfile import TemporaryDirectory
import glob
from PIL import Image

st.set_page_config(page_title="YOLO Trainer UI", layout="wide")
st.title("Mini UI: config / train / view results — YOLO + W&B")

# --- Sidebar: dataset selection
st.sidebar.header("Dataset")
use_upload = st.sidebar.checkbox("Upload dataset zip", value=True)
dataset_path = None

if use_upload:
    uploaded = st.sidebar.file_uploader("Upload dataset.zip (YOLO format)", type=['zip'])
    if uploaded:
        tmp = TemporaryDirectory()
        zip_path = os.path.join(tmp.name, 'dataset.zip')
        with open(zip_path, 'wb') as f:
            f.write(uploaded.getbuffer())
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall('dataset')  # extracts to ./dataset
        dataset_path = 'dataset'
        st.sidebar.success("Dataset extracted to ./dataset")
else:
    dataset_path = st.sidebar.text_input("Path to dataset folder", value='dataset')

# --- Sidebar: hyperparameters
st.sidebar.header("Hyperparams")
model_name = st.sidebar.selectbox("YOLO model (pretrained)", ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'])
epochs = st.sidebar.number_input("epochs", min_value=1, value=20)
batch = st.sidebar.number_input("batch size", min_value=1, value=16)
imgsz = st.sidebar.number_input("image size", min_value=128, value=640)
device = st.sidebar.text_input("device (e.g. 0 or cpu)", value='0')
run_name = st.sidebar.text_input("Run name", value=f"run_{int(time.time())}")
project = st.sidebar.text_input("W&B project name", value="yolo_streamlit")

# --- Sidebar: control
st.sidebar.header("Control")
start_button = st.sidebar.button("Start training")
status_placeholder = st.sidebar.empty()

# --- Main
st.markdown("## Training console")
console = st.empty()

def find_data_yaml(root_folder):
    """Find the first data.yaml file recursively under root_folder."""
    root = Path(root_folder)
    matches = list(root.rglob("data.yaml"))  # recursive search
    if not matches:
        return None
    if len(matches) > 1:
        print(f"⚠️ Multiple data.yaml found, using the first: {matches[0]}")
    return matches[0]

def show_detection_examples():
    """Display up to 5 example detection images from the last YOLO run."""
    base_dir = Path("runs/detect")
    if not base_dir.exists():
        st.warning("No detection results found yet.")
        return
    # pick latest exp folder
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        st.warning("No detection experiment found in runs/detect/")
        return
    last_exp = max(subdirs, key=lambda d: d.stat().st_mtime)
    st.success(f"Displaying examples from: {last_exp.name}")
    img_files = list(last_exp.glob("*.jpg")) + list(last_exp.glob("*.png"))
    if not img_files:
        st.warning("No images found in the latest experiment folder.")
        return
    cols = st.columns(min(3, len(img_files)))
    for i, img_path in enumerate(img_files[:5]):
        cols[i % 3].image(Image.open(img_path), caption=img_path.name, use_column_width=True)

if start_button:
    if not os.path.exists(dataset_path):
        st.sidebar.error("Dataset path not found.")
    else:
        # detect data.yaml
        data_yaml = find_data_yaml(dataset_path)
        if not data_yaml:
            st.sidebar.error("data.yaml not found in dataset folder. Please ensure YOLO format.")
        else:
            status_placeholder.info("Launching training...")
            # build command
            cmd = [
                "python", "train_yolo.py",
                "--data", str(data_yaml),
                "--epochs", str(epochs),
                "--batch", str(batch),
                "--imgsz", str(imgsz),
                "--model", model_name,
                "--project", project,
                "--name", run_name,
                "--device", device,
                "--save_examples", "outputs/examples"
            ]
            # display command
            console.text("Running command:\n" + " ".join(map(str, cmd)))

            # run subprocess and stream logs
            p = subprocess.Popen(
                list(map(str, cmd)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            log_text = ""
            for line in p.stdout:
                log_text += line
                console.text(log_text)
            p.wait()

            status_placeholder.success("Training finished. Fetching results...")

            # --- Show example images
            st.header("📸 Example detections")
            show_detection_examples()

            # --- W&B summary
            st.markdown("### W&B summary (latest run)")
            runs = []
            try:
                api = wandb.Api()
                entity = wandb.run.entity if wandb.run else None
                runs = api.runs(f"{entity}/{project}") if entity else api.runs(project)
            except Exception as e:
                st.info("Could not access W&B API:", e)
                runs = []

            if runs:
                matched = [r for r in runs if r.name == run_name]
                r = matched[0] if matched else runs[0]
                st.write("Run:", r.name)
                summary = getattr(r, 'summary', {})
                st.write("Summary metrics (selected):")
                st.json({k: summary.get(k) for k in list(summary.keys())[:10]})
            else:
                st.info("No runs found on W&B (or couldn't access).")

            # --- Local artifacts / metrics
            st.markdown("### Local artifacts")
            cm_path = Path("outputs/eval/confusion.png")
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion matrix")
            else:
                st.info("No local confusion matrix found.")

            metrics_file = Path("outputs/eval/metrics.txt")
            if metrics_file.exists():
                st.text(metrics_file.read_text())
            else:
                st.info("No local metrics file found.")

st.markdown("---")
st.markdown("### Notes")
st.markdown("- Assure-toi d’avoir une GPU compatible et d’installer les drivers CUDA si besoin.")
st.markdown("- `train_yolo.py` créera `outputs/examples` avec quelques images (si possible).")

