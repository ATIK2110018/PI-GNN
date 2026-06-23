"""
Kaggle setup script for PI-GNN / PIGNN
Usage: paste the entire file into a Kaggle notebook cell.
"""

import os, glob, subprocess, sys, time
from pathlib import Path

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# ── 1. Clone repo ──────────────────────────────────────────────────────────
REPO = "https://github.com/ATIK2110018/PI-GNN.git"
DST  = "/kaggle/working/PI-GNN"

os.chdir("/kaggle/working")
if os.path.isdir(DST):
    log("Repo already exists, pulling...")
    os.system(f"git -C {DST} pull 2>/dev/null")
else:
    log(f"Cloning {REPO} ...")
    os.system(f"git clone {REPO}")
log("Clone done.")

os.chdir(DST)

# ── 2. Install PyTorch Geometric (pre-compiled wheels) ──────────────────────
try:
    import torch_geometric
    log(f"torch-geometric {torch_geometric.__version__} already installed")
except ImportError:
    import torch
    TORCH_VER = torch.__version__.split("+")[0]  # e.g. "2.5.0"
    CUDA_VER  = torch.version.cuda                 # e.g. "12.1"
    WHEEL_URL = f"https://data.pyg.org/whl/torch-{TORCH_VER}+cu{CUDA_VER.replace('.','')}.html"

    log(f"PyTorch {TORCH_VER} / CUDA {CUDA_VER}")
    log(f"Installing PyG wheels from {WHEEL_URL} ...")

    pkgs = ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster"]
    for pkg in pkgs:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-f", WHEEL_URL, "-q"],
            capture_output=True,
        )
    subprocess.run([sys.executable, "-m", "pip", "install", "torch-geometric", "-q"])
    log("torch-geometric installed.")

# ── 3. Install remaining deps ───────────────────────────────────────────────
for pkg in ["pyproj", "rasterio"]:
    try:
        __import__(pkg.replace("-", "_"))
        log(f"{pkg} already installed")
    except ImportError:
        log(f"Installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"])
log("All deps ready.")

# ── 4. Link data ────────────────────────────────────────────────────────────
data_root = None
for entry in sorted(glob.glob("/kaggle/input/*/")):
    if (Path(entry) / "hecras").is_dir():
        data_root = entry.rstrip("/")
        break
if data_root is None:
    for root, dirs, _ in os.walk("/kaggle/input"):
        if "hecras" in dirs:
            data_root = root
            break

if data_root:
    log(f"Data: {data_root}")
    os.system("rm -rf ./data")
    os.system(f"ln -s '{data_root}' ./data")
    nlcd = "./data/Sacramento_River_NLCD_LandCover.tif"
    if os.path.exists(nlcd):
        log("NLCD confirmed")
    else:
        log("WARNING: NLCD missing!")
else:
    log("ERROR: Data not found!")
    sys.exit(1)

log("Setup complete. Starting training...")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.system("python main.py")
