# Paste this ENTIRE cell into a Kaggle notebook and run it.

import os, glob, sys, time
from pathlib import Path

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# ── 1. Clone repo ──────────────────────────────────────────────────────────
REPO = "https://github.com/ATIK2110018/PI-GNN.git"
DST  = "/kaggle/working/PI-GNN"

os.chdir("/kaggle/working")
if os.path.isdir(DST):
    log("Removing old clone for fresh start...")
    os.system("rm -rf PI-GNN")

log(f"Cloning {REPO} ...")
os.system(f"git clone {REPO}")
log("Clone done.")
os.chdir(DST)

# ── 2. Install PyTorch Geometric (pre-compiled wheels) ─────────────────────
try:
    import torch_geometric
    log(f"torch-geometric {torch_geometric.__version__} already installed")
except ImportError:
    import torch
    TORCH_VER = torch.__version__.split("+")[0]
    CUDA_VER  = torch.version.cuda
    WHEEL_URL = f"https://data.pyg.org/whl/torch-{TORCH_VER}+cu{CUDA_VER.replace('.','')}.html"
    log(f"Installing PyG wheels (PyTorch {TORCH_VER}, CUDA {CUDA_VER}) ...")
    for pkg in ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster"]:
        os.system(f"pip install {pkg} -f {WHEEL_URL} -q")
    os.system("pip install torch-geometric -q")
    log("torch-geometric installed.")

# ── 3. Install remaining deps ───────────────────────────────────────────────
for pkg in ["pyproj", "rasterio"]:
    try:
        __import__(pkg.replace("-", "_"))
        log(f"{pkg} already installed")
    except ImportError:
        log(f"Installing {pkg} ...")
        os.system(f"pip install {pkg} -q")
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
    log(f"Data found: {data_root}")
    os.system("rm -rf ./data")
    os.system(f"ln -s '{data_root}' ./data")
    if os.path.exists("./data/Sacramento_River_NLCD_LandCover.tif"):
        log("NLCD confirmed")
    else:
        log("WARNING: NLCD missing!")
else:
    log("ERROR: Data not found!")
    sys.exit(1)

# ── 5. Run (real-time output) ──────────────────────────────────────────────
log("Starting training...\n" + "=" * 60)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"

import subprocess
proc = subprocess.Popen(
    ["python", "main.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    bufsize=1, text=True
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
if proc.returncode != 0:
    log(f"ERROR: main.py exited with code {proc.returncode}")
