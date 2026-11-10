#!/bin/bash
#SBATCH --job-name=msgnn_patch2_smoke
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out
#SBATCH --error=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.err

# mSWE-GNN Stockbridge patch 2 smoke test: RainMSGNN (rain forcing in latent space)
# checks with random weights + a 200-step overfit of single transitions. No training.

set -euo pipefail

REPO_DIR="$HOME/flood/01_repositories/mSWE-GNN_Stockbridge"
SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
DATASET="$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_100_5m_v01"
OUT_DIR="$HOME/flood/03_outputs/mSWE-GNN_Stockbridge/patch2_smoke_v01"

cd "$REPO_DIR"
module load apptainer/1.4.1

echo "===== mSWE-GNN Stockbridge patch 2 smoke ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Git commit: $(git rev-parse HEAD)"
git status --short
echo "Dataset: $DATASET"
echo "Output: $OUT_DIR"

test -f "$SIF"
test -f "$DATASET/static_graph.pt"

if [ -e "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
    echo "ERROR: $OUT_DIR is not empty. Refusing to overwrite."
    exit 1
fi

nvidia-smi || true

echo "===== Python packages used by the original repo imports ====="
apptainer exec --nv "$SIF" python -c "
import importlib.util as u
for m in ['torch','torch_geometric','numpy','scipy','sklearn','matplotlib','yaml','networkx','shapely','xarray','tqdm','lightning','meshkernel','triangle','pandas','wandb']:
    print(f'{m:16s}', 'yes' if u.find_spec(m) else 'MISSING')
"

echo "===== Smoke test ====="
apptainer exec --nv "$SIF" python stockbridge/smoke_patch2.py \
    --dataset-root "$DATASET" \
    --output-dir "$OUT_DIR" \
    --extra-event R001 \
    --scales 4 \
    --previous-t 2 \
    --rain-scale-mm 10 \
    --wet-t 12 \
    --rollout-steps 24 \
    --overfit-steps 200 \
    --overfit-lr 1e-3 \
    --device auto

echo "===== Done ====="
echo "Date: $(date)"
ls -la "$OUT_DIR"
