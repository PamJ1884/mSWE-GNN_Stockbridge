#!/bin/bash
#SBATCH --job-name=msgnn_patch1_smoke
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out
#SBATCH --error=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.err

# mSWE-GNN Stockbridge patch 1 smoke test: multiscale data + original MSGNN
# forward with random weights. No training.

set -euo pipefail

REPO_DIR="$HOME/flood/01_repositories/mSWE-GNN_Stockbridge"
SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
# DATASET and OUTPUT_DIR come from the environment (no default dataset), e.g.
#   DATASET=$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2plus3_300_5m_v01 \
#   OUTPUT_DIR=$HOME/flood/03_outputs/mSWE-GNN_Stockbridge/<run_name> \
#   sbatch stockbridge/run_smoke_patch1.sbatch
: "${DATASET:?set DATASET to the SWE-GNN dataset folder (e.g. swegnn/stockbridge_phase2plus3_300_5m_v01)}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the output folder}"
OUT_DIR="$OUTPUT_DIR"

cd "$REPO_DIR"
module load apptainer/1.4.1

echo "===== mSWE-GNN Stockbridge patch 1 smoke ====="
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
for m in ['torch','torch_geometric','numpy','scipy','sklearn','matplotlib','yaml','networkx','shapely','xarray','tqdm','lightning','meshkernel','triangle']:
    print(f'{m:16s}', 'yes' if u.find_spec(m) else 'MISSING')
"

echo "===== Smoke test ====="
apptainer exec --nv "$SIF" python stockbridge/smoke_patch1.py \
    --dataset-root "$DATASET" \
    --output-dir "$OUT_DIR" \
    --event R019 \
    --times 0,12 \
    --scales 4,5 \
    --previous-t 2,3 \
    --device auto

echo "===== Done ====="
echo "Date: $(date)"
ls -la "$OUT_DIR"
