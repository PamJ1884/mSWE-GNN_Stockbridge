#!/bin/bash
#SBATCH --job-name=msgnn_patch3_pilot
#SBATCH --account=comet_smhfs
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%A_%a.out
#SBATCH --error=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%A_%a.err

# mSWE-GNN Stockbridge patch 3: pilot training of RainMSGNN (one GPU per array task).
#
# Array tasks (same seed, 4 scales, config.yaml hyperparameters):
#   0  fine        previous_t 2   -> patch3_pilot_v01/fine/
#   1  all_scales  previous_t 2   -> patch3_pilot_v01/all_scales/
#   2  fine        previous_t 3   -> patch3_pilot_v01/fine_p3/
#   3  all_scales  previous_t 3   -> patch3_pilot_v01/all_scales_p3/
#
# Launch (from the repository root):
#   sbatch stockbridge/run_pilot_patch3.sbatch                    # tasks 0-1 (p = 2), 12 h each
#   sbatch --array=2-3 stockbridge/run_pilot_patch3.sbatch        # tasks 2-3 (p = 3), no code edits
#   sbatch --array=0-3 stockbridge/run_pilot_patch3.sbatch        # all four
#   sbatch --time=24:00:00 stockbridge/run_pilot_patch3.sbatch    # other time limit (default 12 h)
# Extra arguments after the script name go to train_rain_msgnn.py, e.g.
#   sbatch stockbridge/run_pilot_patch3.sbatch --epochs 60 --max-starts-per-event 6
# Every task refuses to write into a non-empty output directory.

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
INJECTIONS=(fine all_scales fine all_scales)
PREVIOUS_TS=(2 2 3 3)
VARIANTS=(fine all_scales fine_p3 all_scales_p3)
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#VARIANTS[@]}" ]; then
    echo "ERROR: array task $TASK_ID has no variant (valid: 0-$((${#VARIANTS[@]} - 1)))"
    exit 1
fi
INJECTION="${INJECTIONS[$TASK_ID]}"
PREVIOUS_T="${PREVIOUS_TS[$TASK_ID]}"
VARIANT="${VARIANTS[$TASK_ID]}"
SEED=666  # models.seed of config.yaml, explicit so every task uses the same one

REPO_DIR="$HOME/flood/01_repositories/mSWE-GNN_Stockbridge"
SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
DATASET="$HOME/flood/02_data/CityCAT_Winchester/swegnn/stockbridge_phase2_100_5m_v01"
OUT_DIR="$HOME/flood/03_outputs/mSWE-GNN_Stockbridge/patch3_pilot_v01/$VARIANT"

cd "$REPO_DIR"
module load apptainer/1.4.1

echo "===== mSWE-GNN Stockbridge patch 3 pilot ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}} task $TASK_ID | variant $VARIANT"
echo "Rain injection: $INJECTION | previous_t: $PREVIOUS_T | seed: $SEED"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Git commit: $(git rev-parse HEAD)"
git status --short
echo "Dataset: $DATASET"
echo "Output: $OUT_DIR"
echo "Extra arguments: $*"

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

echo "===== Training ====="
apptainer exec --nv "$SIF" python stockbridge/train_rain_msgnn.py \
    --dataset-root "$DATASET" \
    --output-dir "$OUT_DIR" \
    --rain-injection "$INJECTION" \
    --rain-forcing multiplicative \
    --rain-residual \
    --previous-t "$PREVIOUS_T" \
    --num-scales 4 \
    --seed "$SEED" \
    --device cuda \
    "$@"

echo "===== Done ====="
echo "Date: $(date)"
ls -la "$OUT_DIR"
