#!/bin/bash
#SBATCH --job-name=swegnn300_5m_prepare
#SBATCH --account=comet_smhfs
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.out
#SBATCH --error=/mnt/nfs/home/c4063745/ResearchPam_HPC/logs/%x_%j.err

# Patch 4b: Stockbridge phase 2+3 (300 events) -> SWE-GNN 5 m dataset for RainMSGNN.
#
#   1. reproduction: convert pt/stockbridge_phase2_core_v01 into a temporary
#      folder and require static_graph.pt + 100 events bit-identical to
#      swegnn/stockbridge_phase2_100_5m_v01. Stops here on any difference
#      (the temporary folder is kept for inspection).
#   2. conversion: pt/stockbridge_phase2plus3_300_v01 -> swegnn/stockbridge_phase2plus3_300_5m_v01
#      (refuses to start if the output exists).
#   3. validation: <output>/validation_report.json (stops on failure).
#   4. informational, never fails the job: train_rain_msgnn.py --smoke on the
#      new dataset, CPU.
#
# Same CPU partition, resources and container as the job that built the
# reference (Stockbridge-SWEGNN/slurm/m16/swegnn_phase2_100_5m_convert_v01.sbatch).
# Launch from the repository root:
#   sbatch stockbridge/data_prep/run_prepare_phase2plus3_300.sbatch
# Estimate: ~20-40 min; disk 5.6 GiB final + 1.9 GiB temporary (removed on success).

set -euo pipefail

REPO_DIR="$HOME/flood/01_repositories/mSWE-GNN_Stockbridge"
SIF="$HOME/ResearchPam_HPC/containers/stockbridge_swegnn_pyg_torch212_cu121_v01.sif"
BASE="$HOME/flood/02_data/CityCAT_Winchester"
PT2="$BASE/pt/stockbridge_phase2_core_v01"
PT300="$BASE/pt/stockbridge_phase2plus3_300_v01"
PT300_MANIFEST="$PT300/stockbridge_phase2plus3_300_manifest.csv"
REFERENCE="$BASE/swegnn/stockbridge_phase2_100_5m_v01"
OUT="$BASE/swegnn/stockbridge_phase2plus3_300_5m_v01"
JOB="${SLURM_JOB_ID:-local$$}"
TMP_ROOT="$HOME/ResearchPam_HPC/tmp"
REPRO_DIR="$TMP_ROOT/swegnn_phase2_repro_$JOB"
LOG_DIR="$HOME/ResearchPam_HPC/logs"
REPRO_REPORT="$LOG_DIR/swegnn300_reproduction_report_$JOB.json"
SMOKE_OUT="$HOME/flood/03_outputs/mSWE-GNN_Stockbridge/patch4b_dataset_smoke/$JOB"
CELL_SIZE=5.0
RAIN_SCALE=300000.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
cd "$REPO_DIR"
module load apptainer/1.4.1
PY=(apptainer exec "$SIF" python)

echo "===== Patch 4b: SWE-GNN 300 events at 5 m ====="
echo "Date: $(date)"
echo "Host: $(hostname) | job $JOB"
echo "Git commit: $(git rev-parse HEAD)"
git status --short -- stockbridge/data_prep
echo "Container: $SIF"
echo "Reference: $REFERENCE"
echo "Output: $OUT"
echo "Cell size: $CELL_SIZE | rainfall scale reference: $RAIN_SCALE"

test -f "$SIF"
test -d "$PT2/train" && test -d "$PT300/train" && test -f "$PT300_MANIFEST"
test -f "$REFERENCE/static_graph.pt" && test -f "$REFERENCE/manifest.json"
if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    echo "ERROR: $OUT already exists. Refusing to overwrite."
    exit 1
fi
if [ -e "$REPRO_DIR" ]; then
    echo "ERROR: $REPRO_DIR already exists."
    exit 1
fi
for SPLIT in train valid test; do
    echo "PT counts $SPLIT: phase2 $(find "$PT2/$SPLIT" -maxdepth 1 -name '*.pt' | wc -l)," \
         "phase2+3 $(find "$PT300/$SPLIT" -maxdepth 1 -name '*.pt' | wc -l)"
done
echo "PT manifest sha256: $(sha256sum "$PT300_MANIFEST" | cut -d' ' -f1)"
echo "Reference manifest.json options:"
"${PY[@]}" -c "
import json, sys
m = json.load(open(sys.argv[1]))
for k in ('script_name', 'pt_root', 'cell_size', 'rainfall_scale_reference', 'split_counts'):
    print(f'  {k}: {m.get(k)}')
assert float(m['cell_size']) == $CELL_SIZE, 'reference cell_size differs'
assert float(m['rainfall_scale_reference']) == $RAIN_SCALE, 'reference rainfall scale differs'
" "$REFERENCE/manifest.json"
mkdir -p "$TMP_ROOT" "$LOG_DIR"
echo "Free disk (output filesystem):"
df -h "$BASE" "$TMP_ROOT" | sed 's/^/  /'
FREE_KB=$(df -Pk "$BASE" | awk 'NR == 2 {print $4}')
if [ "$FREE_KB" -lt $((10 * 1024 * 1024)) ]; then
    echo "ERROR: less than 10 GiB free on the output filesystem."
    exit 1
fi

echo
echo "===== 1/4 Reproduction of stockbridge_phase2_100_5m_v01 ($(date +%T)) ====="
# Every event file stores source_path as a string, so use the pt_root string
# recorded by the reference (same folder as $PT2, checked) to get identical bytes.
REF_PT_ROOT=$("${PY[@]}" -c "import json, sys; print(json.load(open(sys.argv[1]))['pt_root'])" \
    "$REFERENCE/manifest.json")
if [ "$(readlink -f "$REF_PT_ROOT")" != "$(readlink -f "$PT2")" ]; then
    echo "ERROR: reference pt_root $REF_PT_ROOT is not $PT2"
    exit 1
fi
echo "PT root (string from the reference manifest): $REF_PT_ROOT"
"${PY[@]}" stockbridge/data_prep/prepare_swegnn_dataset.py \
    --pt-root "$REF_PT_ROOT" \
    --output-root "$REPRO_DIR" \
    --cell-size "$CELL_SIZE" \
    --rainfall-scale-reference "$RAIN_SCALE"
if ! "${PY[@]}" stockbridge/data_prep/compare_swegnn_datasets.py \
        --reference "$REFERENCE" --candidate "$REPRO_DIR" --report "$REPRO_REPORT"; then
    echo "STOP: reproduction is not bit-identical. Report: $REPRO_REPORT"
    echo "Temporary dataset kept for inspection: $REPRO_DIR"
    exit 2
fi
case "$REPRO_DIR" in
    "$TMP_ROOT"/swegnn_phase2_repro_*) rm -rf -- "$REPRO_DIR" ;;
    *) echo "Not removing unexpected path $REPRO_DIR" ;;
esac
echo "Temporary reproduction removed."

echo
echo "===== 2/4 Conversion of 300 events ($(date +%T)) ====="
"${PY[@]}" stockbridge/data_prep/prepare_swegnn_dataset.py \
    --pt-root "$PT300" \
    --split-manifest "$PT300_MANIFEST" \
    --output-root "$OUT" \
    --cell-size "$CELL_SIZE" \
    --rainfall-scale-reference "$RAIN_SCALE"
cp "$REPRO_REPORT" "$OUT/reproduction_report.json"

echo
echo "===== 3/4 Validation ($(date +%T)) ====="
"${PY[@]}" stockbridge/data_prep/validate_swegnn_dataset.py \
    --dataset "$OUT" \
    --pt-manifest "$PT300_MANIFEST" \
    --reference-dataset "$REFERENCE"

echo
echo "===== 4/4 (informational) train_rain_msgnn.py --smoke on the new dataset ($(date +%T)) ====="
set +e
timeout 30m "${PY[@]}" stockbridge/train_rain_msgnn.py \
    --dataset-root "$OUT" --output-dir "$SMOKE_OUT" --smoke --device cpu
SMOKE_STATUS=$?
set -e
echo "Smoke exit status: $SMOKE_STATUS (0 = dataset accepted by train_rain_msgnn.py)"

echo
echo "===== Done ($(date)) ====="
du -sh "$OUT"
ls -la "$OUT"
