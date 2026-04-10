#!/usr/bin/env bash
# =============================================================================
# Agile Autonomy — Google Cloud Training Script
# =============================================================================
# Supports two Google Cloud deployment modes:
#   MODE=gce      — Compute Engine VM with GPU (simpler, more control)
#   MODE=vertex   — Vertex AI Custom Training Job (managed, scalable)
#
# Usage:
#   chmod +x gcloud_train.sh
#
#   # Compute Engine:
#   MODE=gce bash gcloud_train.sh
#
#   # Vertex AI:
#   MODE=vertex bash gcloud_train.sh
#
# Prerequisites:
#   gcloud CLI installed and authenticated:
#       gcloud auth login
#       gcloud auth application-default login
#       gcloud config set project YOUR_PROJECT_ID
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these values before running
# ---------------------------------------------------------------------------
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-YOUR_PROJECT_ID}"   # your GCP project
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
BUCKET="${BUCKET:-gs://${PROJECT_ID}-agile-autonomy}"   # GCS bucket for data/ckpts
VM_NAME="${VM_NAME:-agile-autonomy-train}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"           # 8 vCPUs, 30 GB RAM
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"                 # t4 / v100 / a100
GPU_COUNT="${GPU_COUNT:-1}"
DISK_SIZE="${DISK_SIZE:-300GB}"
MODE="${MODE:-gce}"                                     # gce | vertex
# ---------------------------------------------------------------------------

DATASET_URL="https://zenodo.org/record/5517791/files/agile_autonomy_dataset.tar.xz?download=1"
REPO_URL="https://github.com/rayeed221/agile_autonomy.git"

# ===========================================================================
# Helper: print section headers
# ===========================================================================
section() { echo; echo "=== $* ==="; echo; }

# ===========================================================================
# Validate gcloud is configured
# ===========================================================================
section "Checking gcloud configuration"
if ! command -v gcloud &>/dev/null; then
    echo "ERROR: gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install"
    exit 1
fi
if [ "${PROJECT_ID}" = "YOUR_PROJECT_ID" ]; then
    echo "ERROR: Set PROJECT_ID at the top of this script or export GOOGLE_CLOUD_PROJECT=<id>"
    exit 1
fi
gcloud config set project "${PROJECT_ID}"
echo "Project : ${PROJECT_ID}"
echo "Region  : ${REGION}"
echo "Mode    : ${MODE}"

# ===========================================================================
# Create GCS bucket (if it does not exist)
# ===========================================================================
section "Setting up GCS bucket: ${BUCKET}"
if ! gsutil ls "${BUCKET}" &>/dev/null; then
    gsutil mb -l "${REGION}" "${BUCKET}"
    echo "Bucket created: ${BUCKET}"
else
    echo "Bucket already exists: ${BUCKET}"
fi

# ===========================================================================
# MODE: Google Compute Engine (GCE)
# ===========================================================================
if [ "${MODE}" = "gce" ]; then

    section "Creating Compute Engine VM: ${VM_NAME}"

    # Deep Learning VM image includes CUDA, cuDNN, Python, and TensorFlow
    gcloud compute instances create "${VM_NAME}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
        --image-family="tf-latest-gpu" \
        --image-project="deeplearning-platform-release" \
        --maintenance-policy="TERMINATE" \
        --restart-on-failure \
        --boot-disk-size="${DISK_SIZE}" \
        --boot-disk-type="pd-ssd" \
        --metadata="install-nvidia-driver=True" \
        --scopes="storage-full,logging-write,monitoring" \
        --tags="agile-autonomy"

    echo "VM '${VM_NAME}' created. Waiting 60 s for boot ..."
    sleep 60

    # -----------------------------------------------------------------------
    # Build the remote setup script and pipe it via SSH
    # -----------------------------------------------------------------------
    section "Running remote setup on ${VM_NAME}"

    gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" \
        --command="echo 'SSH connection successful'"

    # Upload the cloud training config to the VM
    gcloud compute scp \
        "$(dirname "$0")/planner_learning/config/train_settings_cloud.yaml" \
        "${VM_NAME}:~/train_settings_cloud.yaml" \
        --zone="${ZONE}"

    # Execute setup + training remotely
    gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" -- bash -s <<REMOTE_SCRIPT
set -euo pipefail

echo ">>> System info"
nvidia-smi
python3 --version

echo ">>> Cloning repository"
if [ ! -d ~/agile_autonomy ]; then
    git clone ${REPO_URL} ~/agile_autonomy
else
    git -C ~/agile_autonomy pull
fi

echo ">>> Installing Python dependencies"
pip install -q --upgrade pip
pip install -q open3d pyquaternion opencv-python-headless scipy pandas tqdm pyyaml

# TF is pre-installed on Deep Learning VMs; upgrade only if needed
python3 -c "import tensorflow as tf; print('TF version:', tf.__version__)"

echo ">>> Downloading dataset (~6 GB)"
DATASET_ARCHIVE=~/agile_autonomy_dataset.tar.xz
DATASET_DIR=~/agile_autonomy_data

if [ ! -d "\${DATASET_DIR}/train" ]; then
    if [ ! -f "\${DATASET_ARCHIVE}" ]; then
        wget -q --show-progress -O "\${DATASET_ARCHIVE}" "${DATASET_URL}"
    fi
    mkdir -p "\${DATASET_DIR}"
    echo "Extracting ..."
    tar -xf "\${DATASET_ARCHIVE}" -C "\${DATASET_DIR}" --strip-components=1
    echo "Dataset ready at \${DATASET_DIR}"
else
    echo "Dataset already extracted at \${DATASET_DIR}"
fi

echo ">>> Configuring training paths"
cp ~/train_settings_cloud.yaml ~/agile_autonomy/planner_learning/config/train_settings_cloud.yaml
python3 - <<'PYEOF'
import yaml, os

cfg_path = os.path.expanduser('~/agile_autonomy/planner_learning/config/train_settings_cloud.yaml')
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

home = os.path.expanduser('~')
cfg['log_dir']              = os.path.join(home, 'agile_autonomy_checkpoints')
cfg['train']['train_dir']   = os.path.join(home, 'agile_autonomy_data', 'train')
cfg['train']['val_dir']     = os.path.join(home, 'agile_autonomy_data', 'val')

os.makedirs(cfg['log_dir'], exist_ok=True)

with open(cfg_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

print('Config updated:')
print(yaml.dump(cfg, default_flow_style=False))
PYEOF

echo ">>> Starting training (output streamed below)"
cd ~/agile_autonomy/planner_learning
nohup python3 train.py \
    --settings_file=config/train_settings_cloud.yaml \
    > ~/training.log 2>&1 &
TRAIN_PID=\$!
echo "Training running as PID \${TRAIN_PID} — logs at ~/training.log"
tail -f ~/training.log &
wait \${TRAIN_PID}
echo "Training complete."
REMOTE_SCRIPT

    # -----------------------------------------------------------------------
    # Sync checkpoints to GCS
    # -----------------------------------------------------------------------
    section "Syncing checkpoints to GCS"
    gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" \
        --command="gsutil -m rsync -r ~/agile_autonomy_checkpoints ${BUCKET}/checkpoints/"
    echo "Checkpoints synced to ${BUCKET}/checkpoints/"

    # -----------------------------------------------------------------------
    # Optional: delete the VM after training to save cost
    # -----------------------------------------------------------------------
    section "Training complete"
    echo "To delete the VM and stop billing, run:"
    echo "  gcloud compute instances delete ${VM_NAME} --zone=${ZONE} --quiet"
    echo ""
    echo "To download checkpoints locally:"
    echo "  gsutil -m rsync -r ${BUCKET}/checkpoints/ ./checkpoints/"

# ===========================================================================
# MODE: Vertex AI Custom Training Job
# ===========================================================================
elif [ "${MODE}" = "vertex" ]; then

    section "Submitting Vertex AI Custom Training Job"

    JOB_NAME="agile-autonomy-train-$(date +%Y%m%d%H%M%S)"
    DATA_GCS="${BUCKET}/dataset"
    CKPT_GCS="${BUCKET}/checkpoints/${JOB_NAME}"

    # -----------------------------------------------------------------------
    # Upload dataset to GCS (first time only — ~20 GB extracted)
    # -----------------------------------------------------------------------
    section "Uploading dataset to GCS: ${DATA_GCS}"
    if ! gsutil ls "${DATA_GCS}/train" &>/dev/null; then
        echo "Downloading dataset locally then uploading to GCS ..."
        TMPDIR=$(mktemp -d)
        ARCHIVE="${TMPDIR}/agile_autonomy_dataset.tar.xz"
        wget -q --show-progress -O "${ARCHIVE}" "${DATASET_URL}"
        mkdir -p "${TMPDIR}/dataset"
        tar -xf "${ARCHIVE}" -C "${TMPDIR}/dataset" --strip-components=1
        gsutil -m rsync -r "${TMPDIR}/dataset/" "${DATA_GCS}/"
        rm -rf "${TMPDIR}"
        echo "Dataset uploaded to ${DATA_GCS}"
    else
        echo "Dataset already on GCS at ${DATA_GCS}"
    fi

    # -----------------------------------------------------------------------
    # Write a self-contained training entrypoint for Vertex AI
    # -----------------------------------------------------------------------
    VERTEX_SCRIPT_DIR=$(mktemp -d)
    cat >"${VERTEX_SCRIPT_DIR}/vertex_train.py" <<'VERTEX_PY'
#!/usr/bin/env python3
"""
Vertex AI training entrypoint.
Environment variables injected by Vertex AI:
  AIP_TRAINING_DATA_URI   — GCS path to training data
  AIP_MODEL_DIR           — GCS path for output model/checkpoints
"""
import os, subprocess, sys

# 1. Install extra dependencies not present in the base container
DEPS = ['open3d', 'pyquaternion', 'opencv-python-headless', 'scipy', 'tqdm']
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + DEPS, check=True)

import yaml

# 2. Download dataset from GCS to local disk
DATA_URI  = os.environ.get('AIP_TRAINING_DATA_URI', '')
MODEL_DIR = os.environ.get('AIP_MODEL_DIR', '/tmp/output')
LOCAL_DATA = '/tmp/agile_autonomy_data'
LOCAL_REPO = '/tmp/agile_autonomy'

os.makedirs(LOCAL_DATA, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if DATA_URI:
    print(f'Downloading dataset from {DATA_URI} ...')
    subprocess.run(['gsutil', '-m', 'rsync', '-r', DATA_URI + '/', LOCAL_DATA + '/'], check=True)
    DATA_TRAIN = os.path.join(LOCAL_DATA, 'train')
    DATA_VAL   = os.path.join(LOCAL_DATA, 'val')
else:
    raise EnvironmentError('AIP_TRAINING_DATA_URI not set')

# 3. Clone repo
if not os.path.isdir(LOCAL_REPO):
    subprocess.run(['git', 'clone', 'https://github.com/rayeed221/agile_autonomy.git', LOCAL_REPO], check=True)

PLANNER_DIR = os.path.join(LOCAL_REPO, 'planner_learning')
CFG_PATH    = os.path.join(PLANNER_DIR, 'config', 'train_settings_vertex.yaml')

# 4. Load base config and override paths
BASE_CFG = os.path.join(PLANNER_DIR, 'config', 'train_settings.yaml')
with open(BASE_CFG) as f:
    cfg = yaml.safe_load(f)

cfg['log_dir']              = MODEL_DIR
cfg['train']['train_dir']   = DATA_TRAIN
cfg['train']['val_dir']     = DATA_VAL
cfg['train']['batch_size']  = int(os.environ.get('BATCH_SIZE', '16'))
cfg['train']['max_training_epochs'] = int(os.environ.get('EPOCHS', '100'))

with open(CFG_PATH, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

# 5. Run training
os.chdir(PLANNER_DIR)
subprocess.run([sys.executable, 'train.py', f'--settings_file={CFG_PATH}'], check=True)

# 6. Sync checkpoints back to GCS model dir (Vertex does this automatically
#    when AIP_MODEL_DIR is a gs:// path, but we sync explicitly for safety)
if MODEL_DIR.startswith('gs://'):
    pass  # Vertex handles upload
else:
    gcs_dest = os.environ.get('AIP_MODEL_DIR', '')
    if gcs_dest:
        subprocess.run(['gsutil', '-m', 'rsync', '-r', MODEL_DIR + '/', gcs_dest + '/'])

print('Vertex AI training job complete.')
VERTEX_PY

    # -----------------------------------------------------------------------
    # Submit the Vertex AI custom job
    # -----------------------------------------------------------------------
    # Uses a pre-built TF GPU container (no Dockerfile needed)
    CONTAINER_IMAGE="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest"

    gcloud ai custom-jobs create \
        --region="${REGION}" \
        --display-name="${JOB_NAME}" \
        --worker-pool-spec="\
machine-type=${MACHINE_TYPE},\
accelerator-type=$(echo "${GPU_TYPE}" | tr '[:lower:]-' '[:upper:]_' | sed 's/NVIDIA_//'),\
accelerator-count=${GPU_COUNT},\
container-image-uri=${CONTAINER_IMAGE},\
local-package-path=${VERTEX_SCRIPT_DIR},\
python-module=vertex_train" \
        --args="--" \
        --env-vars="\
AIP_TRAINING_DATA_URI=${DATA_GCS},\
AIP_MODEL_DIR=${CKPT_GCS},\
BATCH_SIZE=16,\
EPOCHS=100"

    echo ""
    echo "Vertex AI job '${JOB_NAME}' submitted."
    echo "Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
    echo "Checkpoints will be saved to: ${CKPT_GCS}"
    echo ""
    echo "To stream logs:"
    echo "  gcloud ai custom-jobs stream-logs \$(gcloud ai custom-jobs list --region=${REGION} --format='value(name)' | head -1) --region=${REGION}"
    echo ""
    echo "To download checkpoints after the job completes:"
    echo "  gsutil -m rsync -r ${CKPT_GCS}/ ./checkpoints/"

    rm -rf "${VERTEX_SCRIPT_DIR}"

else
    echo "ERROR: Unknown MODE '${MODE}'. Set MODE=gce or MODE=vertex"
    exit 1
fi
