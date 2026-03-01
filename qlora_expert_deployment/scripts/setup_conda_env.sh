#!/bin/bash
# Conda Environment Setup Script for Qwen2.5-32B QLoRA Expert Deployment
# Compatible with Python 3.10, Ubuntu 22.04, and NVIDIA 4080S (32GB)

set -e

echo "======================================"
echo "Qwen2.5-32B QLoRA Environment Setup"
echo "======================================"

# Configuration
ENV_NAME="qwen-expert-env"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    log_error "Conda is not installed. Please install Miniconda/Anaconda first."
    exit 1
fi

log_info "Conda found: $(which conda)"

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    log_warn "Environment ${ENV_NAME} already exists. Removing..."
    conda env remove -n ${ENV_NAME} -y
fi

# Create new conda environment
log_info "Creating conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
log_info "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
log_info "Installing PyTorch with CUDA ${CUDA_VERSION}..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
log_info "Installing vLLM..."
pip install vllm==0.4.0

# Install transformers and related libraries
log_info "Installing transformers and dependencies..."
pip install transformers==4.37.0 \
    accelerate==0.26.0 \
    peft==0.8.0 \
    auto-gptq==0.6.0 \
    optimum==1.16.0

# Install ModelScope
log_info "Installing ModelScope..."
pip install modelscope==1.14.0

# Install additional utilities
log_info "Installing additional utilities..."
pip install sentencepiece \
    protobuf \
    einops \
    tiktoken \
    scipy \
    numpy==1.24.3

# Install flash-attn (optional but recommended for better performance)
log_info "Installing flash-attn (this may take a while)..."
pip install flash-attn==2.5.0 --no-build-isolation || log_warn "flash-attn installation failed, skipping..."

# Verify installations
log_info "Verifying installations..."

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import modelscope; print(f'ModelScope: {modelscope.__version__}')"

# Check CUDA availability
log_info "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Create necessary directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info "Creating directory structure..."
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/configs"

log_info "Environment setup completed successfully!"
echo ""
echo "======================================"
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To download models, run:"
echo "  python scripts/deploy_experts.py --download"
echo ""
echo "To start deployment, run:"
echo "  python scripts/deploy_experts.py --experts expert_a expert_b --validate"
echo "======================================"
