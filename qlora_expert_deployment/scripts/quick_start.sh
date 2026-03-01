#!/bin/bash
# Quick Start Script for Qwen2.5-32B Expert Models
# Complete deployment workflow

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================"
echo "Qwen2.5-32B Expert Models Quick Start"
echo "======================================"
echo ""

# Activate conda environment
echo "[1/5] Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen-expert-env

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    echo "Please run: bash scripts/setup_conda_env.sh"
    exit 1
fi

echo "✓ Conda environment activated"
echo ""

# Run health check
echo "[2/5] Running health check..."
cd "$PROJECT_ROOT"
python scripts/health_check.py --config configs/deployment_config.json

if [ $? -ne 0 ]; then
    echo "Warning: Health check failed. Please review the errors above."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Download models if not present
echo "[3/5] Checking model files..."
if [ ! -d "$PROJECT_ROOT/models/base_model" ] || \
   [ ! -d "$PROJECT_ROOT/models/expert_a" ] || \
   [ ! -d "$PROJECT_ROOT/models/expert_b" ]; then
    echo "Models not found. Downloading from ModelScope..."
    python scripts/deploy_experts.py --config configs/deployment_config.json --download
else
    echo "✓ Model files found"
fi

echo ""

# Start deployment
echo "[4/5] Starting expert model deployment..."
echo "Choose deployment mode:"
echo "1) Interactive Python deployment"
echo "2) REST API server"
echo "3) Batch inference"
read -p "Enter choice (1-3): " mode

case $mode in
    1)
        echo "Starting interactive deployment..."
        python scripts/deploy_experts.py \
            --config configs/deployment_config.json \
            --experts expert_a expert_b \
            --validate
        ;;
    2)
        echo "Starting REST API server..."
        python scripts/expert_server.py \
            --config configs/deployment_config.json \
            --host 0.0.0.0 \
            --port 8000
        ;;
    3)
        echo "Batch inference mode selected"
        read -p "Enter input file path: " input_file
        python scripts/batch_inference.py \
            --config configs/deployment_config.json \
            --input "$input_file"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "[5/5] Deployment completed!"
echo ""
echo "======================================"
echo "Quick Reference:"
echo "======================================"
echo "Health Check:  python scripts/health_check.py"
echo "Download:      python scripts/deploy_experts.py --download"
echo "Deploy:        python scripts/deploy_experts.py"
echo "API Server:    python scripts/expert_server.py"
echo "======================================"
