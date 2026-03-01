#!/bin/bash
set -e

echo "=========================================="
echo "Qwen2.5 32B 专家模型部署脚本"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

show_usage() {
    echo "用法: $0 <command> [options]"
    echo ""
    echo "命令:"
    echo "  setup         环境准备"
    echo "  download      下载模型"
    echo "  deploy-a      部署专家A服务"
    echo "  deploy-b      部署专家B服务"
    echo "  deploy-all    部署专家A和专家B服务"
    echo "  stop          停止所有服务"
    echo "  status        查看服务状态"
    echo "  test          测试专家服务"
    echo "  help          显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 setup"
    echo "  $0 download"
    echo "  $0 deploy-a"
    echo "  $0 deploy-b --port 8001"
    echo "  $0 deploy-all"
    echo "  $0 stop"
}

run_setup() {
    echo "=========================================="
    echo "步骤 1/3: 环境准备"
    echo "=========================================="
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate qwen_experts
    
    echo "安装vLLM及相关依赖..."
    pip install --upgrade pip
    pip install vllm fastapi uvicorn pydantic
    pip install transformers accelerate peft bitsandbytes
    pip install unsloth
    
    echo "环境准备完成！"
}

run_download() {
    echo "=========================================="
    echo "步骤 2/3: 下载模型"
    echo "=========================================="
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate qwen_experts
    
    MODELS_DIR="$BASE_DIR/models"
    mkdir -p "$MODELS_DIR"
    
    echo "下载专家A模型..."
    if [ ! -d "$MODELS_DIR/qwen25-32b-expert-a-qlora" ]; then
        modelscope download --model GaryLeenene/qwen25-32b-expert-a-qlora --local_dir "$MODELS_DIR/qwen25-32b-expert-a-qlora"
    else
        echo "专家A模型已存在"
    fi
    
    echo "下载专家B模型..."
    if [ ! -d "$MODELS_DIR/qwen25-32b-expert-b-qlora" ]; then
        modelscope download --model GaryLeenene/qwen25-32b-expert-b-qlora --local_dir "$MODELS_DIR/qwen25-32b-expert-b-qlora"
    else
        echo "专家B模型已存在"
    fi
    
    echo "下载Qwen2.5 32B基础模型 (4bit AWQ)..."
    if [ ! -d "$MODELS_DIR/qwen25-32b-awq" ]; then
        modelscope download --model qwen/Qwen2.5-32B-Instruct-AWQ --local_dir "$MODELS_DIR/qwen25-32b-awq"
    else
        echo "基础模型已存在"
    fi
    
    echo "模型下载完成！"
}

run_deploy() {
    local expert_id=$1
    local port=${2:-8000}
    local log_file="$BASE_DIR/logs/expert_${expert_id}.log"
    local pid_file="$BASE_DIR/logs/expert_${expert_id}.pid"
    
    echo "=========================================="
    echo "步骤 3/3: 部署专家${expert_id}服务"
    echo "=========================================="
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate qwen_experts
    
    if [ -f "$pid_file" ]; then
        local old_pid=$(cat "$pid_file")
        if kill -0 $old_pid 2>/dev/null; then
            echo "专家${expert_id}服务已在运行 (PID: $old_pid)"
            exit 0
        fi
    fi
    
    nohup python "$SCRIPT_DIR/deploy_experts.py" --port $port > "$log_file" 2>&1 &
    local new_pid=$!
    echo $new_pid > "$pid_file"
    
    echo "专家${expert_id}服务已启动 (PID: $new_pid)"
    echo "等待服务启动..."
    sleep 8
    
    if curl -s http://localhost:$port/health > /dev/null; then
        echo "专家${expert_id}服务启动成功！"
        echo "服务地址: http://localhost:$port"
        echo "API接口: http://localhost:$port/generate"
        echo "日志文件: $log_file"
    else
        echo "专家${expert_id}服务启动失败，请查看日志: $log_file"
        rm -f "$pid_file"
        exit 1
    fi
}

run_stop() {
    echo "=========================================="
    echo "停止服务"
    echo "=========================================="
    
    for expert_id in A B; do
        local pid_file="$BASE_DIR/logs/expert_${expert_id}.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 $pid 2>/dev/null; then
                kill $pid 2>/dev/null || true
                echo "专家${expert_id}服务已停止"
            else
                echo "专家${expert_id}服务未运行"
            fi
            rm -f "$pid_file"
        else
            echo "专家${expert_id}服务未运行"
        fi
    done
}

run_status() {
    echo "=========================================="
    echo "服务状态"
    echo "=========================================="
    
    for expert_id in A B; do
        local pid_file="$BASE_DIR/logs/expert_${expert_id}.pid"
        local port=$((8000 + $(echo $expert_id | tr 'A-B' '0-1')))
        
        echo ""
        echo "专家${expert_id} (端口: $port):"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 $pid 2>/dev/null; then
                echo "  状态: 运行中 (PID: $pid)"
                curl -s http://localhost:$port/health 2>/dev/null || echo "  无法连接到服务"
            else
                echo "  状态: 已停止"
            fi
        else
            echo "  状态: 未运行"
        fi
    done
}

run_test() {
    local expert_id=${1:-A}
    local port=$((8000 + $(echo $expert_id | tr 'A-B' '0-1')))
    
    echo "=========================================="
    echo "测试专家${expert_id}服务"
    echo "=========================================="
    
    local prompt="请介绍一下Qwen2.5模型的特点"
    
    echo "请求: $prompt"
    echo ""
    
    local response=$(curl -s -X POST "http://localhost:$port/generate" \
        -H "Content-Type: application/json" \
        -d "{\"expert_id\": \"$expert_id\", \"prompt\": \"$prompt\"}")
    
    echo "响应: $response"
}

case "${1:-help}" in
    setup)
        run_setup
        ;;
    download)
        run_download
        ;;
    deploy-a)
        run_deploy "A" "${2:-8000}"
        ;;
    deploy-b)
        run_deploy "B" "${2:-8001}"
        ;;
    deploy-all)
        run_deploy "A" 8000
        run_deploy "B" 8001
        ;;
    stop)
        run_stop
        ;;
    status)
        run_status
        ;;
    test)
        run_test "${2:-A}" "${3:-8000}"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "未知命令: $1"
        show_usage
        exit 1
        ;;
esac
