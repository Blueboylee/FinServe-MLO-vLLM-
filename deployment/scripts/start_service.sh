#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Qwen2.5 32B 专家模型服务"
echo "=========================================="

show_usage() {
    echo "用法: $0 <command> [options]"
    echo ""
    echo "命令:"
    echo "  deploy-a      部署专家A服务"
    echo "  deploy-b      部署专家B服务"
    echo "  deploy-all    部署专家A和专家B服务"
    echo "  stop          停止所有服务"
    echo "  status        查看服务状态"
    echo "  test          测试专家服务"
    echo ""
    echo "示例:"
    echo "  $0 deploy-a"
    echo "  $0 deploy-b --port 8001"
    echo "  $0 deploy-all"
    echo "  $0 stop"
}

deploy_expert() {
    local expert_id=$1
    local port=${2:-8000}
    local log_file="$BASE_DIR/logs/expert_${expert_id}.log"
    local pid_file="$BASE_DIR/logs/expert_${expert_id}.pid"
    
    echo "=========================================="
    echo "启动专家${expert_id}服务 (端口: $port)"
    echo "日志文件: $log_file"
    echo "=========================================="
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate qwen_experts
    
    nohup python "$SCRIPT_DIR/deploy_experts.py" --port $port > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    
    echo "专家${expert_id}服务已启动 (PID: $!)"
    echo "等待服务启动..."
    sleep 5
    
    if curl -s http://localhost:$port/health > /dev/null; then
        echo "专家${expert_id}服务启动成功！"
        echo "服务地址: http://localhost:$port"
        echo "API接口: http://localhost:$port/generate"
    else
        echo "专家${expert_id}服务启动失败，请查看日志: $log_file"
        exit 1
    fi
}

stop_expert() {
    local expert_id=$1
    local pid_file="$BASE_DIR/logs/expert_${expert_id}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        kill $pid 2>/dev/null || true
        rm -f "$pid_file"
        echo "专家${expert_id}服务已停止"
    else
        echo "专家${expert_id}服务未运行"
    fi
}

deploy_all() {
    echo "=========================================="
    echo "部署所有专家服务"
    echo "=========================================="
    
    deploy_expert "A" 8000 &
    deploy_expert "B" 8001 &
    
    wait
    
    echo "所有专家服务已启动"
    echo "专家A: http://localhost:8000"
    echo "专家B: http://localhost:8001"
}

test_service() {
    local expert_id=${1:-A}
    local port=${2:-8000}
    
    echo "=========================================="
    echo "测试专家${expert_id}服务 (端口: $port)"
    echo "=========================================="
    
    local prompt="请介绍一下Qwen2.5模型的特点"
    
    local response=$(curl -s -X POST "http://localhost:$port/generate" \
        -H "Content-Type: application/json" \
        -d "{\"expert_id\": \"$expert_id\", \"prompt\": \"$prompt\"}")
    
    echo "请求: $prompt"
    echo "响应: $response"
}

case "${1:-}" in
    deploy-a)
        deploy_expert "A" "${2:-8000}"
        ;;
    deploy-b)
        deploy_expert "B" "${2:-8001}"
        ;;
    deploy-all)
        deploy_all
        ;;
    stop)
        stop_expert "A"
        stop_expert "B"
        ;;
    status)
        echo "专家A服务状态:"
        if [ -f "$BASE_DIR/logs/expert_A.pid" ]; then
            pid=$(cat "$BASE_DIR/logs/expert_A.pid")
            if kill -0 $pid 2>/dev/null; then
                echo "  运行中 (PID: $pid)"
                curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "  无法连接到服务"
            else
                echo "  已停止"
            fi
        else
            echo "  未运行"
        fi
        
        echo ""
        echo "专家B服务状态:"
        if [ -f "$BASE_DIR/logs/expert_B.pid" ]; then
            pid=$(cat "$BASE_DIR/logs/expert_B.pid")
            if kill -0 $pid 2>/dev/null; then
                echo "  运行中 (PID: $pid)"
                curl -s http://localhost:8001/health | python -m json.tool 2>/dev/null || echo "  无法连接到服务"
            else
                echo "  已停止"
            fi
        else
            echo "  未运行"
        fi
        ;;
    test)
        test_service "${2:-A}" "${3:-8000}"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
