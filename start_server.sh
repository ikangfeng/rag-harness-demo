#!/bin/bash
# 启动 Gradio 可观测前端
# 用法: bash start_server.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
export RAG_VECTORSTORE_DIR="/Users/feng/Documents/trae_projects/zjj/project2_vectorstore"

cd "$DIR"
echo "🚀 启动工业级 RAG 演示前端..."
echo "📍 地址: http://localhost:7860"
echo ""

python3 gradio_server.py
