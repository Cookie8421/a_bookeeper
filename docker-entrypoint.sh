#!/bin/bash

# 启动 Flask 后端（后台运行）
cd /app/backend
gunicorn --config gunicorn.conf.py rag_ollama_faiss:app &

# 启动 Nginx 前端代理
nginx -g "daemon off;"