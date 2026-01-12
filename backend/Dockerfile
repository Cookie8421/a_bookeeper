FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# 复制前端文件
COPY frontend/ .

# 安装依赖并构建
RUN npm install --legacy-peer-deps
RUN npm run build

# 使用 Python 作为基础镜像
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# 复制 Nginx 配置
COPY nginx/nginx.conf /etc/nginx/nginx.conf

# 复制后端文件
COPY backend/requirements.txt /app/backend/requirements.txt
COPY backend/rag_ollama_faiss.py /app/backend/rag_ollama_faiss.py
COPY backend/gunicorn.conf.py /app/backend/gunicorn.conf.py
COPY backend /app/backend

# 安装 Python 依赖
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt

# 复制前端构建文件
COPY --from=frontend-builder /app/frontend/build /usr/share/nginx/html

# 创建数据和日志目录
RUN mkdir -p /app/data /app/backend/logs
# 确保 Nginx 目录存在（某些精简镜像默认不创建缓存目录）
RUN mkdir -p /var/cache/nginx /var/log/nginx /var/lib/nginx/body /run/nginx

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app /usr/share/nginx/html /var/log/nginx /var/cache/nginx /var/lib/nginx /run/nginx

# 启动脚本（以 root 设置可执行权限）
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

USER appuser

# 暴露端口
EXPOSE 8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]