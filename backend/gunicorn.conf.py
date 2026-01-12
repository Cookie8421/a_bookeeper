bind = "0.0.0.0:8911"  # 监听所有地址，方便容器端口映射和外部访问
workers = 2
worker_class = "sync"  # 或 "gevent" 如果需要异步
timeout = 120
preload_app = True

# 日志
accesslog = "/app/backend/logs/access.log"
errorlog = "/app/backend/logs/error.log"
loglevel = "info"