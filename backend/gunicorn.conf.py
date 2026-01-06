bind = "127.0.0.1:8911"  # 只监听本地
workers = 2
worker_class = "sync"  # 或 "gevent" 如果需要异步
timeout = 120
preload_app = True

# 日志
accesslog = "/app/backend/logs/access.log"
errorlog = "/app/backend/logs/error.log"
loglevel = "info"