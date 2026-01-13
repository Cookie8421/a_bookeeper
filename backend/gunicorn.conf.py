bind = "0.0.0.0:8911"  # 监听所有地址，方便容器端口映射和外部访问
workers = 2
worker_class = "sync"  # 或 "gevent" 如果需要异步
timeout = 180  # 增加超时时间以支持长时间运行的 LLM 请求（平均 40 秒，设置为 3 分钟提供缓冲）
preload_app = False  # 设为 False，让每个 worker 独立初始化 Ollama 连接，避免 fork 后连接失效

# 日志 - 输出到 stdout/stderr 以便在 docker-compose logs 中查看
accesslog = "-"  # stdout
errorlog = "-"   # stderr (包含所有 print() 输出)
loglevel = "info"