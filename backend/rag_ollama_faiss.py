# rag_ollama_faiss.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone
import os
import threading
import re
import faiss
import portalocker
import numpy as np
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# ==================== 配置加载 ====================
app = Flask(__name__)
# CORS(app, resources={
#     r"/api/*": {"origins": ["http://localhost:3000"]},  # 只允许 React 前端
#     r"/add": {"origins": ["http://localhost:3000"]},
#     r"/query": {"origins": ["http://localhost:3000"]},
#     r"/summary": {"origins": ["http://localhost:3000"]},
#     r"/health": {"origins": ["http://localhost:3000"]},
# })

DATA_DIR = "myRAG/data"
os.makedirs(DATA_DIR, exist_ok=True)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EXPENSES_FILE = os.path.join(DATA_DIR, "expense.txt")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")          # ← 索引文件
PROGRESS_FILE = os.path.join(DATA_DIR, "embedding_progress.txt")  # ← 进度文件

print("EXPENSES_FILE:" , EXPENSES_FILE)
print("INDEX_FILE:"    , INDEX_FILE)
print("PROGRESS_FILE:" , PROGRESS_FILE)
# ================ 1. 初始化 Ollama 组件 ================
# 检查 Ollama 是否运行
try:
    import requests
    assert requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2).status_code == 200
except:
    raise RuntimeError("❌ 请先启动 Ollama：ollama serve")

# Embedding 模型（nomic-embed-text，768维）
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL,
)

# LLM 模型（qwen:4b，中文优化）
llm = OllamaLLM(
    model="qwen3:4b",
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
    stop=["\n\n"],  # ← 关键！防过早
    keep_alive="5m",    # 保持模型加载
    stream=False,       # 强制非流式
)

def add_expense(raw_text: str):

    # 常见日期格式正则表达式
    date_patterns = [
        r'(\d{1,2}月\d{1,2}日)',            # 3月11日
        r'(\d{4}年\d{1,2}月\d{1,2}日)',     # 2024年3月11日
        r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # 2024-03-11 或 2024/03/11
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # 03-11-2024 或 03/11/2024
    ]
    
    # 查找日期
    found_date = None
    for pattern in date_patterns:
        match = re.search(pattern, raw_text)
        if match:
            date_str = match.group(1)
            try:
                # 尝试解析日期
                if '年' in date_str:
                    date = datetime.strptime(date_str, "%Y年%m月%d日")
                elif '月' in date_str:
                    # 处理"3月11日"格式
                    current_year = datetime.now().year
                    date = datetime.strptime(f"{current_year}年{date_str}", "%Y年%m月%d日")
                elif '/' in date_str or '-' in date_str:
                    parts = re.split('[-/]', date_str)
                    if len(parts[0]) == 4:  # 2024-03-11
                        date = datetime.strptime(date_str, "%Y-%m-%d")
                    else:  # 03-11-2024
                        date = datetime.strptime(date_str, "%m-%d-%Y")
                else:
                    continue
                
                # 转换为带时区的datetime对象
                found_date = date.replace(tzinfo=timezone.utc).astimezone()
                # 从原始文本中移除日期部分
                raw_text = raw_text.replace(date_str, "").strip()
                break
            except ValueError as e:
                print(f"日期解析错误: {e}")
                continue

    # 生成 ISO 8601 时间戳（带时区）
    now = found_date if found_date else datetime.now(timezone.utc).astimezone()  # 本地时区
    timestamp = now.isoformat(timespec='seconds')  # e.g., 2025-12-25T08:30:12+08:00
    line = f"{timestamp} | {raw_text.strip()}\n"
    
    # 原子追加到 expenses.txt
    try:
        with open(EXPENSES_FILE, "a", encoding="utf-8") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        print(f"✅ 已记账: {line.strip()}")
        return True
    except IOError as e:
        print(f"❌ 写入文件失败: {e}")
        return False

# ==================== 加载/创建 FAISS 索引 ====================
def load_or_create_index():
    dim = 768
    # 情况 1: 索引文件存在 → 直接加载
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        print(f"✅ 加载 FAISS 索引，条目数: {index.ntotal}")
        return index

    # 情况 2: 索引文件不存在 → 检查账本是否为空
    if not os.path.exists(EXPENSES_FILE):
        open(EXPENSES_FILE, "w", encoding="utf-8").close()
    
    with open(EXPENSES_FILE, "r", encoding="utf-8") as f:
        all_lines = [line for line in f if line.strip() and not line.startswith("#")]
    
    if not all_lines:
        # 账本也为空 → 新建空索引   
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        faiss.normalize_L2(np.zeros((1, dim), dtype="float32"))
        print("🆕 创建空 FAISS 索引（首次启动）")
        return index

    # 情况 3: 账本有数据但索引丢失 → 自动重建
    print(f"⚠️ 索引文件缺失，但账本有 {len(all_lines)} 条记录，正在重建...")
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    faiss.normalize_L2(np.zeros((1, dim), dtype="float32"))
    
    # 从头开始处理所有账单（模拟首次全量）
    texts, ids = [], []
    for i, line in enumerate(all_lines, start=1):
        try:
            parts = line.strip().split("|", 1)
            if len(parts) >= 2:
                texts.append(parts[1].strip())
                ids.append(i)
        except Exception as e:
            print(f"⚠️ 跳过无效行 {i}: {e}")
            continue

    if texts:
        vectors = np.array(embedding_model.embed_documents(texts)).astype("float32")
        faiss.normalize_L2(vectors)
        ids_arr = np.array(ids, dtype=np.int64)
        index.add_with_ids(vectors, ids_arr)
        print(f"✅ 重建索引成功，已处理 {len(ids)} 条账单")
    else:
        print("⚠️ 账本中无有效文本，创建空索引")
    
    # 保存重建后的索引
    faiss.write_index(index, INDEX_FILE)
    save_progress(len(all_lines), "nomic-embed-text@20251229")  # 更新进度
    return index

# ==================== 进度管理 ====================
def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return 0, "nomic-embed-text@20251229"
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        last_line = int(lines[0].split("=")[1]) if len(lines) > 0 else 0
        version = lines[1].split("=")[1] if len(lines) > 1 else "unknown"
    return last_line, version

def save_progress(last_line: int, version: str):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write(f"last_processed_line = {last_line}\n")
        f.write(f"embedding_model_version = {version}\n")

def update_embeddings_incremental():
    # 1. 加载索引 & 进度
    index = load_or_create_index()  # 用你前面定义的函数
    last_line, current_version = load_progress()
    
    # 读取所有账单行
    if not os.path.exists(EXPENSES_FILE):
        open(EXPENSES_FILE, "w", encoding="utf-8").close()
    with open(EXPENSES_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    new_lines = lines[last_line:]  # 从 last_line 开始（0-indexed → line 1 是 index 0）
    
    if not new_lines:
        print("⏭️ 无新增账单")
        return
    
    print(f"🔄 发现 {len(new_lines)} 条新账单（行 {last_line+1} ~ {total_lines}）")

    # 3. 提取 raw_text 并生成 embedding
    texts, new_ids = [], []
    for i, line in enumerate(new_lines, start=last_line+1):
        if line.strip() and not line.startswith("#"):
            try:
                # 解析： "timestamp | text" → 取 | 后部分
                parts = line.strip().split("|", 1)
                if len(parts) >= 2:
                    raw_text = parts[1].strip()
                    texts.append(raw_text)
                    new_ids.append(i)  # 行号作为 ID
            except Exception as e:
                print(f"⚠️ 跳过无效行 {i}: {e}")
                continue
    
    if not texts:
        print("⏭️ 无有效新账单")
        return

    vectors = np.array(embedding_model.embed_documents(texts)).astype("float32")
    faiss.normalize_L2(vectors)

    # 4. 增量加入 FAISS
    ids_arr = np.array(new_ids, dtype=np.int64)
    index.add_with_ids(vectors, ids_arr)

    # 5. 持久化
    faiss.write_index(index, INDEX_FILE)
    save_progress(total_lines, current_version)

    print(f"✅ 新增 {len(new_ids)} 条 embedding，索引已保存")

# ==================== 按 ID 获取原文（替代 chunks） ====================
def get_context_by_ids(ids: list) -> str:
    """根据行号列表，从 expenses.txt 读取原文"""
    if not ids:
        return ""
    with open(EXPENSES_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    context_parts = []
    for i, line_id in enumerate(ids):
        if 1 <= line_id <= len(lines):
            line = lines[line_id - 1].strip()
            if line and not line.startswith("#"):
                # 提取 raw_text 部分
                parts = line.split("|", 1)
                text = parts[1].strip() if len(parts) > 1 else line
                context_parts.append(f"[{i+1}] {text} （{parts[0].strip()}）")
    return "\n".join(context_parts)

# ================ RAG 推理函数 ================
def rag_query(question: str, k: int = 3) -> str:
    index = load_or_create_index()
    if index.ntotal == 0:
        return "📦 当前无账单，请先记账。"
    
    # 1. 生成 query embedding
    query_vec = embedding_model.embed_query(question)
    print("维度数应为 768：", len(query_vec))
    query_vec = np.array(query_vec).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vec)

    # 2. 检索 Top-K
    scores, indices = index.search(query_vec, k * 5)
    print("Top scores:", scores)
    print("Top indices:", indices)
    
    # 3. 拼接上下文
    valid_ids = [int(idx) for idx in indices[0] if idx != -1]
    print("有效 ID:", valid_ids)
    context = get_context_by_ids(valid_ids)
    if not context:
        return "🔍 未找到相关账单。"

    # 4. 构造 prompt & 生成
    prompt = f"""你是一名资深个人财务管理师，今天是 {datetime.now().strftime("%Y年%m月%d日")}，请结合当前日期和上下文回答专业、简洁地回答问题。
若上下文不足，请回答“根据现有资料无法回答”。

上下文：
{context}

问题：{question}
回答："""
    
    print("Prompt:", prompt)
    return llm.invoke(prompt,options={"num_predict": 8192})

# ==================== Flask API 路由 ====================

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        # 检查 Ollama 是否可达
        assert requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2).status_code == 200
        # 检查 FAISS 索引是否加载
        load_or_create_index()
        return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
    
@app.route('/add', methods=['POST'])
def add_expense_api():
    """添加账单接口"""
    data = request.json
    raw_text = data.get('text', '').strip()
    
    if not raw_text:
        return jsonify({'error': 'text is required'}), 400
    
    success = add_expense(raw_text)
    if not success:
        return jsonify({'error': 'failed to write to file'}), 500

    # 异步更新 embedding（避免阻塞 API）
    thread = threading.Thread(target=update_embeddings_incremental)
    thread.start()

    return jsonify({
        'message': 'expense added successfully',
        'timestamp': datetime.now().isoformat(),
        'text': raw_text
    }), 200

@app.route('/query', methods=['GET'])
def query_api():
    """语义查询接口"""
    question = request.args.get('q', '').strip()
    if not question:
        return jsonify({'error': 'q parameter is required'}), 400

    try:
        answer = rag_query(question)
        return jsonify({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': f'query failed: {str(e)}'}), 500
    
@app.route('/summary', methods=['GET'])
def summary_api():
    """时间范围汇总接口"""
    start_date = request.args.get('start')  # 格式：2025-12-01
    end_date = request.args.get('end')      # 格式：2025-12-31
    if not start_date or not end_date:
        return jsonify({'error': 'start and end date required (YYYY-MM-DD)'}), 400

    # 读取指定时间范围内的账单（简化版：直接让 LLM 做汇总）
    with open(EXPENSES_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    filtered_lines = []
    for line in lines:
        if line.strip() and not line.startswith("#"):
            try:
                ts_str = line.split("|", 1)[0].strip()
                ts = datetime.fromisoformat(ts_str)
                start_dt = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(end_date)
                if start_dt.date() <= ts.date() <= end_dt.date():
                    filtered_lines.append(line.strip())
            except:
                continue

    if not filtered_lines:
        return jsonify({'summary': '该时间段内无账单记录'}), 200

    # 让 LLM 做汇总
    context = "\n".join(filtered_lines)
    prompt = f"""请对以下时间段内的账单进行分类汇总：
时间段：{start_date} 至 {end_date}
账单记录：
{context}

请回答以下问题：
1. 有哪些消费种类？
2. 各种类的总金额是多少？
3. 哪些种类占支出大头（占比最高）？
请按清晰的格式回答。"""
    print("Prompt:", prompt)
    summary = llm.invoke(prompt,options={"num_predict": 8192})
    return jsonify({
        'summary': summary,
        'date_range': {'start': start_date, 'end': end_date},
        'total_records': len(filtered_lines)
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'internal server error'}), 500

# ==================== 启动入口 ====================
if __name__ == '__main__':
    print("🟢 Flask RAG API 启动中...")
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"🌐 Ollama 地址: {OLLAMA_BASE_URL}")
    print("💡 API 端点:")
    print("   POST /add    -> 添加账单")
    print("   GET  /query  -> 语义查询")
    print("   GET  /summary-> 时间汇总")
    print("   GET  /health -> 健康检查")
    
    # 预加载索引（避免首次查询慢）
    load_or_create_index()
    
    app.run(host='0.0.0.0', port=8911, debug=False)
