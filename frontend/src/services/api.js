import axios from 'axios';

// API 基础 URL（需要修改为你的服务地址）
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8911';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180000, // 180 秒（3 分钟）超时，支持长时间运行的 LLM 请求（平均 40 秒）
});

// 添加账单
export const addExpense = (text) => {
  return api.post('/add', { text });
};

// 查询账单
export const queryExpense = (question) => {
  return api.get(`/query`, { params: { q: question } });
};

// 时间范围汇总
export const getSummary = (start, end) => {
  return api.get(`/summary`, { params: { start, end } });
};

// 健康检查
export const healthCheck = () => {
  return api.get('/health');
};