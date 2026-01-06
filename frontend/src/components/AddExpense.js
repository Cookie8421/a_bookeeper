import React, { useState } from 'react';
import { addExpense } from '../services/api';

const AddExpense = ({ onAdd }) => {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setLoading(true);
    setMessage('');

    try {
      await addExpense(input);
      setMessage('✅ 账单添加成功！');
      setInput('');
      onAdd(); // ✅ 修正：调用父组件传入的 onAdd 回调函数
    } catch (error) {
      console.error('添加账单失败:', error);
      setMessage('❌ 添加失败: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="add-expense">
      <h2>📝 添加账单</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="例如：早餐 煎饼果子 8元"
          className="input-field"
          disabled={loading}
        />
        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? '添加中...' : '添加'}
        </button>
      </form>
      {message && <div className={`message ${message.includes('✅') ? 'success' : 'error'}`}>{message}</div>}
    </div>
  );
};

export default AddExpense;