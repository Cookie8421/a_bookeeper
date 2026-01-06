import React, { useState } from 'react';
import { queryExpense } from '../services/api';

const QuerySection = () => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleQuery = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError('');
    setAnswer('');

    try {
      const response = await queryExpense(question);
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('查询失败:', error);
      setError(error.response?.data?.error || error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="query-section">
      <h2>🔍 语义查询</h2>
      <div className="query-input">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="例如：今天早餐花了多少？今年和谁看过电影？"
          className="input-field"
          onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
        />
        <button onClick={handleQuery} disabled={loading} className="btn-primary">
          {loading ? '查询中...' : '查询'}
        </button>
      </div>
      
      {(answer || error) && (
        <div className={`result ${error ? 'error' : 'success'}`}>
          <h3>{error ? '❌ 错误' : '💡 回答'}</h3>
          <p style={{ whiteSpace: 'pre-line' }}>{error || answer}</p>
        </div>
      )}

      {/* 常用查询示例 */}
      <div className="examples">
        <h4>💡 常用查询示例：</h4>
        <div className="example-buttons">
          {[
            '今天早餐花了多少？',
            '本周总支出是多少？',
            '今年和谁看过电影？',
            '12月整体收支情况',
            '本月交通费多少？'
          ].map((example, index) => (
            <button
              key={index}
              onClick={() => {
                setQuestion(example);
                setTimeout(handleQuery, 100); // 延迟触发查询
              }}
              className="example-btn"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QuerySection;