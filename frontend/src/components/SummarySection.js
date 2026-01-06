import React, { useState } from 'react';
import { getSummary } from '../services/api';

const SummarySection = () => {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSummary = async () => {
    if (!startDate || !endDate) {
      setError('请填写开始和结束日期');
      return;
    }

    setLoading(true);
    setError('');
    setSummary('');

    try {
      const response = await getSummary(startDate, endDate);
      setSummary(response.data.summary);
    } catch (error) {
      console.error('获取汇总失败:', error);
      setError(error.response?.data?.error || error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="summary-section">
      <h2>📊 时间范围汇总</h2>
      <div className="date-inputs">
        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          className="input-field"
        />
        <span> 至 </span>
        <input
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          className="input-field"
        />
        <button onClick={handleSummary} disabled={loading} className="btn-primary">
          {loading ? '汇总中...' : '汇总'}
        </button>
      </div>

      {(summary || error) && (
        <div className={`result ${error ? 'error' : 'success'}`}>
          <h3>{error ? '❌ 错误' : '📈 汇总结果'}</h3>
          <p style={{ whiteSpace: 'pre-line' }}>{error || summary}</p>
        </div>
      )}
    </div>
  );
};

export default SummarySection;