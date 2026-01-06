import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import AddExpense from './components/AddExpense';
import QuerySection from './components/QuerySection';
import SummarySection from './components/SummarySection';
import { healthCheck } from './services/api';
import './App.css';

function App() {
  const [isHealthy, setIsHealthy] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  // 检查 API 健康状态
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck();
        setIsHealthy(true);
      } catch (error) {
        console.error('API 健康检查失败:', error);
        setIsHealthy(false);
      }
    };

    checkHealth();
  }, []);

  const handleAddExpense = () => {
    setLastUpdate(new Date().toLocaleString());
  };

  return (
    <div className="App">
      <Header />
      
      <div className="status-bar">
        <span className={`status ${isHealthy ? 'healthy' : 'unhealthy'}`}>
          {isHealthy ? '🟢 API 服务正常' : '🔴 API 服务异常'}
        </span>
        {lastUpdate && <span className="last-update">上次更新: {lastUpdate}</span>}
      </div>

      <main className="main-content">
        <AddExpense onAdd={handleAddExpense} />
        <QuerySection />
        <SummarySection />
      </main>

      <footer className="footer">
        <p>🤖 基于 RAG 技术的智能记账助手</p>
      </footer>
    </div>
  );
}

export default App;