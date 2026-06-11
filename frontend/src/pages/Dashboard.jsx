import { useState, useEffect } from 'react';
import { api } from '../api/client';

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [history] = useState(() => JSON.parse(localStorage.getItem('medseg_history') || '[]'));

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'offline', gpu_available: false, models_loaded: [] }));
  }, []);

  const stats = [
    { value: history.length, label: 'Total Analyses', color: 'var(--accent)' },
    { value: health?.gpu_available ? 'Yes' : 'No', label: 'GPU Available', color: health?.gpu_available ? 'var(--success)' : 'var(--warning)' },
    { value: health?.models_loaded?.length || 0, label: 'Models Loaded', color: 'var(--accent)' },
    { value: health?.status || '...', label: 'API Status', color: health?.status === 'ok' ? 'var(--success)' : 'var(--danger)' },
  ];

  return (
    <div>
      <h1 className="page-title">Dashboard</h1>
      <div className="stats-grid">
        {stats.map((s, i) => (
          <div className="stat-card" key={i} style={{ animationDelay: `${i * 0.1}s` }}>
            <div className="stat-value" style={{ color: s.color }}>{s.value}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3 style={{ marginBottom: 16 }}>🧠 Available Models</h3>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {['Faster R-CNN (Detection)', 'U-Net (Segmentation)', 'Attention U-Net (Segmentation)'].map(m => (
            <div key={m} className="model-option">{m}</div>
          ))}
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3 style={{ marginBottom: 16 }}>📋 Recent Analyses</h3>
        {history.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>No analyses yet. Go to Analyze to get started.</p>
        ) : (
          <table>
            <thead><tr><th>Time</th><th>Mode</th><th>Detections</th></tr></thead>
            <tbody>
              {history.slice(-5).reverse().map((h, i) => (
                <tr key={i}>
                  <td>{new Date(h.timestamp).toLocaleString()}</td>
                  <td>{h.mode}</td>
                  <td>{h.detections ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
