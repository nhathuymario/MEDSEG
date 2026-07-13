import { useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';

const MODEL_INFO = {
  detector: {
    label: 'Phát hiện tổn thương da Clinical + ISIC',
    detail: 'Faster R-CNN đa miền',
  },
  isic_segmentor: {
    label: 'Phân đoạn tổn thương da ISIC',
    detail: 'Attention U-Net',
  },
  chest_segmentor: {
    label: 'Phân đoạn phổi X-quang',
    detail: 'Attention U-Net',
  },
};

const MODE_LABELS = {
  detect: 'Phát hiện tổn thương da',
  segment: 'Phân đoạn phổi X-quang',
  pipeline: 'Pipeline ISIC đầy đủ',
};

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [history] = useState(() => JSON.parse(localStorage.getItem('medseg_history') || '[]'));

  useEffect(() => {
    api.health()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', gpu_available: false, models_loaded: [] }));
  }, []);

  const loadedModels = useMemo(() => health?.models_loaded || [], [health]);

  const stats = [
    { value: history.length, label: 'Tổng lượt phân tích', color: 'var(--accent)' },
    { value: health?.gpu_available ? 'Có' : 'Không', label: 'GPU khả dụng', color: health?.gpu_available ? 'var(--success)' : 'var(--warning)' },
    { value: loadedModels.length, label: 'Model đã tải', color: 'var(--accent)' },
    { value: health?.status === 'ok' ? 'ổn định' : (health?.status ? 'mất kết nối' : '...'), label: 'Trạng thái API', color: health?.status === 'ok' ? 'var(--success)' : 'var(--danger)' },
  ];

  return (
    <div>
      <h1 className="page-title">Tổng quan</h1>
      <div className="stats-grid">
        {stats.map((s, i) => (
          <div className="stat-card" key={s.label} style={{ animationDelay: `${i * 0.1}s` }}>
            <div className="stat-value" style={{ color: s.color }}>{s.value}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3 style={{ marginBottom: 16 }}>Model backend đã tải</h3>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {loadedModels.length === 0 ? (
            <p style={{ color: 'var(--text-secondary)' }}>Backend chưa tải checkpoint nào.</p>
          ) : (
            loadedModels.map((name) => {
              const info = MODEL_INFO[name] || { label: name, detail: 'Model backend' };
              return (
                <div key={name} className="model-option ready">
                  <span className="model-title">{info.label}</span>
                  <span className="model-meta">{info.detail}</span>
                  <span className="status-pill ready">{name}</span>
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3 style={{ marginBottom: 16 }}>Phân tích gần đây</h3>
        {history.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>Chưa có lượt phân tích nào. Vào trang Phân tích để bắt đầu.</p>
        ) : (
          <table>
            <thead><tr><th>Thời gian</th><th>Chế độ</th><th>Vùng phát hiện</th></tr></thead>
            <tbody>
              {history.slice(-5).reverse().map((h, i) => (
                <tr key={`${h.timestamp}-${i}`}>
                  <td>{new Date(h.timestamp).toLocaleString()}</td>
                  <td>{MODE_LABELS[h.mode] || h.mode}</td>
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
