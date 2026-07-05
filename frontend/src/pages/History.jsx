const MODE_LABELS = {
  detect: 'Phát hiện tổn thương da',
  segment: 'Phân đoạn phổi X-quang',
  pipeline: 'Pipeline ISIC đầy đủ',
};

export default function History() {
  const history = JSON.parse(localStorage.getItem('medseg_history') || '[]').reverse();

  const clearHistory = () => {
    localStorage.removeItem('medseg_history');
    window.location.reload();
  };

  return (
    <div>
      <h1 className="page-title">Lịch sử phân tích</h1>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <p style={{ color: 'var(--text-secondary)' }}>Đã ghi nhận {history.length} lượt phân tích</p>
        {history.length > 0 && (
          <button className="btn btn-outline" onClick={clearHistory}>Xóa lịch sử</button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: 48 }}>
          <p style={{ color: 'var(--text-secondary)' }}>Chưa có lượt phân tích nào</p>
        </div>
      ) : (
        <div className="card">
          <table>
            <thead>
              <tr><th>#</th><th>Ngày giờ</th><th>Chế độ</th><th>Trạng thái</th></tr>
            </thead>
            <tbody>
              {history.map((h, i) => (
                <tr key={i}>
                  <td>{history.length - i}</td>
                  <td>{new Date(h.timestamp).toLocaleString()}</td>
                  <td><span className="model-option selected" style={{ padding: '4px 10px', fontSize: '0.75rem' }}>{MODE_LABELS[h.mode] || h.mode}</span></td>
                  <td style={{ color: 'var(--success)' }}>Hoàn tất</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
