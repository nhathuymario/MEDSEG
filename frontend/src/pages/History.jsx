export default function History() {
  const history = JSON.parse(localStorage.getItem('medseg_history') || '[]').reverse();

  const clearHistory = () => {
    localStorage.removeItem('medseg_history');
    window.location.reload();
  };

  return (
    <div>
      <h1 className="page-title">Analysis History</h1>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <p style={{ color: 'var(--text-secondary)' }}>{history.length} analyses recorded</p>
        {history.length > 0 && (
          <button className="btn btn-outline" onClick={clearHistory}>🗑️ Clear History</button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: 48 }}>
          <div style={{ fontSize: '3rem', marginBottom: 12 }}>📋</div>
          <p style={{ color: 'var(--text-secondary)' }}>No analyses yet</p>
        </div>
      ) : (
        <div className="card">
          <table>
            <thead>
              <tr><th>#</th><th>Date & Time</th><th>Mode</th><th>Status</th></tr>
            </thead>
            <tbody>
              {history.map((h, i) => (
                <tr key={i}>
                  <td>{history.length - i}</td>
                  <td>{new Date(h.timestamp).toLocaleString()}</td>
                  <td><span className="model-option selected" style={{ padding: '4px 10px', fontSize: '0.75rem' }}>{h.mode}</span></td>
                  <td style={{ color: 'var(--success)' }}>✅ Complete</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
