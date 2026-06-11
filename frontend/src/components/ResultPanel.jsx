export default function ResultPanel({ result, mode }) {
  if (!result) return null;

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3 style={{ marginBottom: 16 }}>📊 Analysis Results</h3>

      {/* Inference time */}
      <div className="metrics-grid" style={{ marginBottom: 16 }}>
        <div className="metric-card">
          <div className="metric-value good">
            {(result.inference_time_ms || result.total_time_ms || 0).toFixed(0)}ms
          </div>
          <div className="metric-label">Inference Time</div>
        </div>

        {(mode === 'detect' || mode === 'pipeline') && result.boxes && (
          <div className="metric-card">
            <div className="metric-value" style={{ color: 'var(--accent)' }}>{result.boxes?.length || 0}</div>
            <div className="metric-label">Detections</div>
          </div>
        )}

        {result.detection && (
          <div className="metric-card">
            <div className="metric-value" style={{ color: 'var(--accent)' }}>{result.detection?.boxes?.length || 0}</div>
            <div className="metric-label">Detections</div>
          </div>
        )}

        {result.dice_score != null && (
          <div className="metric-card">
            <div className={`metric-value ${result.dice_score > 0.8 ? 'good' : 'warn'}`}>
              {(result.dice_score * 100).toFixed(1)}%
            </div>
            <div className="metric-label">Dice Score</div>
          </div>
        )}
      </div>

      {/* Detection table */}
      {(result.boxes?.length > 0 || result.detection?.boxes?.length > 0) && (
        <div className="table-wrap">
          <table>
            <thead><tr><th>#</th><th>Bounding Box</th><th>Confidence</th></tr></thead>
            <tbody>
              {(result.boxes || result.detection?.boxes || []).map((box, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                    [{box.map(v => v.toFixed(0)).join(', ')}]
                  </td>
                  <td>
                    <span style={{
                      color: (result.scores || result.detection?.scores)?.[i] > 0.7 ? 'var(--success)' : 'var(--warning)',
                      fontWeight: 600
                    }}>
                      {((result.scores || result.detection?.scores)?.[i] * 100)?.toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
