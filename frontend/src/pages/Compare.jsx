import { useState, useRef } from 'react';
import { api } from '../api/client';
import ImageViewer from '../components/ImageViewer';

export default function Compare() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState({ unet: null, attn: null });
  const [loading, setLoading] = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResults({ unet: null, attn: null });
  };

  const runCompare = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const [unet, attn] = await Promise.all([api.segment(file), api.segment(file)]);
      setResults({ unet, attn });
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  return (
    <div>
      <h1 className="page-title">Compare Models</h1>

      <div className="upload-zone" onClick={() => inputRef.current.click()} style={{ marginBottom: 16 }}>
        <div className="icon">📁</div>
        <p>Upload an image to compare U-Net vs Attention U-Net</p>
        <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => handleFile(e.target.files[0])} />
      </div>

      {preview && (
        <>
          <button className="btn btn-primary" onClick={runCompare} disabled={loading} style={{ marginBottom: 16 }}>
            {loading ? <><span className="loading-spinner" /> Comparing...</> : '⚖️ Compare Models'}
          </button>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
            <div className="card">
              <h3 style={{ marginBottom: 8 }}>Original</h3>
              <ImageViewer src={preview} />
            </div>
            <div className="card">
              <h3 style={{ marginBottom: 8 }}>U-Net</h3>
              {results.unet ? (
                <>
                  <ImageViewer src={`data:image/png;base64,${results.unet.overlay_base64}`} />
                  <div className="metric-card" style={{ marginTop: 8 }}>
                    <div className="metric-value good">{results.unet.inference_time_ms?.toFixed(0)}ms</div>
                    <div className="metric-label">Inference Time</div>
                  </div>
                </>
              ) : <p style={{ color: 'var(--text-secondary)' }}>Click compare to run</p>}
            </div>
            <div className="card">
              <h3 style={{ marginBottom: 8 }}>Attention U-Net</h3>
              {results.attn ? (
                <>
                  <ImageViewer src={`data:image/png;base64,${results.attn.overlay_base64}`} />
                  <div className="metric-card" style={{ marginTop: 8 }}>
                    <div className="metric-value good">{results.attn.inference_time_ms?.toFixed(0)}ms</div>
                    <div className="metric-label">Inference Time</div>
                  </div>
                </>
              ) : <p style={{ color: 'var(--text-secondary)' }}>Click compare to run</p>}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
