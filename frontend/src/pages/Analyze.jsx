import { useState, useRef } from 'react';
import { useAnalysis } from '../hooks/useAnalysis';
import ImageViewer from '../components/ImageViewer';
import ResultPanel from '../components/ResultPanel';

const MODES = [
  { id: 'detect', label: '🎯 Detection (Faster R-CNN)' },
  { id: 'segment', label: '🔲 Segmentation (U-Net)' },
  { id: 'pipeline', label: '🔄 Full Pipeline' },
];

export default function Analyze() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [mode, setMode] = useState('pipeline');
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef();
  const { analyze, result, loading, error } = useAnalysis();

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleSubmit = () => {
    if (file) {
      analyze(file, mode);
      const history = JSON.parse(localStorage.getItem('medseg_history') || '[]');
      history.push({ timestamp: Date.now(), mode, detections: null });
      localStorage.setItem('medseg_history', JSON.stringify(history));
    }
  };

  return (
    <div>
      <h1 className="page-title">Analyze Medical Image</h1>

      {/* Upload */}
      <div
        className={`upload-zone ${dragOver ? 'dragover' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="icon">📁</div>
        <p><strong>Click or drag</strong> to upload medical image</p>
        <p style={{ fontSize: '0.8rem', marginTop: 4 }}>Supports JPEG, PNG</p>
        <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => handleFile(e.target.files[0])} />
      </div>

      {preview && (
        <>
          {/* Model selector */}
          <div className="model-selector">
            {MODES.map(m => (
              <button key={m.id} className={`model-option ${mode === m.id ? 'selected' : ''}`} onClick={() => setMode(m.id)}>
                {m.label}
              </button>
            ))}
          </div>

          <button className="btn btn-primary" onClick={handleSubmit} disabled={loading} style={{ marginBottom: 16 }}>
            {loading ? <><span className="loading-spinner" /> Analyzing...</> : '🚀 Run Analysis'}
          </button>

          {error && <div className="error-msg">❌ {error}</div>}

          <div className="results-panel">
            <div className="card">
              <h3 style={{ marginBottom: 12 }}>Original Image</h3>
              <ImageViewer src={preview} />
            </div>
            {result && (
              <div className="card">
                <h3 style={{ marginBottom: 12 }}>Result</h3>
                {result.combined_overlay_base64 || result.overlay_base64 ? (
                  <ImageViewer src={`data:image/png;base64,${result.combined_overlay_base64 || result.overlay_base64}`} />
                ) : (
                  <ImageViewer src={preview} />
                )}
              </div>
            )}
          </div>

          {result && <ResultPanel result={result} mode={mode} />}
        </>
      )}
    </div>
  );
}
