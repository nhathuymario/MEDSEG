import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../api/client';
import { useAnalysis } from '../hooks/useAnalysis';
import ImageViewer from '../components/ImageViewer';
import ResultPanel from '../components/ResultPanel';

const MODES = [
  {
    id: 'detect',
    title: 'Phát hiện tổn thương da đa miền',
    description: 'Faster R-CNN · ảnh clinical + dermoscopic',
    requiredModels: ['detector'],
  },
  {
    id: 'segment',
    title: 'Phân đoạn phổi X-quang',
    description: 'Chest Attention U-Net',
    requiredModels: ['chest_segmentor'],
  },
  {
    id: 'pipeline',
    title: 'Full pipeline tổn thương da',
    description: 'Phát hiện vùng tổn thương + phân đoạn mask',
    requiredModels: ['detector', 'isic_segmentor'],
  },
];

export default function Analyze() {
  const { mode: routeMode } = useParams();
  const mode = MODES.some((item) => item.id === routeMode) ? routeMode : 'detect';
  const activeMode = MODES.find((item) => item.id === mode) || MODES[0];
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [health, setHealth] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const inputRef = useRef();
  const { analyze, result, loading, error, setResult } = useAnalysis();

  useEffect(() => {
    api.health()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', models_loaded: [] }));
  }, []);

  useEffect(() => {
    setResult(null);
  }, [mode, setResult]);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const loadedModels = useMemo(
    () => new Set(health?.models_loaded || []),
    [health],
  );

  const missingModels = activeMode.requiredModels.filter((name) => !loadedModels.has(name));
  const selectedModeReady = missingModels.length === 0;

  const handleFile = (nextFile) => {
    if (!nextFile) return;
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setResult(null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    handleFile(event.dataTransfer.files[0]);
  };

  const handleSubmit = async () => {
    if (!file || !selectedModeReady) return;
    const analysisResult = await analyze(file, mode, threshold);
    if (!analysisResult) return;
    const detections = (
      analysisResult.boxes
      || analysisResult.detection?.boxes
      || []
    ).length;
    const history = JSON.parse(localStorage.getItem('medseg_history') || '[]');
    history.push({ timestamp: analysisResult.timestamp, mode, detections });
    localStorage.setItem('medseg_history', JSON.stringify(history));
  };

  return (
    <div>
      <h1 className="page-title">{activeMode.title}</h1>

      <div className="card function-summary">
        <div>
          <span className="function-kicker">Chức năng đang chọn</span>
          <h3>{activeMode.title}</h3>
          <p>{activeMode.description}</p>
        </div>
        <span className={`status-pill ${selectedModeReady ? 'ready' : 'pending'}`}>
          {selectedModeReady ? 'sẵn sàng' : `thiếu: ${missingModels.join(', ')}`}
        </span>
      </div>

      <div
        className={`upload-zone ${dragOver ? 'dragover' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(event) => { event.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="icon">Tệp</div>
        <p><strong>Bấm để chọn</strong> hoặc kéo thả ảnh y tế vào đây</p>
        <p style={{ fontSize: '0.8rem', marginTop: 4 }}>Hỗ trợ JPEG, PNG và WebP</p>
        <input ref={inputRef} type="file" accept="image/*" hidden onChange={(event) => handleFile(event.target.files[0])} />
      </div>

      {preview && (
        <>
          {(mode === 'detect' || mode === 'pipeline') && (
            <div className="card" style={{ marginBottom: 16 }}>
              <h3 style={{ marginBottom: 12 }}>Tùy chỉnh</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <label style={{ flexShrink: 0 }}>Ngưỡng Confidence (mAP): {threshold.toFixed(2)}</label>
                <input 
                  type="range" 
                  min="0.05" 
                  max="1.0" 
                  step="0.05" 
                  value={threshold} 
                  onChange={(e) => setThreshold(parseFloat(e.target.value))} 
                  style={{ flexGrow: 1 }}
                />
              </div>
            </div>
          )}

          <button className="btn btn-primary" onClick={handleSubmit} disabled={loading || !selectedModeReady} style={{ marginBottom: 16 }}>
            {loading ? <><span className="loading-spinner" /> Đang phân tích...</> : `Chạy ${activeMode.title}`}
          </button>

          {error && <div className="error-msg">{error}</div>}

          <div className="results-panel">
            <div className="card">
              <h3 style={{ marginBottom: 12 }}>Ảnh gốc</h3>
              <ImageViewer src={preview} showOpacity={false} />
            </div>
            {result && (
              <div className="card">
                <h3 style={{ marginBottom: 12 }}>Kết quả</h3>
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
