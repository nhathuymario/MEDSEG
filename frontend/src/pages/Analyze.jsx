import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import { useAnalysis } from '../hooks/useAnalysis';
import ImageViewer from '../components/ImageViewer';
import ResultPanel from '../components/ResultPanel';

const MODES = [
  {
    id: 'detect',
    title: 'Phát hiện tổn thương da',
    description: 'ISIC Faster R-CNN',
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
    title: 'Pipeline ISIC đầy đủ',
    description: 'Phát hiện + phân đoạn tổn thương da',
    requiredModels: ['detector', 'isic_segmentor'],
  },
];

export default function Analyze() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [mode, setMode] = useState('detect');
  const [dragOver, setDragOver] = useState(false);
  const [health, setHealth] = useState(null);
  const inputRef = useRef();
  const { analyze, result, loading, error } = useAnalysis();

  useEffect(() => {
    api.health()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', models_loaded: [] }));
  }, []);

  const loadedModels = useMemo(
    () => new Set(health?.models_loaded || []),
    [health],
  );

  const modeStatuses = useMemo(() => {
    return MODES.reduce((acc, item) => {
      const missing = item.requiredModels.filter((name) => !loadedModels.has(name));
      acc[item.id] = { ready: missing.length === 0, missing };
      return acc;
    }, {});
  }, [loadedModels]);

  const selectedModeReady = modeStatuses[mode]?.ready ?? false;

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

  const handleSubmit = async () => {
    if (!file || !selectedModeReady) return;
    const analysisResult = await analyze(file, mode);
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
      <h1 className="page-title">Phân tích ảnh y tế</h1>

      <div
        className={`upload-zone ${dragOver ? 'dragover' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="icon">Tệp</div>
        <p><strong>Bấm để chọn</strong> hoặc kéo thả ảnh y tế vào đây</p>
        <p style={{ fontSize: '0.8rem', marginTop: 4 }}>Hỗ trợ JPEG, PNG</p>
        <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => handleFile(e.target.files[0])} />
      </div>

      {preview && (
        <>
          <div className="model-selector">
            {MODES.map((m) => {
              const status = modeStatuses[m.id] || { ready: false, missing: m.requiredModels };
              return (
                <button
                  key={m.id}
                  className={`model-option ${mode === m.id ? 'selected' : ''} ${status.ready ? 'ready' : 'pending'}`}
                  onClick={() => setMode(m.id)}
                  disabled={!status.ready}
                >
                  <span className="model-title">{m.title}</span>
                  <span className="model-meta">{m.description}</span>
                  <span className={`status-pill ${status.ready ? 'ready' : 'pending'}`}>
                    {status.ready ? 'sẵn sàng' : `thiếu: ${status.missing.join(', ')}`}
                  </span>
                </button>
              );
            })}
          </div>

          <button className="btn btn-primary" onClick={handleSubmit} disabled={loading || !selectedModeReady} style={{ marginBottom: 16 }}>
            {loading ? <><span className="loading-spinner" /> Đang phân tích...</> : 'Chạy phân tích'}
          </button>

          {error && <div className="error-msg">{error}</div>}

          <div className="results-panel">
            <div className="card">
              <h3 style={{ marginBottom: 12 }}>Ảnh gốc</h3>
              <ImageViewer src={preview} />
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
