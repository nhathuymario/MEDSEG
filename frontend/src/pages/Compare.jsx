import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import ImageViewer from '../components/ImageViewer';

const WORKFLOWS = [
  {
    id: 'detect',
    name: 'Phát hiện tổn thương da ISIC',
    dataset: 'ISIC 2018',
    task: 'Phát hiện',
    model: 'Faster R-CNN',
    output: 'Bounding box và độ tin cậy',
    endpoint: '/api/detect',
    requiredModels: ['detector'],
  },
  {
    id: 'full-isic',
    name: 'Pipeline ISIC đầy đủ',
    dataset: 'ISIC 2018',
    task: 'Phát hiện + phân đoạn',
    model: 'Faster R-CNN + Attention U-Net',
    output: 'Bounding box, mask tổn thương và ảnh overlay',
    endpoint: '/api/pipeline',
    requiredModels: ['detector', 'isic_segmentor'],
  },
  {
    id: 'chest-segmentation',
    name: 'Phân đoạn phổi X-quang',
    dataset: 'Montgomery CXR',
    task: 'Phân đoạn',
    model: 'Attention U-Net',
    output: 'Mask vùng phổi',
    endpoint: '/api/segment',
    requiredModels: ['chest_segmentor'],
  },
];

export default function Compare() {
  const [health, setHealth] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  useEffect(() => {
    api.health()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', models_loaded: [] }));
  }, []);

  const loadedModels = useMemo(
    () => new Set(health?.models_loaded || []),
    [health],
  );

  const isicCompareReady = loadedModels.has('detector') && loadedModels.has('isic_segmentor');

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const runCompare = async () => {
    if (!file || !isicCompareReady) return;
    setLoading(true);
    setError(null);
    try {
      const [detect, pipeline] = await Promise.all([
        api.detect(file),
        api.pipeline(file),
      ]);
      setResult({ detect, pipeline });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 className="page-title">So sánh workflow</h1>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3 style={{ marginBottom: 12 }}>So sánh thực tế trên ảnh ISIC</h3>
        <p style={{ color: 'var(--text-secondary)' }}>
          Tải một ảnh da liễu lên để chạy cùng lúc hai luồng: phát hiện tổn thương và pipeline ISIC đầy đủ. Phân đoạn phổi X-quang không được đưa vào phép so sánh này vì khác loại ảnh và khác mục tiêu y khoa.
        </p>
      </div>

      <div className="upload-zone" onClick={() => inputRef.current.click()} style={{ marginBottom: 16 }}>
        <div className="icon">Tệp</div>
        <p><strong>Bấm để chọn</strong> ảnh da liễu ISIC cần so sánh</p>
        <p style={{ fontSize: '0.8rem', marginTop: 4 }}>Hỗ trợ JPEG, PNG</p>
        <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => handleFile(e.target.files[0])} />
      </div>

      {preview && (
        <>
          <button className="btn btn-primary" onClick={runCompare} disabled={loading || !isicCompareReady} style={{ marginBottom: 16 }}>
            {loading ? <><span className="loading-spinner" /> Đang so sánh...</> : 'Chạy so sánh ISIC'}
          </button>

          {!isicCompareReady && (
            <div className="error-msg">
              Cần tải đủ detector và isic_segmentor để chạy so sánh ISIC.
            </div>
          )}

          {error && <div className="error-msg">{error}</div>}

          <div className="results-panel">
            <div className="card">
              <h3 style={{ marginBottom: 12 }}>Ảnh gốc</h3>
              <ImageViewer src={preview} />
            </div>

            <div className="card">
              <h3 style={{ marginBottom: 12 }}>Phát hiện tổn thương</h3>
              {result?.detect ? (
                <>
                  <ImageViewer src={`data:image/png;base64,${result.detect.overlay_base64}`} />
                  <div className="metrics-grid" style={{ marginTop: 12 }}>
                    <div className="metric-card">
                      <div className="metric-value good">{result.detect.inference_time_ms?.toFixed(0)}ms</div>
                      <div className="metric-label">Thời gian suy luận</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-value" style={{ color: 'var(--accent)' }}>{result.detect.boxes?.length || 0}</div>
                      <div className="metric-label">Vùng phát hiện</div>
                    </div>
                  </div>
                </>
              ) : (
                <p style={{ color: 'var(--text-secondary)' }}>Chạy so sánh để xem bounding box.</p>
              )}
            </div>

            <div className="card">
              <h3 style={{ marginBottom: 12 }}>Pipeline ISIC đầy đủ</h3>
              {result?.pipeline ? (
                <>
                  <ImageViewer src={`data:image/png;base64,${result.pipeline.combined_overlay_base64}`} />
                  <div className="metrics-grid" style={{ marginTop: 12 }}>
                    <div className="metric-card">
                      <div className="metric-value good">{result.pipeline.total_time_ms?.toFixed(0)}ms</div>
                      <div className="metric-label">Tổng thời gian</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-value" style={{ color: 'var(--accent)' }}>{result.pipeline.detection?.boxes?.length || 0}</div>
                      <div className="metric-label">Vùng phát hiện</div>
                    </div>
                  </div>
                </>
              ) : (
                <p style={{ color: 'var(--text-secondary)' }}>Chạy so sánh để xem box kết hợp mask.</p>
              )}
            </div>
          </div>
        </>
      )}

      <div className="card" style={{ marginTop: 16 }}>
        <h3 style={{ marginBottom: 12 }}>Trạng thái workflow</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Luồng xử lý</th>
                <th>Bộ dữ liệu</th>
                <th>Tác vụ</th>
                <th>Mô hình</th>
                <th>Đầu ra</th>
                <th>Đường dẫn API</th>
                <th>Trạng thái</th>
              </tr>
            </thead>
            <tbody>
              {WORKFLOWS.map((workflow) => {
                const missing = workflow.requiredModels.filter((name) => !loadedModels.has(name));
                const ready = missing.length === 0;
                return (
                  <tr key={workflow.id}>
                    <td>{workflow.name}</td>
                    <td>{workflow.dataset}</td>
                    <td>{workflow.task}</td>
                    <td>{workflow.model}</td>
                    <td>{workflow.output}</td>
                    <td><code>{workflow.endpoint}</code></td>
                    <td>
                      <span className={`status-pill ${ready ? 'ready' : 'pending'}`}>
                        {ready ? 'sẵn sàng' : `thiếu: ${missing.join(', ')}`}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3 style={{ marginBottom: 12 }}>Nguyên tắc so sánh công bằng</h3>
        <p style={{ color: 'var(--text-secondary)' }}>
          So sánh trực tiếp chỉ nên thực hiện trên cùng loại ảnh và cùng mục tiêu. Vì vậy trang này chạy so sánh trên ảnh ISIC. Chest X-ray segmentation vẫn được liệt kê để biết trạng thái, nhưng không đem so với ISIC bằng cùng một ảnh.
        </p>
      </div>
    </div>
  );
}
