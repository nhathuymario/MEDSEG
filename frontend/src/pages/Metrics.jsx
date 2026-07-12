import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';

const TEST_HISTORY_KEY = 'medseg_test_results';

const TEST_SCOPE = [
  {
    group: 'Code test',
    target: 'Kiểm tra dataset, model, transform, API và pipeline có chạy đúng shape/response.',
    command: 'pytest tests/ -v',
  },
  {
    group: 'Detection evaluation',
    target: 'Chạy toàn bộ test split để lấy mAP, Precision, Recall, TP/FP/FN và IoU per-image.',
    command: 'python scripts/evaluate.py --model detection --config configs/detection_config.yaml --checkpoint outputs/checkpoints/best_detection.pth --split test --output-csv outputs/metrics/isic2018_detection_test_per_image.csv',
  },
  {
    group: 'Overfitting check',
    target: 'Đọc outputs/logs/*_training_history.csv: train loss giảm nhưng val metric đứng/yếu đi là dấu hiệu học vẹt.',
    command: 'Xem bảng Training history ở trang này sau khi train.',
  },
];

const CHECKS = [
  ['Train/val/test split', 'ISIC detection dùng 1815 / 389 / 390 ảnh. Test phải độc lập, không dùng để train.'],
  ['Checkpoint selection', 'Detection lưu best_detection.pth theo val_mAP@0.5, không chỉ theo train loss.'],
  ['CSV per-image', 'Mỗi ảnh test có TP/FP/FN, precision, recall, best_iou và inference_time_ms.'],
  ['Summary JSON', 'scripts/evaluate.py sinh *.summary.json để web đọc số thật.'],
  ['FE test history', 'Trang này tự lưu snapshot kết quả test vào localStorage khi đọc được summary mới.'],
];

function fmt(value, digits = 4) {
  if (value === null || value === undefined || value === '') return '-';
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : value;
}

function getStoredTestHistory() {
  try {
    const history = JSON.parse(localStorage.getItem(TEST_HISTORY_KEY) || '[]');
    return history.filter((item) =>
      item.detection
      || item.isicSegmentation?.tp !== undefined
      || item.chestSegmentation?.tp !== undefined
    );
  } catch {
    return [];
  }
}

function makeSnapshot(metrics) {
  const summaries = metrics?.summaries || {};
  const det = summaries.isic2018_detection;
  const isicSeg = summaries.isic2018_segmentation;
  const chest = summaries.chest_xray_segmentation;

  if (!det && !isicSeg && !chest) return null;

  const runs = [
    ['Detection ISIC', det],
    ['Phân đoạn ISIC', isicSeg],
    ['Phân đoạn X-ray', chest],
  ].filter(([, item]) => item?.updated_at);
  const latestUpdatedAt = Math.max(...runs.map(([, item]) => item.updated_at), 0);
  const latestTests = runs
    .filter(([, item]) => item.updated_at === latestUpdatedAt)
    .map(([label]) => label);
  const isLatest = (label) => latestTests.includes(label);

  return {
    id: [
      det?.updated_at,
      isicSeg?.updated_at,
      chest?.updated_at,
    ].join('|'),
    savedAt: latestUpdatedAt || Date.now(),
    latestTests,
    detection: det && isLatest('Detection ISIC')
      ? {
          numImages: det.num_images,
          map: det.map,
          precision: det.precision,
          recall: det.recall,
          tp: det.tp,
          fp: det.fp,
          fn: det.fn,
          checkpoint: det.checkpoint,
          csv: det.csv,
        }
      : null,
    isicSegmentation: isicSeg && isLatest('Phân đoạn ISIC')
      ? {
          numImages: isicSeg.num_images,
          tp: isicSeg.metrics?.tp,
          fp: isicSeg.metrics?.fp,
          fn: isicSeg.metrics?.fn,
          dice: isicSeg.metrics?.dice,
          iou: isicSeg.metrics?.iou,
          checkpoint: isicSeg.checkpoint,
          csv: isicSeg.csv,
        }
      : null,
    chestSegmentation: chest && isLatest('Phân đoạn X-ray')
      ? {
          numImages: chest.num_images,
          tp: chest.metrics?.tp,
          fp: chest.metrics?.fp,
          fn: chest.metrics?.fn,
          dice: chest.metrics?.dice,
          iou: chest.metrics?.iou,
          checkpoint: chest.checkpoint,
          csv: chest.csv,
        }
      : null,
  };
}

function saveSnapshot(metrics) {
  const snapshot = makeSnapshot(metrics);
  if (!snapshot) return getStoredTestHistory();

  const history = getStoredTestHistory();
  const filtered = history.filter((item) => item.id !== snapshot.id);
  const next = [snapshot, ...filtered].slice(0, 20);
  localStorage.setItem(TEST_HISTORY_KEY, JSON.stringify(next));
  return next;
}

function InfoTable({ item }) {
  return (
    <table>
      <tbody>
        <tr><th>Dataset</th><td>{item.dataset || '-'}</td></tr>
        <tr><th>Checkpoint</th><td><code>{item.checkpoint || '-'}</code></td></tr>
        <tr><th>Split</th><td>{item.split || '-'} ({item.num_images || 0} ảnh)</td></tr>
        <tr><th>CSV per-image</th><td><code>{item.csv || '-'}</code></td></tr>
      </tbody>
    </table>
  );
}

function EmptyMetrics({ title }) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <h3 style={{ marginBottom: 8 }}>{title}</h3>
      <span className="status-pill warning">Chưa có summary thật</span>
      <p style={{ marginTop: 12, color: 'var(--text-secondary)' }}>
        Chạy evaluate với <code>--output-csv</code>; script sẽ tự tạo file <code>*.summary.json</code> để trang này đọc và lưu.
      </p>
    </div>
  );
}

function DetectionSection({ data }) {
  if (!data) return <EmptyMetrics title="Phát hiện tổn thương da ISIC" />;
  const rows = [
    ['mAP@0.5', fmt(data.map), 'Average Precision tại ngưỡng IoU 0.5.'],
    [`Precision@${data.confidence_threshold}`, fmt(data.precision), 'Tỉ lệ box dự đoán đúng trong các box model trả ra.'],
    [`Recall@${data.confidence_threshold}`, fmt(data.recall), 'Tỉ lệ tổn thương thật được model phát hiện.'],
    ['TP / FP / FN', `${data.tp} / ${data.fp} / ${data.fn}`, 'Số box đúng, báo nhầm và bỏ sót trên tập test.'],
    ['Best IoU mean/std', `${fmt(data.best_iou_mean)} / ${fmt(data.best_iou_std)}`, 'IoU tốt nhất theo từng ảnh, mean và std.'],
  ];

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <h3 style={{ marginBottom: 12 }}>Phát hiện tổn thương da ISIC</h3>
      <InfoTable item={data} />
      <div className="table-wrap" style={{ marginTop: 16 }}>
        <table>
          <thead>
            <tr>
              <th>Chỉ số</th>
              <th>Giá trị</th>
              <th>Ý nghĩa</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([metric, value, meaning]) => (
              <tr key={metric}>
                <td>{metric}</td>
                <td><strong>{value}</strong></td>
                <td>{meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SegmentationSection({ title, data }) {
  if (!data) return <EmptyMetrics title={title} />;
  const metricNames = ['dice', 'iou', 'sensitivity', 'specificity', 'pixel_accuracy'];

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <h3 style={{ marginBottom: 12 }}>{title}</h3>
      <InfoTable item={data} />
      <div className="table-wrap" style={{ marginTop: 16 }}>
        <table>
          <thead>
            <tr>
              <th>Chỉ số</th>
              <th>Toàn test</th>
              <th>Mean per-image</th>
              <th>Std per-image</th>
            </tr>
          </thead>
          <tbody>
            {metricNames.map((name) => (
              <tr key={name}>
                <td>{name}</td>
                <td><strong>{fmt(data.metrics?.[name])}</strong></td>
                <td>{fmt(data.per_image_mean?.[name])}</td>
                <td>{fmt(data.per_image_std?.[name])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PipelineSection({ data }) {
  if (!data) return <EmptyMetrics title="Full pipeline ISIC — Detection → Segmentation" />;
  const global = data.global_metrics || {};
  const time = data.per_image?.time_ms || {};
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <h3 style={{ marginBottom: 8 }}>Full pipeline ISIC — Detection → Segmentation</h3>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>
        Benchmark end-to-end trên {data.num_images} ảnh {data.split}: lỗi detection được truyền sang kết quả segmentation cuối cùng.
      </p>
      <div className="table-wrap"><table><tbody>
        <tr><th>Success rate</th><td><strong>{fmt((data.success_rate || 0) * 100, 2)}%</strong> ({data.num_images - data.zero_detection_images}/{data.num_images} ảnh có detection)</td></tr>
        <tr><th>Ảnh không phát hiện</th><td>{data.zero_detection_images}</td></tr>
        <tr><th>Global Dice / IoU</th><td><strong>{fmt(global.dice)} / {fmt(global.iou)}</strong></td></tr>
        <tr><th>Sensitivity / Specificity</th><td>{fmt(global.sensitivity)} / {fmt(global.specificity)}</td></tr>
        <tr><th>Pixel accuracy</th><td>{fmt(global.pixel_accuracy)}</td></tr>
        <tr><th>Latency mean / median / P95</th><td>{fmt(time.mean, 1)} / {fmt(time.median, 1)} / {fmt(time.p95, 1)} ms</td></tr>
        <tr><th>95% CI mean Dice</th><td>{fmt(data.per_image?.dice?.ci95_mean?.[0])} – {fmt(data.per_image?.dice?.ci95_mean?.[1])}</td></tr>
        <tr><th>CSV per-image</th><td><code>outputs/metrics/isic2018_pipeline_test_per_image.csv</code></td></tr>
      </tbody></table></div>
    </div>
  );
}

function BaselineSection({ attention, baseline }) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <h3 style={{ marginBottom: 8 }}>Baseline comparison — ISIC segmentation</h3>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>
        So sánh chỉ hợp lệ khi dùng cùng 390 ảnh test, preprocessing và threshold. Không điền số baseline khi chưa có checkpoint thật.
      </p>
      <div className="table-wrap"><table>
        <thead><tr><th>Model</th><th>Trạng thái</th><th>Dice</th><th>IoU</th><th>Sensitivity</th><th>Specificity</th><th>Pixel accuracy</th></tr></thead>
        <tbody>
          <tr><td>Attention U-Net</td><td><span className="status-pill success">Đã đánh giá</span></td><td>{fmt(attention?.metrics?.dice)}</td><td>{fmt(attention?.metrics?.iou)}</td><td>{fmt(attention?.metrics?.sensitivity)}</td><td>{fmt(attention?.metrics?.specificity)}</td><td>{fmt(attention?.metrics?.pixel_accuracy)}</td></tr>
          <tr><td>U-Net baseline</td><td><span className={`status-pill ${baseline ? 'success' : 'warning'}`}>{baseline ? 'Đã đánh giá' : 'Chưa train checkpoint'}</span></td><td>{fmt(baseline?.metrics?.dice)}</td><td>{fmt(baseline?.metrics?.iou)}</td><td>{fmt(baseline?.metrics?.sensitivity)}</td><td>{fmt(baseline?.metrics?.specificity)}</td><td>{fmt(baseline?.metrics?.pixel_accuracy)}</td></tr>
        </tbody>
      </table></div>
      {!baseline && <p style={{ marginTop: 12 }}><code>python scripts/train.py --config configs/segmentation_unet_baseline_config.yaml</code></p>}
    </div>
  );
}

function LargeSetDiagnostic({ data }) {
  if (!data) return <EmptyMetrics title="Large-set diagnostic — U-Net trên toàn bộ ISIC" />;
  const diceStats = data.per_image_statistics?.dice || {};
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <h3 style={{ marginBottom: 8 }}>Large-set diagnostic — U-Net trên toàn bộ ISIC</h3>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>
        Chạy trên {data.num_images} ảnh, gồm cả train/validation/test. Kết quả này dùng để kiểm tra độ ổn định,
        không thay thế benchmark test độc lập.
      </p>
      <InfoTable item={data} />
      <div className="table-wrap" style={{ marginTop: 16 }}><table><tbody>
        <tr><th>Global Dice / IoU</th><td><strong>{fmt(data.metrics?.dice)} / {fmt(data.metrics?.iou)}</strong></td></tr>
        <tr><th>Mean Dice per-image</th><td>{fmt(diceStats.mean)}</td></tr>
        <tr><th>95% CI mean Dice</th><td>{fmt(diceStats.ci95_mean?.[0])} – {fmt(diceStats.ci95_mean?.[1])}</td></tr>
        <tr><th>Sensitivity / Specificity</th><td>{fmt(data.metrics?.sensitivity)} / {fmt(data.metrics?.specificity)}</td></tr>
        <tr><th>Phạm vi</th><td><span className="status-pill warning">Diagnostic có train data</span></td></tr>
      </tbody></table></div>
    </div>
  );
}

function TrainingHistoryTable({ title, data, metricKey, metricLabel, expectedPath }) {
  const rows = data?.rows || [];
  const bestVal = rows.reduce((best, row) => {
    const value = Number(row[metricKey]);
    if (!Number.isFinite(value)) return best;
    return !best || value > Number(best[metricKey]) ? row : best;
  }, null);

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <h3 style={{ marginBottom: 12 }}>{title}</h3>
      {!data ? (
        <p style={{ color: 'var(--text-secondary)' }}>
          Chưa có <code>{expectedPath}</code>. File này được tạo khi chạy train lại mô hình.
        </p>
      ) : (
        <>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>
            Đã train {data.epochs} epoch. Best {metricLabel}: <strong>{fmt(bestVal?.[metricKey])}</strong>
            {bestVal ? ` tại epoch ${bestVal.epoch}` : ''}. Nếu train loss giảm nhưng validation metric không tăng, model có dấu hiệu học vẹt.
          </p>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.6 }}>
            <strong>Epoch:</strong> một vòng train qua toàn bộ tập huấn luyện.{' '}
            <strong>Train loss:</strong> sai số trên tập train, thường càng thấp càng tốt.{' '}
            {metricKey === 'val_dice' && <><strong>Train Dice:</strong> độ trùng khớp mask trên tập train.{' '}</>}
            <strong>{metricLabel}:</strong> {metricKey === 'val_dice' ? 'độ trùng khớp mask trên tập validation, càng cao càng tốt' : 'mAP phát hiện trên tập validation, càng cao càng tốt'}.{' '}
            {metricKey === 'val_dice' && <><strong>Val loss:</strong> sai số trên tập validation; tăng trong khi train loss giảm có thể là học vẹt.{' '}</>}
            <strong>LR:</strong> learning rate, độ lớn mỗi bước cập nhật trọng số.
          </p>
          <div className="table-wrap training-history-scroll">
            <table>
              <thead>
                <tr>
                  <th>Epoch</th>
                  <th>Train loss</th>
                  {metricKey === 'val_dice' && <th>Train Dice</th>}
                  <th>{metricLabel}</th>
                  {metricKey === 'val_dice' && <th>Val loss</th>}
                  <th>LR</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.epoch}>
                    <td>{row.epoch}</td>
                    <td>{fmt(row.train_loss)}</td>
                    {metricKey === 'val_dice' && <td>{fmt(row.train_dice)}</td>}
                    <td>{fmt(row[metricKey])}</td>
                    {metricKey === 'val_dice' && <td>{fmt(row.val_loss)}</td>}
                    <td>{fmt(row.lr, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function TrainingHistory({ history }) {
  return (
    <>
      <TrainingHistoryTable title="Training history — Detection ISIC" data={history?.detection} metricKey="val_map" metricLabel="Val mAP" expectedPath="outputs/logs/best_detection_training_history.csv" />
      <TrainingHistoryTable title="Training history — Phân đoạn ISIC" data={history?.isic2018_segmentation} metricKey="val_dice" metricLabel="Val Dice" expectedPath="outputs/logs/best_segmentation_training_history.csv" />
      <TrainingHistoryTable title="Training history — Phân đoạn X-ray" data={history?.chest_xray_segmentation} metricKey="val_dice" metricLabel="Val Dice" expectedPath="outputs/logs/best_chest_xray_segmentation_training_history.csv" />
    </>
  );
}

function TestHistory({ history, onClear }) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 12 }}>
        <h3>Lịch sử kết quả test đã lưu</h3>
        {history.length > 0 && (
          <button className="btn btn-outline no-print" onClick={onClear}>Xóa lịch sử test</button>
        )}
      </div>
      {history.length === 0 ? (
        <p style={{ color: 'var(--text-secondary)' }}>
          Chưa có snapshot nào. Khi trang đọc được summary sau evaluate, FE sẽ tự lưu một bản ở đây.
        </p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Thời gian lưu</th>
                <th>Nhóm vừa test</th>
                <th>Số ảnh test</th>
                <th>Chỉ số chính</th>
                <th>TP / FP / FN</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr key={`${item.id}-${item.savedAt}`}>
                  <td>{new Date(item.savedAt).toLocaleString()}</td>
                  <td>{item.latestTests?.join(', ') || '-'}</td>
                  <td>{item.detection?.numImages ?? item.isicSegmentation?.numImages ?? item.chestSegmentation?.numImages ?? '-'}</td>
                  <td>
                    {item.detection
                      ? `mAP ${fmt(item.detection.map)} · Precision ${fmt(item.detection.precision)} · Recall ${fmt(item.detection.recall)}`
                      : item.isicSegmentation
                        ? `Dice ${fmt(item.isicSegmentation.dice)} · IoU ${fmt(item.isicSegmentation.iou)}`
                        : `Dice ${fmt(item.chestSegmentation?.dice)} · IoU ${fmt(item.chestSegmentation?.iou)}`}
                  </td>
                  <td>
                    {item.detection
                      ? `${item.detection.tp} / ${item.detection.fp} / ${item.detection.fn}`
                      : item.isicSegmentation?.tp !== undefined
                        ? `${item.isicSegmentation.tp} / ${item.isicSegmentation.fp} / ${item.isicSegmentation.fn}`
                        : item.chestSegmentation?.tp !== undefined
                          ? `${item.chestSegmentation.tp} / ${item.chestSegmentation.fp} / ${item.chestSegmentation.fn}`
                          : '-'}
                    {' '}
                    <small>({item.detection ? 'tổn thương' : 'pixel'})</small>
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

export default function Metrics() {
  const [metrics, setMetrics] = useState(null);
  const [testHistory, setTestHistory] = useState(() => getStoredTestHistory());
  const [error, setError] = useState('');
  const [lastRefresh, setLastRefresh] = useState(null);
  const [evaluation, setEvaluation] = useState({ running: false });
  const [limits, setLimits] = useState({ detection: 390, 'isic-segmentation': 390, 'chest-xray-segmentation': 22, 'isic-pipeline': 390, 'isic-unet-all': 2594 });

  const refreshMetrics = useCallback(async () => {
    try {
      const data = await api.metrics();
      setMetrics(data);
      setTestHistory(saveSnapshot(data));
      setLastRefresh(Date.now());
      setError('');
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refreshMetrics();
    const refreshStatus = async () => {
      try {
        const status = await api.evaluationStatus();
        setEvaluation((previous) => {
          if (previous.running && !status.running && status.success) refreshMetrics();
          return status;
        });
      } catch (err) {
        setError(err.message);
      }
    };
    refreshStatus();
    const timer = window.setInterval(refreshStatus, 2000);
    return () => window.clearInterval(timer);
  }, [refreshMetrics]);

  const runEvaluation = async (kind) => {
    try {
      setError('');
      setEvaluation(await api.runEvaluation(kind, limits[kind]));
    } catch (err) {
      setError(err.message);
    }
  };

  const summaries = metrics?.summaries || {};
  const topStats = useMemo(() => {
    const det = summaries.isic2018_detection;
    const isicSeg = summaries.isic2018_segmentation;
    const chest = summaries.chest_xray_segmentation;
    return [
      ['ISIC test', det?.num_images || isicSeg?.num_images || 0],
      ['CXR test', chest?.num_images || 0],
      ['Dice ISIC', fmt(isicSeg?.metrics?.dice)],
      ['mAP detection', fmt(det?.map)],
    ];
  }, [summaries]);

  const clearTestHistory = () => {
    localStorage.removeItem(TEST_HISTORY_KEY);
    setTestHistory([]);
  };

  return (
    <div>
      <div className="report-header">
        <div>
          <h1 className="page-title">Chỉ số đánh giá</h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Trang này đọc số liệu thật từ <code>outputs/metrics</code> và <code>outputs/logs</code>, rồi tự lưu snapshot kết quả test ở FE.
          </p>
          {lastRefresh && (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              Cập nhật lần cuối: {new Date(lastRefresh).toLocaleString()}
            </p>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-outline no-print" onClick={refreshMetrics}>Làm mới số liệu</button>
          <button className="btn btn-primary no-print" onClick={() => window.print()}>Xuất PDF</button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ marginBottom: 16, color: 'var(--danger)' }}>
          Không đọc được metrics từ API: {error}
        </div>
      )}

      <div className="stats-grid">
        {topStats.map(([label, value]) => (
          <div className="stat-card" key={label}>
            <div className="stat-value">{value}</div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>

      <div className="card no-print" style={{ marginBottom: 16 }}>
        <h3 style={{ marginBottom: 8 }}>Chạy đánh giá tự động</h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>
          Chọn số ảnh từ test split có sẵn. Kết quả chạy ít ảnh chỉ phản ánh mẫu đã chọn, không thay thế kết quả test toàn bộ.
        </p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Mô hình</th><th>Số ảnh</th><th>Tối đa</th><th>Thao tác</th></tr></thead>
            <tbody>
              {[
                ['detection', 'Detection ISIC', 390],
                ['isic-segmentation', 'Phân đoạn ISIC', 390],
                ['chest-xray-segmentation', 'Phân đoạn X-ray', 22],
                ['isic-pipeline', 'Full pipeline ISIC', 390],
                ['isic-unet-all', 'U-Net diagnostic toàn bộ ISIC', 2594],
              ].map(([kind, label, total]) => (
                <tr key={kind}>
                  <td><strong>{label}</strong></td>
                  <td><input type="number" min="1" max={total} value={limits[kind]} onChange={(event) => setLimits({ ...limits, [kind]: Math.max(1, Math.min(total, Number(event.target.value) || 1)) })} style={{ width: 90 }} /></td>
                  <td>{total} ảnh test</td>
                  <td><button className="btn btn-primary" disabled={evaluation.running} onClick={() => runEvaluation(kind)}>{evaluation.running && evaluation.kind === kind ? 'Đang chạy...' : 'Chạy test'}</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {evaluation.message && <p style={{ marginTop: 12, color: evaluation.success === false ? 'var(--danger)' : 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>{evaluation.success === false ? 'Lỗi: ' : evaluation.success ? 'Hoàn thành: ' : ''}{evaluation.message}</p>}
      </div>

      <TestHistory history={testHistory} onClear={clearTestHistory} />

      <div className="card" style={{ marginBottom: 16 }}>
        <h3 style={{ marginBottom: 12 }}>Cần test và lưu gì sau train?</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Nhóm test</th>
                <th>Mục tiêu</th>
                <th>Lệnh/cách chạy</th>
              </tr>
            </thead>
            <tbody>
              {TEST_SCOPE.map((item) => (
                <tr key={item.group}>
                  <td><strong>{item.group}</strong></td>
                  <td>{item.target}</td>
                  <td><code>{item.command}</code></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3 style={{ marginBottom: 12 }}>Chống học vẹt và chống số ảo</h3>
        <div className="table-wrap">
          <table>
            <tbody>
              {CHECKS.map(([name, detail]) => (
                <tr key={name}>
                  <td><strong>{name}</strong></td>
                  <td>{detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <TrainingHistory history={metrics?.training_history} />
      <DetectionSection data={summaries.isic2018_detection} />
      <SegmentationSection title="Phân đoạn tổn thương da ISIC" data={summaries.isic2018_segmentation} />
      <SegmentationSection title="Phân đoạn phổi X-quang" data={summaries.chest_xray_segmentation} />
      <PipelineSection data={summaries.isic2018_pipeline} />
      <BaselineSection attention={summaries.isic2018_segmentation} baseline={summaries.isic2018_unet_baseline} />
      <LargeSetDiagnostic data={summaries.isic2018_unet_all_diagnostic} />
    </div>
  );
}
