const PROJECT_FLOW = [
  {
    step: 'User requirement',
    status: 'Đã có',
    detail: 'Upload ảnh y tế, chọn workflow, xem vùng phát hiện/phân đoạn và theo dõi chỉ số.',
  },
  {
    step: 'Features',
    status: 'Đã có',
    detail: 'Dashboard, Analyze, Compare, Metrics, History, xuất PDF báo cáo chỉ số.',
  },
  {
    step: 'Tech solutions',
    status: 'Đã có',
    detail: 'React + Vite, FastAPI, PyTorch, torchvision, Albumentations, pytest.',
  },
  {
    step: 'Logic + AI',
    status: 'Đã có',
    detail: 'Faster R-CNN cho detection; Attention U-Net cho segmentation; pipeline detection -> crop ROI -> segmentation.',
  },
  {
    step: 'Implement',
    status: 'Đã có',
    detail: 'Source code, API, frontend, checkpoint, config train/evaluate và tài liệu đều có trong project.',
  },
  {
    step: 'Test',
    status: 'Đã có, cần chạy lại khi nộp',
    detail: 'Có unit/integration test bằng pytest và evaluation trên nhiều ảnh test với CSV per-image.',
  },
];

const TEST_SCOPE = [
  {
    group: 'Code test',
    target: 'Kiểm tra model, dataset, transform, API và full pipeline có chạy đúng shape/response.',
    command: 'pytest tests/ -v',
  },
  {
    group: 'Segmentation evaluation',
    target: 'Chạy trên toàn bộ tập test, không dùng một ảnh đơn lẻ, rồi lấy Dice/IoU/Sensitivity/Specificity/Pixel Accuracy.',
    command: 'python scripts/evaluate.py --model segmentation --config configs/segmentation_config.yaml --checkpoint outputs/checkpoints/best_segmentation.pth --split test --output-csv outputs/metrics/isic2018_test_per_image.csv',
  },
  {
    group: 'Detection evaluation',
    target: 'Chạy trên toàn bộ tập test ISIC, lấy mAP@0.5, Precision, Recall, TP/FP/FN và IoU theo ảnh.',
    command: 'python scripts/evaluate.py --model detection --config configs/detection_config.yaml --checkpoint outputs/checkpoints/best_detection.pth --split test --output-csv outputs/metrics/isic2018_detection_test_per_image.csv',
  },
  {
    group: 'Manual UI test',
    target: 'Upload nhiều ảnh đại diện: ảnh dễ, ảnh khó, ảnh nhỏ/lớn, ảnh không đúng miền dữ liệu, kiểm tra overlay và thời gian phản hồi.',
    command: 'npm run dev + uvicorn api.main:app --reload --port 8000',
  },
];

const COMMON_CHECKS = [
  ['Train/val/test split', 'ISIC: 1815 / 389 / 390; Chest X-ray: 96 / 20 / 22. Cần giữ test độc lập, không dùng để train.'],
  ['Loss train/val', 'Training loop lưu CSV history trong outputs/logs/*_training_history.csv sau mỗi lần train. Segmentation có train_loss, train_dice, val_loss, val_dice; detection hiện có train_loss.'],
  ['Inference time/FPS', 'CSV per-image có inference_time_ms; dùng để tính tốc độ trung bình khi cần.'],
  ['Precision/Recall/F1', 'Detection đã có Precision và Recall; F1 có thể tính thêm từ 2 chỉ số này.'],
  ['Baseline/Ablation', 'Có U-Net và Attention U-Net trong code; báo cáo hiện chủ yếu dùng Attention U-Net, nếu cần mạnh hơn thì so sánh thêm U-Net.'],
  ['Threshold', 'Segmentation threshold = 0.5; detection confidence = 0.5 và IoU = 0.5. Phải ghi rõ khi trình bày.'],
];

const SEGMENTATION_RESULTS = [
  {
    title: 'Phân đoạn phổi X-quang',
    dataset: 'Montgomery County CXR Set',
    model: 'Attention U-Net',
    checkpoint: 'outputs/checkpoints/best_chest_xray_segmentation.pth',
    split: '22 ảnh test',
    csv: 'outputs/metrics/chest_xray_test_per_image.csv',
    rows: [
      { metric: 'Dice', value: '0.9811', mean: '0.9807', std: '0.0115' },
      { metric: 'IoU / Jaccard', value: '0.9628', mean: '0.9623', std: '0.0215' },
      { metric: 'Sensitivity / Recall', value: '0.9804', mean: '0.9807', std: '0.0227' },
      { metric: 'Specificity', value: '0.9932', mean: '0.9931', std: '0.0027' },
      { metric: 'Pixel Accuracy', value: '0.9897', mean: '0.9897', std: '0.0064' },
    ],
  },
  {
    title: 'Phân đoạn tổn thương da ISIC',
    dataset: 'ISIC 2018',
    model: 'Attention U-Net',
    checkpoint: 'outputs/checkpoints/best_segmentation.pth',
    split: '390 ảnh test',
    csv: 'outputs/metrics/isic2018_test_per_image.csv',
    rows: [
      { metric: 'Dice', value: '0.8847', mean: '0.8707', std: '0.1443' },
      { metric: 'IoU / Jaccard', value: '0.7932', mean: '0.7927', std: '0.1728' },
      { metric: 'Sensitivity / Recall', value: '0.8611', mean: '0.9135', std: '0.1495' },
      { metric: 'Specificity', value: '0.9774', mean: '0.9757', std: '0.0377' },
      { metric: 'Pixel Accuracy', value: '0.9532', mean: '0.9532', std: '0.0741' },
    ],
  },
];

const DETECTION_RESULT = {
  title: 'Phát hiện tổn thương da ISIC',
  dataset: 'ISIC 2018',
  model: 'Faster R-CNN ResNet-50 FPN',
  checkpoint: 'outputs/checkpoints/best_detection.pth',
  split: '390 ảnh test',
  csv: 'outputs/metrics/isic2018_detection_test_per_image.csv',
  rows: [
    { metric: 'mAP@0.5', value: '0.7142', meaning: 'Average Precision tại ngưỡng IoU 0.5.' },
    { metric: 'Precision@0.5', value: '0.8768', meaning: 'Tỉ lệ box dự đoán đúng trong các box model trả ra.' },
    { metric: 'Recall@0.5', value: '0.7846', meaning: 'Tỉ lệ tổn thương thật được model phát hiện.' },
    { metric: 'TP / FP / FN', value: '306 / 43 / 84', meaning: 'Số box đúng, báo nhầm và bỏ sót trên tập test.' },
    { metric: 'Best IoU mean/std', value: '0.6879 / 0.3365', meaning: 'IoU tốt nhất theo từng ảnh, tính trung bình và độ lệch chuẩn.' },
  ],
};

const METRIC_DETAILS = [
  ['Dice', '2TP / (2TP + FP + FN)', 'Chỉ số quan trọng nhất cho segmentation y tế, đo độ trùng giữa mask dự đoán và mask thật.'],
  ['IoU / Jaccard', 'TP / (TP + FP + FN)', 'Đo phần giao trên phần hợp nhất, thường khắt khe hơn Dice.'],
  ['Sensitivity / Recall', 'TP / (TP + FN)', 'Cho biết model bỏ sót vùng thật nhiều hay ít.'],
  ['Specificity', 'TN / (TN + FP)', 'Cho biết model nhận đúng vùng nền tốt không.'],
  ['Pixel Accuracy', '(TP + TN) / tổng pixel', 'Dễ cao nếu nền nhiều, nên chỉ đọc kèm Dice và IoU.'],
  ['mAP@0.5', 'AP với IoU >= 0.5', 'Chỉ số chính cho object detection, đánh giá chất lượng bounding box theo confidence.'],
];

function InfoTable({ item }) {
  return (
    <table>
      <tbody>
        <tr><th>Dataset</th><td>{item.dataset}</td></tr>
        <tr><th>Model</th><td>{item.model}</td></tr>
        <tr><th>Checkpoint</th><td><code>{item.checkpoint}</code></td></tr>
        <tr><th>Split test</th><td>{item.split}</td></tr>
        <tr><th>CSV per-image</th><td><code>{item.csv}</code></td></tr>
      </tbody>
    </table>
  );
}

export default function Metrics() {
  const exportPdf = () => {
    window.print();
  };

  return (
    <div>
      <div className="report-header">
        <div>
          <h1 className="page-title">Chỉ số đánh giá</h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Một trang tổng hợp quy trình, phạm vi test và số liệu thật của MedSeg trên các tập test độc lập.
          </p>
        </div>
        <button className="btn btn-primary no-print" onClick={exportPdf}>Xuất PDF</button>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">390</div>
          <div className="stat-label">Ảnh test ISIC</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">22</div>
          <div className="stat-label">Ảnh test X-quang</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">0.8847</div>
          <div className="stat-label">Dice ISIC segmentation</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">0.7142</div>
          <div className="stat-label">mAP@0.5 detection</div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3 style={{ marginBottom: 12 }}>Mức độ hoàn thành theo sơ đồ bài</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Bước</th>
                <th>Trạng thái</th>
                <th>Minh chứng trong project</th>
              </tr>
            </thead>
            <tbody>
              {PROJECT_FLOW.map((item) => (
                <tr key={item.step}>
                  <td><strong>{item.step}</strong></td>
                  <td><span className="status-pill ready">{item.status}</span></td>
                  <td>{item.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3 style={{ marginBottom: 12 }}>Bây giờ cần test những gì?</h3>
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
        <h3 style={{ marginBottom: 12 }}>Chỉ số chung cần theo dõi</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Chỉ số/phần kiểm tra</th>
                <th>Áp dụng trong bài này</th>
              </tr>
            </thead>
            <tbody>
              {COMMON_CHECKS.map(([name, detail]) => (
                <tr key={name}>
                  <td><strong>{name}</strong></td>
                  <td>{detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {SEGMENTATION_RESULTS.map((section) => (
        <div className="card" style={{ marginBottom: 16 }} key={section.title}>
          <h3 style={{ marginBottom: 12 }}>{section.title}</h3>
          <InfoTable item={section} />
          <div className="table-wrap" style={{ marginTop: 16 }}>
            <table>
              <thead>
                <tr>
                  <th>Chỉ số</th>
                  <th>Giá trị toàn test</th>
                  <th>Mean per-image</th>
                  <th>Std per-image</th>
                </tr>
              </thead>
              <tbody>
                {section.rows.map((row) => (
                  <tr key={row.metric}>
                    <td>{row.metric}</td>
                    <td><strong>{row.value}</strong></td>
                    <td>{row.mean}</td>
                    <td>{row.std}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      <div className="card" style={{ marginBottom: 16 }}>
        <h3 style={{ marginBottom: 12 }}>{DETECTION_RESULT.title}</h3>
        <InfoTable item={DETECTION_RESULT} />
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
              {DETECTION_RESULT.rows.map((row) => (
                <tr key={row.metric}>
                  <td>{row.metric}</td>
                  <td><strong>{row.value}</strong></td>
                  <td>{row.meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 12 }}>Công thức và cách đọc nhanh</h3>
        <div className="metrics-grid">
          {METRIC_DETAILS.map(([name, formula, description]) => (
            <div className="metric-card" key={name} style={{ textAlign: 'left' }}>
              <div className="metric-value" style={{ fontSize: '1rem', color: 'var(--text-primary)' }}>{name}</div>
              <div style={{ fontFamily: 'monospace', color: 'var(--accent)', margin: '8px 0' }}>{formula}</div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
