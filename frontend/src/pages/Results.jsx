import { useParams } from 'react-router-dom';

export default function Results() {
  const { id } = useParams();

  return (
    <div>
      <h1 className="page-title">Kết quả phân tích #{id}</h1>
      <div className="card">
        <p style={{ color: 'var(--text-secondary)' }}>
          Trang xem chi tiết kết quả. Chọn một lượt phân tích trong trang Lịch sử để xem đầy đủ.
        </p>
      </div>
    </div>
  );
}
