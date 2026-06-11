import { useParams } from 'react-router-dom';

export default function Results() {
  const { id } = useParams();

  return (
    <div>
      <h1 className="page-title">Analysis Result #{id}</h1>
      <div className="card">
        <p style={{ color: 'var(--text-secondary)' }}>
          Detailed result view. Select an analysis from the History page to view full results.
        </p>
      </div>
    </div>
  );
}
