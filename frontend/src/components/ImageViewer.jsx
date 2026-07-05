import { useState } from 'react';

export default function ImageViewer({ src, alt = 'Ảnh y tế' }) {
  const [opacity, setOpacity] = useState(1);

  if (!src) return null;

  return (
    <div className="preview-container">
      <img src={src} alt={alt} style={{ opacity }} />
      <div className="slider-group">
        <label>Độ trong suốt</label>
        <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(e.target.value)} />
      </div>
    </div>
  );
}
