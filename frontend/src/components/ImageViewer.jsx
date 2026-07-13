import { useState } from 'react';

export default function ImageViewer({
  src,
  alt = 'Ảnh y tế',
  showOpacity = true,
  variant = 'default',
}) {
  const [opacity, setOpacity] = useState(1);
  const [dimensions, setDimensions] = useState(null);

  if (!src) return null;

  return (
    <div className="preview-container">
      <div className={`image-stage image-stage--${variant}`}>
        <img
          src={src}
          alt={alt}
          style={{ opacity }}
          onLoad={(event) => setDimensions({
            width: event.currentTarget.naturalWidth,
            height: event.currentTarget.naturalHeight,
          })}
        />
      </div>
      <div className="image-toolbar">
        <span>{dimensions ? `${dimensions.width} × ${dimensions.height}px` : 'Đang đọc kích thước...'}</span>
        <span>Giữ nguyên tỉ lệ</span>
      </div>
      {showOpacity && (
        <div className="slider-group">
          <label>Độ trong suốt</label>
          <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
        </div>
      )}
    </div>
  );
}
