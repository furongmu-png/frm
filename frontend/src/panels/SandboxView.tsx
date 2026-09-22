// frontend/src/panels/SandboxView.tsx
// Physical sandbox view — renders base64-encoded frames from the model.

import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';

const containerStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
};

const placeholderWrapStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
};

const placeholderTextStyle: CSSProperties = {
  color: '#9ca3af',
  fontSize: '13px',
};

const imageWrapStyle: CSSProperties = {
  flex: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  minHeight: 0,
};

const footerStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  padding: '6px 8px',
  fontSize: '12px',
  color: '#aaa',
  borderTop: '1px solid #1a1a1a',
};

export default function SandboxView() {
  const snapshot = useModelStore((s) => s.snapshot);
  const [imgSrc, setImgSrc] = useState<string | null>(null);

  const frameB64 = snapshot?.frame_b64 ?? null;

  useEffect(() => {
    if (!frameB64) {
      setImgSrc(null);
      return;
    }
    const url = `data:image/png;base64,${frameB64}`;
    const img = new Image();
    img.onload = () => setImgSrc(url);
    img.src = url;
  }, [frameB64]);

  if (!frameB64) {
    return (
      <div style={placeholderWrapStyle}>
        <span style={placeholderTextStyle}>Waiting for sandbox data...</span>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <div style={imageWrapStyle}>
        {imgSrc && (
          <img
            src={imgSrc}
            alt={`Sandbox frame, modality ${snapshot?.modality ?? 'unknown'}, step ${snapshot?.step ?? 0}`}
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
          />
        )}
      </div>
      <div style={footerStyle}>
        <span>
          Modality: <strong style={{ color: '#e0e0e0' }}>{snapshot?.modality ?? '—'}</strong>
        </span>
        <span>
          Step: <strong style={{ color: '#e0e0e0' }}>{snapshot?.step ?? 0}</strong>
        </span>
      </div>
    </div>
  );
}
