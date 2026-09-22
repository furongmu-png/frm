// frontend/src/panels/SelfAuthoring.tsx
// Chat-like view of the model's self-authored thoughts.

import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import Markdown from 'react-markdown';
import { useModelStore } from '../store/useModelStore';

const emptyStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '13px',
};

const containerStyle: CSSProperties = {
  height: '100%',
  overflowY: 'auto',
  padding: '8px',
  display: 'flex',
  flexDirection: 'column',
  gap: '6px',
};

const stepLabelStyle: CSSProperties = {
  fontSize: '10px',
  color: '#9ca3af',
  marginTop: '2px',
};

export default function SelfAuthoring() {
  const messages = useModelStore((s) => s.authoringMessages);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  if (messages.length === 0) {
    return <div style={emptyStyle}>Model will self-author content as it learns...</div>;
  }

  return (
    <div ref={containerRef} style={containerStyle}>
      {messages.map((m, i) => (
        <div
          key={i}
          style={{
            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '85%',
            padding: '6px 10px',
            borderRadius: '8px',
            fontSize: '12px',
            lineHeight: 1.4,
            background: m.role === 'user' ? '#1e3a8a' : '#1a1a22',
            color: '#e0e0e0',
          }}
        >
          {m.role === 'model' ? <Markdown>{m.text}</Markdown> : m.text}
          <div style={stepLabelStyle}>step {m.step}</div>
        </div>
      ))}
    </div>
  );
}
