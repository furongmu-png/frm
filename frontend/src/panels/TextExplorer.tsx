// frontend/src/panels/TextExplorer.tsx
// Text stream explorer with question injection.

import { useState } from 'react';
import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';

const containerStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
};

const preStyle: CSSProperties = {
  flex: 1,
  margin: 0,
  padding: '8px',
  fontFamily: 'ui-monospace, Consolas, monospace',
  fontSize: '12px',
  lineHeight: 1.5,
  color: '#e0e0e0',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  overflow: 'auto',
};

const labelsStyle: CSSProperties = {
  display: 'flex',
  gap: '12px',
  padding: '6px 8px',
  fontSize: '11px',
  color: '#aaa',
  borderTop: '1px solid #1a1a1a',
};

const inputRowStyle: CSSProperties = {
  display: 'flex',
  gap: '6px',
  padding: '6px 8px',
  borderTop: '1px solid #1a1a1a',
};

const inputStyle: CSSProperties = {
  flex: 1,
  padding: '4px 8px',
  fontSize: '12px',
  background: '#111',
  border: '1px solid #333',
  borderRadius: '4px',
  color: '#e0e0e0',
  outline: 'none',
};

const buttonStyle: CSSProperties = {
  padding: '4px 12px',
  fontSize: '12px',
  background: '#2563eb',
  border: 'none',
  borderRadius: '4px',
  color: '#fff',
  cursor: 'pointer',
};

const emptyStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '13px',
};

export default function TextExplorer() {
  const textBlock = useModelStore((s) => s.snapshot?.text_block ?? '');
  const modality = useModelStore((s) => s.snapshot?.modality ?? '—');
  const event = useModelStore((s) => s.snapshot?.event ?? '—');
  const injectQuestion = useModelStore((s) => s.injectQuestion);
  const [input, setInput] = useState('');

  const handleAsk = () => {
    const q = input.trim();
    if (!q) return;
    injectQuestion(q);
    setInput('');
  };

  if (!textBlock) {
    return <div style={emptyStyle}>Waiting for text stream...</div>;
  }

  return (
    <div style={containerStyle}>
      <pre style={preStyle}>{textBlock}</pre>
      <div style={labelsStyle}>
        <span>
          Modality: <strong style={{ color: '#e0e0e0' }}>{modality}</strong>
        </span>
        <span>
          Event: <strong style={{ color: '#e0e0e0' }}>{event}</strong>
        </span>
      </div>
      <div style={inputRowStyle}>
        <label htmlFor="text-explorer-question" className="sr-only">
          Ask a question about the text stream
        </label>
        <input
          id="text-explorer-question"
          style={inputStyle}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleAsk();
          }}
          placeholder="Ask a question..."
        />
        <button
          type="button"
          style={buttonStyle}
          onClick={handleAsk}
          aria-label="Ask question"
        >
          Ask
        </button>
      </div>
    </div>
  );
}
