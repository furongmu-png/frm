// frontend/src/panels/CausalGraph.tsx
// Causal emergence graph rendered with Cytoscape.
// Edge colour encodes causal sign: green = positive, red = negative.

import { useMemo } from 'react';
import type { CSSProperties } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import { useModelStore } from '../store/useModelStore';

const POSITIVE_COLOR = '#22c55e';
const NEGATIVE_COLOR = '#ef4444';

const emptyStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '13px',
};

const wrapperStyle: CSSProperties = {
  height: '100%',
  width: '100%',
  display: 'flex',
  flexDirection: 'column',
};

const legendBarStyle: CSSProperties = {
  display: 'flex',
  gap: '12px',
  padding: '4px 10px',
  fontSize: '10px',
  color: '#aaa',
  borderBottom: '1px solid #1a1a1a',
  flexShrink: 0,
};

const cyAreaStyle: CSSProperties = {
  flex: 1,
  minHeight: 0,
};

export default function CausalGraph() {
  const causalGraph = useModelStore((s) => s.snapshot?.causal_graph);

  const elements = useMemo(() => {
    if (!causalGraph) return [];
    const nodes = causalGraph.nodes.map((n) => ({
      data: { id: String(n.id), label: n.label },
    }));
    const edges = causalGraph.edges.map((e) => ({
      data: {
        id: `e-${e.source}-${e.target}`,
        source: String(e.source),
        target: String(e.target),
        strength: e.strength,
        width: Math.max(1, Math.min(6, Math.abs(e.strength) * 6)),
        lineColor: e.strength >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR,
      },
    }));
    return [...nodes, ...edges];
  }, [causalGraph]);

  if (
    !causalGraph ||
    (causalGraph.nodes.length === 0 && causalGraph.edges.length === 0)
  ) {
    return <div style={emptyStyle}>Waiting for causal structure...</div>;
  }

  return (
    <div style={wrapperStyle}>
      <div style={legendBarStyle} aria-hidden="true">
        <span>
          <span style={{ color: POSITIVE_COLOR }}>━</span> positive
        </span>
        <span>
          <span style={{ color: NEGATIVE_COLOR }}>━</span> negative
        </span>
      </div>
      <div
        style={cyAreaStyle}
        role="img"
        aria-label={`Causal emergence graph with ${causalGraph.nodes.length} nodes and ${causalGraph.edges.length} edges. Green edges indicate positive causal influence, red edges indicate negative.`}
      >
        <CytoscapeComponent
          elements={elements}
          layout={{ name: 'cose' }}
          style={{ width: '100%', height: '100%' }}
          stylesheet={[
            {
              selector: 'node',
              style: {
                'background-color': '#3b82f6',
                label: 'data(label)',
                color: '#e0e0e0',
                'text-outline-color': '#0a0a0f',
                'text-outline-width': 2,
                'font-size': 10,
                width: 20,
                height: 20,
              },
            },
            {
              selector: 'edge',
              style: {
                width: 'data(width)',
                'line-color': 'data(lineColor)',
                'target-arrow-color': 'data(lineColor)',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
              },
            },
          ]}
        />
      </div>
    </div>
  );
}
