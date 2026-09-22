// frontend/src/panels/KnowledgeGraphView.tsx
// Knowledge graph rendered with Cytoscape — incremental updates,
// search highlighting, and double-click to query a concept.

import { useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import type cytoscape from 'cytoscape';
import { useModelStore } from '../store/useModelStore';
import type { GraphData } from '../types';

const GROUP_COLORS: Record<string, string> = {
  default: '#6b7280',
  physics: '#3b82f6',
  text: '#22c55e',
  crossmodal: '#a855f7',
  image: '#ef4444',
  audio: '#eab308',
  code: '#a855f7',
};

const DEFAULT_COLOR = '#6b7280';

function colorFor(group?: string): string {
  if (!group) return DEFAULT_COLOR;
  return GROUP_COLORS[group] ?? DEFAULT_COLOR;
}

type ElementDef = cytoscape.ElementDefinition;

function toElements(graph: GraphData): ElementDef[] {
  const nodes: ElementDef[] = graph.nodes.map((n) => ({
    data: {
      id: n.id,
      label: n.label,
      size: Math.max(10, n.size ?? 30),
      color: colorFor(n.group),
      group: n.group ?? 'default',
    },
  }));
  const edges: ElementDef[] = graph.edges.map((e) => ({
    data: {
      id: `e-${e.source}-${e.target}`,
      source: e.source,
      target: e.target,
      width: Math.max(1, Math.min(6, e.weight ?? 1)),
    },
  }));
  return [...nodes, ...edges];
}

const STYLESHEET = [
  {
    selector: 'node',
    style: {
      'background-color': 'data(color)',
      label: 'data(label)',
      color: '#ffffff',
      'text-outline-color': '#0a0a0f',
      'text-outline-width': 2,
      'font-size': 10,
      width: 'data(size)',
      height: 'data(size)',
    },
  },
  {
    selector: 'node.dimmed',
    style: {
      opacity: 0.15,
      'text-opacity': 0.15,
    },
  },
  {
    selector: 'node.highlight',
    style: {
      'border-width': 3,
      'border-color': '#fbbf24',
      'border-style': 'solid',
    },
  },
  {
    selector: 'edge',
    style: {
      width: 'data(width)',
      'line-color': '#6b7280',
      'target-arrow-color': '#6b7280',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      opacity: 0.5,
    },
  },
  {
    selector: 'edge.dimmed',
    style: {
      opacity: 0.05,
    },
  },
];

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
  background: '#0a0a0f',
};

const toolbarStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '6px 8px',
  borderBottom: '1px solid #1a1a1a',
  flexShrink: 0,
};

const searchInputStyle: CSSProperties = {
  flex: 1,
  background: '#111',
  border: '1px solid #333',
  borderRadius: '4px',
  padding: '3px 8px',
  fontSize: '12px',
  color: '#e0e0e0',
  outline: 'none',
  minWidth: 0,
};

const legendStyle: CSSProperties = {
  display: 'flex',
  gap: '8px',
  flexWrap: 'wrap',
  fontSize: '10px',
  color: '#aaa',
};

const cyContainerStyle: CSSProperties = {
  flex: 1,
  minHeight: 0,
  position: 'relative',
};

export default function KnowledgeGraphView() {
  const knowledgeGraph = useModelStore((s) => s.knowledgeGraph);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const eventsBound = useRef(false);
  const [query, setQuery] = useState('');

  // Stable empty array — elements are managed via the cy ref.
  const initialElements = useMemo<ElementDef[]>(() => [], []);
  // Stable layout object — prevents re-layout on every render.
  const layout = useMemo(() => ({ name: 'cose', animate: true, fit: true }), []);

  const handleCy = (cy: cytoscape.Core) => {
    cyRef.current = cy;
    if (!eventsBound.current) {
      eventsBound.current = true;
      // Double-click a node → inject the concept as a question.
      cy.on('dbltap', 'node', (evt) => {
        const label = evt.target?.data('label') as string | undefined;
        if (label) {
          useModelStore.getState().injectQuestion(label);
        }
      });
    }
  };

  // Incremental sync: add/remove elements without recreating the graph.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    const desired = toElements(knowledgeGraph);
    const desiredIds = new Set(desired.map((el) => el.data.id as string));

    // Remove elements no longer present.
    const toRemoveIds: string[] = [];
    cy.elements().each((el) => {
      if (!desiredIds.has(el.id())) toRemoveIds.push(el.id());
    });
    let changed = false;
    if (toRemoveIds.length > 0) {
      toRemoveIds.forEach((id) => cy.remove(cy.getElementById(id)));
      changed = true;
    }

    // Add elements not yet in the graph.
    const existingIds = new Set<string>();
    cy.elements().each((el) => { existingIds.add(el.id()); });
    const toAdd = desired.filter((el) => !existingIds.has(el.data.id as string));
    if (toAdd.length > 0) {
      cy.add(toAdd);
      changed = true;
    }

    if (changed) {
      cy.layout({ name: 'cose', animate: true, fit: true }).run();
    }
  }, [knowledgeGraph]);

  // Search highlight: dim non-matching nodes and their edges.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().removeClass('dimmed highlight');
    const q = query.trim().toLowerCase();
    if (!q) return;
    cy.nodes().each((n) => {
      const label = String(n.data('label') ?? '').toLowerCase();
      if (label.includes(q)) {
        n.addClass('highlight');
      } else {
        n.addClass('dimmed');
      }
    });
    cy.edges().each((e) => {
      const connected =
        e.source().hasClass('highlight') || e.target().hasClass('highlight');
      if (!connected) {
        e.addClass('dimmed');
      }
    });
  }, [query, knowledgeGraph]);

  if (knowledgeGraph.nodes.length === 0 && knowledgeGraph.edges.length === 0) {
    return <div style={emptyStyle}>Waiting for knowledge graph...</div>;
  }

  const groups = Array.from(
    new Set(knowledgeGraph.nodes.map((n) => n.group ?? 'default')),
  );

  return (
    <div style={wrapperStyle}>
      <div style={toolbarStyle}>
        <label htmlFor="kg-search" className="sr-only">
          Search knowledge graph nodes
        </label>
        <input
          id="kg-search"
          style={searchInputStyle}
          placeholder="Search nodes..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div style={legendStyle} aria-hidden="true">
          {groups.map((g) => (
            <span key={g}>
              <span style={{ color: colorFor(g) }}>●</span> {g}
            </span>
          ))}
        </div>
      </div>
      <div
        style={cyContainerStyle}
        role="img"
        aria-label={`Knowledge graph with ${knowledgeGraph.nodes.length} nodes and ${knowledgeGraph.edges.length} edges, grouped by ${groups.join(', ')}. Double-click a node to query that concept.`}
      >
        <CytoscapeComponent
          elements={initialElements}
          layout={layout}
          style={{ width: '100%', height: '100%' }}
          stylesheet={STYLESHEET}
          cy={handleCy}
        />
      </div>
    </div>
  );
}
