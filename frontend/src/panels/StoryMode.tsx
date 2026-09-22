// frontend/src/panels/StoryMode.tsx
// Story mode — auto-plays learning milestones with synchronized replay.
//
// Fetches milestones from the REST API (/story-milestones), displays
// them as a vertical timeline of cards, and auto-advances through them
// at 4-second intervals. Clicking a card jumps all panels to that
// milestone's snapshot step via setPlaybackIndex.

import { useCallback, useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import { apiUrl } from '../config/runtime';
import type { Milestone } from '../types';

const PLAY_INTERVAL_MS = 4000;

const MILESTONE_ICONS: Record<string, string> = {
  free_energy_drop: '📉',
  high_surprise: '😲',
  first_causal_edge: '🔗',
  kg_growth_burst: '🌱',
  concept_cluster: '🧠',
  convergence: '✅',
  // Phase 2: logic discovery and experiment milestones.
  logic_discovery: '🔬',
  experiment_milestone: '⚗️',
  // 终极升级三件套里程碑类型
  embodied_exploration: '👁️',   // 具身探索：模型主动移动视线/触摸发现新物体
  self_narrative: '💬',          // 自我叙事：自传体记忆片段检索并生成叙述
  scientific_breakthrough: '🏆', // 科学突破：自主发现循环得到 ≥1 篇接受论文
  default: '📍',
};

const containerStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
};

const toolbarStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '6px 12px',
  borderBottom: '1px solid #1a1a1a',
  flexShrink: 0,
};

const btnStyle: CSSProperties = {
  padding: '4px 12px',
  fontSize: '12px',
  background: '#1a1a22',
  border: '1px solid #333',
  borderRadius: '4px',
  color: '#e0e0e0',
  cursor: 'pointer',
};

const timelineStyle: CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '8px 12px',
};

const cardStyle = (active: boolean): CSSProperties => ({
  display: 'flex',
  gap: '10px',
  padding: '8px 10px',
  marginBottom: '6px',
  borderRadius: '6px',
  cursor: 'pointer',
  border: active ? '2px solid #3b82f6' : '1px solid #222',
  background: active ? '#1a2a3a' : '#0d0d12',
  transition: 'border-color 0.2s, background 0.2s',
  textAlign: 'left',
  width: '100%',
  fontFamily: 'inherit',
});

const iconStyle: CSSProperties = {
  fontSize: '20px',
  flexShrink: 0,
};

const cardBodyStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '2px',
  minWidth: 0,
};

const emptyStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '13px',
  flexDirection: 'column',
  gap: '8px',
};

export default function StoryMode() {
  const milestones = useModelStore((s) => s.milestones);
  const setPlaybackIndex = useModelStore((s) => s.setPlaybackIndex);
  const history = useModelStore((s) => s.history);
  const [localMilestones, setLocalMilestones] = useState<Milestone[]>([]);
  const [playing, setPlaying] = useState(false);
  const [currentIdx, setCurrentIdx] = useState(-1);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Merge store milestones with locally fetched ones.
  const fetchMilestones = useCallback(async () => {
    try {
      const res = await fetch(apiUrl('/story-milestones'));
      const data = await res.json();
      if (data.milestones && Array.isArray(data.milestones)) {
        setLocalMilestones(data.milestones);
      }
    } catch {
      // REST not available — fall back to store milestones.
    }
  }, []);

  useEffect(() => {
    fetchMilestones();
    // Refresh every 10 seconds while not playing.
    const interval = setInterval(() => {
      if (!playing) fetchMilestones();
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchMilestones, playing]);

  const allMilestones = localMilestones.length > 0 ? localMilestones : milestones;

  // Find the history index closest to a milestone's step.
  const findHistoryIndex = useCallback(
    (step: number): number => {
      // Binary search for the closest step.
      let lo = 0;
      let hi = history.length - 1;
      let best = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const s = history[mid].step;
        if (s === step) return mid;
        if (s < step) {
          best = mid;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      return best;
    },
    [history],
  );

  // Auto-play: advance to the next milestone every PLAY_INTERVAL_MS.
  useEffect(() => {
    if (!playing || allMilestones.length === 0) return;
    if (currentIdx >= allMilestones.length - 1) {
      // Reached the end — stop playing.
      setPlaying(false);
      setCurrentIdx(-1);
      setPlaybackIndex(-1); // return to live
      return;
    }
    const nextIdx = currentIdx < 0 ? 0 : currentIdx + 1;
    const ms = allMilestones[nextIdx];
    const histIdx = findHistoryIndex(ms.step);
    if (histIdx >= 0) {
      setPlaybackIndex(histIdx);
    }
    setCurrentIdx(nextIdx);
    timerRef.current = setTimeout(() => {
      // The effect will re-run with the new currentIdx.
    }, PLAY_INTERVAL_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [playing, currentIdx, allMilestones, findHistoryIndex, setPlaybackIndex]);

  const handlePlay = () => {
    if (allMilestones.length === 0) return;
    setPlaying(true);
    setCurrentIdx(-1); // start from the beginning
  };

  const handleStop = () => {
    setPlaying(false);
    setCurrentIdx(-1);
    setPlaybackIndex(-1);
  };

  const handleCardClick = (ms: Milestone, idx: number) => {
    if (playing) return; // don't interrupt playback
    setCurrentIdx(idx);
    const histIdx = findHistoryIndex(ms.step);
    if (histIdx >= 0) {
      setPlaybackIndex(histIdx);
    }
  };

  if (allMilestones.length === 0) {
    return (
      <div style={emptyStyle}>
        <span style={{ fontSize: '32px' }} aria-hidden="true">📖</span>
        <span>No milestones detected yet.</span>
        <span style={{ fontSize: '11px' }}>
          Run the model for &gt;100 steps to generate learning events.
        </span>
        <button
          type="button"
          style={btnStyle}
          onClick={fetchMilestones}
          aria-label="Refresh milestones"
        >
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <div style={toolbarStyle}>
        {!playing ? (
          <button
            type="button"
            style={btnStyle}
            onClick={handlePlay}
            aria-label="Play story"
          >
            ▶ Play Story
          </button>
        ) : (
          <button
            type="button"
            style={btnStyle}
            onClick={handleStop}
            aria-label="Stop story"
          >
            ⏹ Stop
          </button>
        )}
        <span style={{ color: '#aaa', fontSize: '11px' }} aria-live="polite">
          {allMilestones.length} milestones
          {playing && currentIdx >= 0 && ` · ${currentIdx + 1}/${allMilestones.length}`}
        </span>
        <button
          type="button"
          style={{ ...btnStyle, marginLeft: 'auto' }}
          onClick={fetchMilestones}
          aria-label="Refresh milestones"
        >
          ↻ Refresh
        </button>
      </div>
      <div style={timelineStyle}>
        {allMilestones.map((ms, idx) => {
          const isActive = idx === currentIdx;
          const icon = MILESTONE_ICONS[ms.event] ?? MILESTONE_ICONS.default;
          return (
            <button
              key={`${ms.step}-${idx}`}
              type="button"
              style={cardStyle(isActive)}
              aria-pressed={isActive}
              aria-label={`Milestone ${ms.event || 'Milestone'} at step ${ms.step}${isActive ? ' (active)' : ''}`}
              onClick={() => handleCardClick(ms, idx)}
              disabled={playing}
            >
              <span style={iconStyle} aria-hidden="true">{icon}</span>
              <div style={cardBodyStyle}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: '#e0e0e0' }}>
                  {ms.event || 'Milestone'}
                  <span style={{ color: '#9ca3af', fontWeight: 400 }}> · step {ms.step}</span>
                </span>
                <span style={{ fontSize: '11px', color: '#aaa' }}>
                  {ms.description}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
