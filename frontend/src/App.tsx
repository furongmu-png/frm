// frontend/src/App.tsx
// Window of Consciousness — main layout (Phase 5 + a11y/mobile).
//
// Desktop (>=768px): three resizable columns (react-resizable-panels).
//   - Left   : Control panel (top) + Physical Sandbox / Text (tabbed)
//   - Center : Free Energy chart (top) + Text Heatmap (bottom)
//   - Right  : 10 tabbed panels (Latent 3D, Causal, KG, ...)
// Mobile (<768px): single column. A global nav (Left/Center/Right) picks
//   which column is shown full-width; BeliefStateRaw stays at the bottom.
//
// Accessibility:
//   - Tab widgets use role="tablist"/"tab"/"tabpanel" with roving tabindex
//     and full arrow-key navigation (WAI-ARIA Authoring Practices).
//   - Panel resize separators expose role="separator" (set by the library)
//     plus a descriptive aria-label.
//   - The interactive dashboard surface uses role="application".
//   - Landmarks: <header> (banner) + <main>.
//   - Connection status is an aria-live status region.

import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import type { CSSProperties, KeyboardEvent } from 'react';
import { Panel, Group, Separator } from 'react-resizable-panels';
import { useModelStore } from './store/useModelStore';
import SandboxView from './panels/SandboxView';
import FreeEnergyChart from './panels/FreeEnergyChart';
import TextExplorer from './panels/TextExplorer';
import TextHeatmap from './panels/TextHeatmap';
import ControlPanel from './panels/ControlPanel';
import BeliefStateRaw from './panels/BeliefStateRaw';

// P2.11 性能优化: 右栏 10 个 tab 面板改为懒加载。这些面板一次只显示一个
// （tab 切换），且包含重型依赖（LatentSpace3D → three.js，
// CausalGraph/KnowledgeGraphView → cytoscape）。懒加载把它们拆出主 bundle，
// 显著减小首屏体积。常用/常驻面板（SandboxView、FreeEnergyChart、
// TextHeatmap、ControlPanel、BeliefStateRaw、TextExplorer）保持静态导入。
const LatentSpace3D = lazy(() => import('./panels/LatentSpace3D'));
const CausalGraph = lazy(() => import('./panels/CausalGraph'));
const KnowledgeGraphView = lazy(() => import('./panels/KnowledgeGraphView'));
const SelfAuthoring = lazy(() => import('./panels/SelfAuthoring'));
const StoryMode = lazy(() => import('./panels/StoryMode'));
const ModuleGraphPanel = lazy(() => import('./panels/ModuleGraphPanel'));
const ConfidenceDashboard = lazy(() => import('./panels/ConfidenceDashboard'));
const CommunicationLog = lazy(() => import('./panels/CommunicationLog'));
const ExperimentLog = lazy(() => import('./panels/ExperimentLog'));
const LogicPanel = lazy(() => import('./panels/LogicPanel'));
// Phase G (四.4): 三项认知升级面板（S4 / PCN / Hopfield）。
// 与其它右栏面板一样懒加载，避免拖大首屏 bundle。
const S4StateView = lazy(() => import('./panels/S4StateView'));
const LayerErrorHeatmap = lazy(() => import('./panels/LayerErrorHeatmap'));
const MemorySimilarityGraph = lazy(() => import('./panels/MemorySimilarityGraph'));
// Phase 2 (四.2): 推理链视图面板（神经符号推理结果）。
const ReasoningChainView = lazy(() => import('./panels/ReasoningChainView'));
// 技能层（src/skills/）：20 项技能的状态总览面板。
const SkillsPanel = lazy(() => import('./panels/SkillsPanel'));
// 下一代范式三件套：JEPA / GWT / 自主科学发现引擎。
// 与其它右栏面板一样懒加载，避免拖大首屏 bundle。
const JEPALatentPanel = lazy(() => import('./panels/JEPALatentPanel'));
const ConsciousnessTheaterPanel = lazy(() => import('./panels/ConsciousnessTheaterPanel'));
const DiscoveryConsolePanel = lazy(() => import('./panels/DiscoveryConsolePanel'));
// 终极升级三件套：具身主动感知 / 统一自我模型 (IWSM)。
// 同样懒加载，与现有面板共享 Suspense 边界。
const EmbodiedBodyPanel = lazy(() => import('./panels/EmbodiedBodyPanel'));
const SelfModelPanel = lazy(() => import('./panels/SelfModelPanel'));

const MOBILE_BREAKPOINT = 768;

const panelWrapperStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  overflow: 'hidden',
};

const titleStyle: CSSProperties = {
  padding: '4px 10px',
  fontSize: '11px',
  color: '#aaa',
  borderBottom: '1px solid #1a1a1a',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  flexShrink: 0,
};

const bodyStyle: CSSProperties = {
  flex: 1,
  overflow: 'auto',
  minHeight: 0,
};

// P2.11 性能优化: 懒加载面板的 Suspense fallback 样式（首次切换到该 tab
// 时显示，chunk 加载完成即被替换）。
const panelLoadingStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100%',
  color: '#aaa',
  fontSize: '12px',
  fontFamily: 'monospace',
};

const separatorStyle: CSSProperties = {
  background: '#222',
  position: 'relative',
};

const tablistStyle: CSSProperties = {
  display: 'flex',
  borderBottom: '1px solid #1a1a1a',
  flexShrink: 0,
  overflowX: 'auto',
};

const tabpanelStyle: CSSProperties = {
  flex: 1,
  overflow: 'hidden',
  minHeight: 0,
};

function PanelWrap({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={panelWrapperStyle}>
      <div style={titleStyle}>{title}</div>
      <div style={bodyStyle}>{children}</div>
    </div>
  );
}

type LeftTab = 'sandbox' | 'text';
type RightTab =
  | 'latent'
  | 'causal'
  | 'kg'
  | 'authoring'
  | 'story'
  | 'module'
  | 'confidence'
  | 'comm'
  | 'experiment'
  | 'logic'
  // Phase G (四.4): S4 / PCN / Hopfield 认知升级面板
  | 's4'
  | 'layer_err'
  | 'memory'
  // Phase 2 (四.2): 推理链视图面板
  | 'reasoning'
  // 技能层（src/skills/）：20 项技能状态总览
  | 'skills'
  // 下一代范式三件套：JEPA / GWT / 科学发现引擎
  | 'jepa'
  | 'gwt'
  | 'discovery'
  // 终极升级三件套：具身主动感知 / 统一自我模型 (IWSM)
  | 'embodiment'
  | 'self';
type MobileCol = 'left' | 'center' | 'right';

interface TabDef {
  id: string;
  label: string;
  testId?: string;
}

const LEFT_TABS: TabDef[] = [
  { id: 'sandbox', label: 'Sandbox' },
  { id: 'text', label: 'Text' },
];

const RIGHT_TABS: TabDef[] = [
  { id: 'latent', label: 'Latent 3D' },
  { id: 'causal', label: 'Causal' },
  { id: 'kg', label: 'KG' },
  { id: 'authoring', label: 'Authoring' },
  { id: 'story', label: 'Story' },
  { id: 'module', label: 'Module' },
  { id: 'confidence', label: 'Confidence' },
  { id: 'comm', label: 'Comm' },
  { id: 'experiment', label: 'Exp' },
  { id: 'logic', label: 'Logic' },
  // Phase G (四.4): S4 / PCN / Hopfield 认知升级面板
  { id: 's4', label: 'S4' },
  { id: 'layer_err', label: 'Layer Err' },
  { id: 'memory', label: 'Memory' },
  // Phase 2 (四.2): 推理链视图面板
  { id: 'reasoning', label: 'Reasoning' },
  // 技能层（src/skills/）：20 项技能状态总览
  { id: 'skills', label: 'Skills' },
  // 下一代范式三件套：JEPA / GWT / 科学发现引擎
  { id: 'jepa', label: 'JEPA' },
  { id: 'gwt', label: 'GWT' },
  { id: 'discovery', label: 'Discovery' },
  // 终极升级三件套：具身主动感知 / 统一自我模型 (IWSM)
  { id: 'embodiment', label: 'Embodiment' },
  { id: 'self', label: 'Self' },
];

const COL_NAV_TABS: TabDef[] = [
  { id: 'left', label: 'Left', testId: 'nav-left' },
  { id: 'center', label: 'Center', testId: 'nav-center' },
  { id: 'right', label: 'Right', testId: 'nav-right' },
];

function tabBtn(active: boolean): CSSProperties {
  return {
    padding: '6px 12px',
    fontSize: '11px',
    background: active ? '#1a2a3a' : 'transparent',
    border: 'none',
    borderBottom: active ? '2px solid #3b82f6' : '2px solid transparent',
    color: active ? '#e0e0e0' : '#9ca3af',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  };
}

/**
 * Accessible tab bar (WAI-ARIA tabs pattern):
 *   - role="tablist" container, role="tab" buttons with aria-selected.
 *   - Roving tabindex: only the active tab is in the tab order.
 *   - Arrow Left/Right, Home and End move focus and activation.
 *   - The active tab exposes aria-controls pointing at the rendered panel;
 *     inactive tabs omit aria-controls (their panels are not in the DOM) so
 *     the referenced id always resolves.
 */
function TabBar({
  tabs,
  activeId,
  onChange,
  tablistLabel,
  idPrefix,
  className,
}: {
  tabs: TabDef[];
  activeId: string;
  onChange: (id: string) => void;
  tablistLabel: string;
  idPrefix: string;
  className?: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, idx: number) => {
    let next = idx;
    switch (e.key) {
      case 'ArrowRight':
        next = (idx + 1) % tabs.length;
        break;
      case 'ArrowLeft':
        next = (idx - 1 + tabs.length) % tabs.length;
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = tabs.length - 1;
        break;
      default:
        return;
    }
    e.preventDefault();
    onChange(tabs[next].id);
    refs.current[next]?.focus();
  };

  return (
    <div role="tablist" aria-label={tablistLabel} className={className} style={tablistStyle}>
      {tabs.map((t, i) => {
        const selected = t.id === activeId;
        return (
          <button
            key={t.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={`${idPrefix}-tab-${t.id}`}
            aria-selected={selected}
            aria-controls={selected ? `${idPrefix}-panel-${t.id}` : undefined}
            aria-label={t.label}
            tabIndex={selected ? 0 : -1}
            className="tab-btn"
            data-testid={t.testId}
            style={tabBtn(selected)}
            onClick={() => onChange(t.id)}
            onKeyDown={(e) => onKeyDown(e, i)}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

export default function App() {
  const { connect, disconnect, connected } = useModelStore();
  const [leftTab, setLeftTab] = useState<LeftTab>('sandbox');
  const [rightTab, setRightTab] = useState<RightTab>('latent');
  const [mobileCol, setMobileCol] = useState<MobileCol>('left');

  // Responsive breakpoint: single-column nav on touch widths, three resizable
  // columns otherwise. Initialized synchronously from the viewport to avoid a
  // desktop→mobile flash on small screens.
  const [isMobile, setIsMobile] = useState(
    () => typeof window !== 'undefined' && window.innerWidth < MOBILE_BREAKPOINT,
  );

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < MOBILE_BREAKPOINT);
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  const renderLeftColumn = () => (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <TabBar
        tabs={LEFT_TABS}
        activeId={leftTab}
        onChange={(id) => setLeftTab(id as LeftTab)}
        tablistLabel="Left column panels"
        idPrefix="left"
      />
      <div
        role="tabpanel"
        id={`left-panel-${leftTab}`}
        aria-labelledby={`left-tab-${leftTab}`}
        style={tabpanelStyle}
      >
        {leftTab === 'sandbox' ? <SandboxView /> : <TextExplorer />}
      </div>
    </div>
  );

  const renderCenterColumn = () => (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <PanelWrap title="Free Energy & Prediction Error">
          <FreeEnergyChart />
        </PanelWrap>
      </div>
      <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <PanelWrap title="Text Stream Heatmap">
          <TextHeatmap />
        </PanelWrap>
      </div>
    </div>
  );

  const renderRightColumn = () => (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <TabBar
        tabs={RIGHT_TABS}
        activeId={rightTab}
        onChange={(id) => setRightTab(id as RightTab)}
        tablistLabel="Right column panels"
        idPrefix="right"
      />
      <div
        role="tabpanel"
        id={`right-panel-${rightTab}`}
        aria-labelledby={`right-tab-${rightTab}`}
        style={tabpanelStyle}
      >
        {/* P2.11 性能优化: 右栏 10 个面板均懒加载，单 Suspense 边界
            即可（一次只渲染一个 tab 的面板）。fallback 在对应 chunk
            下载期间显示，已加载的 tab 切回时为同步命中（无 fallback）。 */}
        <Suspense
          fallback={
            <div style={panelLoadingStyle} className="panel-loading">
              Loading...
            </div>
          }
        >
          {rightTab === 'latent' && <LatentSpace3D />}
          {rightTab === 'causal' && <CausalGraph />}
          {rightTab === 'kg' && <KnowledgeGraphView />}
          {rightTab === 'authoring' && <SelfAuthoring />}
          {rightTab === 'story' && <StoryMode />}
          {rightTab === 'module' && <ModuleGraphPanel />}
          {rightTab === 'confidence' && <ConfidenceDashboard />}
          {rightTab === 'comm' && <CommunicationLog />}
          {rightTab === 'experiment' && <ExperimentLog />}
          {rightTab === 'logic' && <LogicPanel />}
          {/* Phase G (四.4): S4 / PCN / Hopfield 认知升级面板 */}
          {rightTab === 's4' && <S4StateView />}
          {rightTab === 'layer_err' && <LayerErrorHeatmap />}
          {rightTab === 'memory' && <MemorySimilarityGraph />}
          {/* Phase 2 (四.2): 推理链视图面板 */}
          {rightTab === 'reasoning' && <ReasoningChainView />}
          {/* 技能层（src/skills/）：20 项技能状态总览 */}
          {rightTab === 'skills' && <SkillsPanel />}
          {/* 下一代范式三件套：JEPA / GWT / 科学发现引擎 */}
          {rightTab === 'jepa' && <JEPALatentPanel />}
          {rightTab === 'gwt' && <ConsciousnessTheaterPanel />}
          {rightTab === 'discovery' && <DiscoveryConsolePanel />}
          {/* 终极升级三件套：具身主动感知 / 统一自我模型 (IWSM) */}
          {rightTab === 'embodiment' && <EmbodiedBodyPanel />}
          {rightTab === 'self' && <SelfModelPanel />}
        </Suspense>
      </div>
    </div>
  );

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        background: '#0a0a0f',
        color: '#e0e0e0',
      }}
    >
      {/* Header — connection status indicator (live region for AT). */}
      <header
        style={{
          padding: '6px 16px',
          borderBottom: '1px solid #222',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          background: '#111',
          flexShrink: 0,
          flexWrap: 'wrap',
        }}
      >
        <h1 style={{ fontSize: '15px', margin: 0, fontWeight: 600, color: '#e0e0e0' }}>
          🧠 ZeroDataModel — Window of Consciousness
        </h1>
        <span
          role="status"
          aria-live="polite"
          aria-label={`Connection status: ${connected ? 'connected' : 'disconnected'}`}
          style={{
            padding: '2px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            background: connected ? '#1a3a1a' : '#3a1a1a',
            color: connected ? '#4ade80' : '#f87171',
          }}
        >
          {connected ? '● Connected' : '● Disconnected'}
        </span>
        {isMobile && (
          <TabBar
            tabs={COL_NAV_TABS}
            activeId={mobileCol}
            onChange={(id) => setMobileCol(id as MobileCol)}
            tablistLabel="Column navigation"
            idPrefix="colnav"
          />
        )}
      </header>

      {isMobile ? (
        <main
          style={{
            flex: 1,
            overflow: 'hidden',
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ flexShrink: 0, borderBottom: '1px solid #222', background: '#0d0d12' }}>
            <ControlPanel />
          </div>
          <div
            data-testid="column"
            role="tabpanel"
            id={`colnav-panel-${mobileCol}`}
            aria-labelledby={`colnav-tab-${mobileCol}`}
            style={{ flex: 1, overflow: 'auto', minHeight: 0 }}
          >
            {mobileCol === 'left' && (
              <div data-testid="left-content" style={{ height: '100%' }}>
                {renderLeftColumn()}
              </div>
            )}
            {mobileCol === 'center' && (
              <div data-testid="center-content" style={{ height: '100%' }}>
                {renderCenterColumn()}
              </div>
            )}
            {mobileCol === 'right' && (
              <div data-testid="right-content" style={{ height: '100%' }}>
                {renderRightColumn()}
              </div>
            )}
          </div>
          <div style={{ flexShrink: 0 }}>
            <BeliefStateRaw />
          </div>
        </main>
      ) : (
        <main
          style={{
            flex: 1,
            overflow: 'hidden',
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Top: Control panel (full width) */}
          <div style={{ flexShrink: 0, borderBottom: '1px solid #222', background: '#0d0d12' }}>
            <ControlPanel />
          </div>

          {/* Main: three resizable columns (interactive application surface) */}
          <div
            role="application"
            aria-label="Zero Data Model interactive dashboard"
            style={{ flex: 1, overflow: 'hidden', minHeight: 0 }}
          >
            <Group orientation="horizontal" style={{ height: '100%' }}>
              {/* Left column: control + sandbox/text */}
              <Panel defaultSize={25} minSize={15}>
                {renderLeftColumn()}
              </Panel>

              <Separator
                aria-label="Resize left and center panels"
                style={{ width: '4px', ...separatorStyle }}
              />

              {/* Center column: free energy chart + text heatmap */}
              <Panel defaultSize={40} minSize={20}>
                <Group orientation="vertical">
                  <Panel defaultSize={50} minSize={15}>
                    <PanelWrap title="Free Energy & Prediction Error">
                      <FreeEnergyChart />
                    </PanelWrap>
                  </Panel>
                  <Separator
                    aria-label="Resize free energy chart and text heatmap"
                    style={{ height: '4px', ...separatorStyle }}
                  />
                  <Panel defaultSize={50} minSize={15}>
                    <PanelWrap title="Text Stream Heatmap">
                      <TextHeatmap />
                    </PanelWrap>
                  </Panel>
                </Group>
              </Panel>

              <Separator
                aria-label="Resize center and right panels"
                style={{ width: '4px', ...separatorStyle }}
              />

              {/* Right column: tabbed panels (Latent 3D, Causal, KG, ...) */}
              <Panel defaultSize={35} minSize={20}>
                {renderRightColumn()}
              </Panel>
            </Group>
          </div>

          {/* Bottom: Belief State raw data (collapsible) */}
          <div style={{ flexShrink: 0 }}>
            <BeliefStateRaw />
          </div>
        </main>
      )}
    </div>
  );
}
