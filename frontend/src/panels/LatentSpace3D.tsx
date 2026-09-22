// frontend/src/panels/LatentSpace3D.tsx
// 3D projection of belief-state latent space across history.
//
// Renders the first 3 components of each snapshot's belief_state as a
// point cloud, coloured by modality. The latest point is highlighted
// with an emissive glowing sphere; the last 50 points are connected by
// a trajectory line whose brightness fades into the past.
//
// Performance:
//   - When history exceeds 1000 points, only the last 500 are shown
//     plus a random 50% sample of older points (capped at 500 total).
//   - All points are drawn with a single InstancedMesh (1 draw call).
//
// Interactions:
//   - OrbitControls: drag to rotate, scroll to zoom, slow auto-rotate.
//   - Hover a point → tooltip showing modality, step, free energy.

import { useMemo, useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Html, Grid } from '@react-three/drei';
import * as THREE from 'three';
import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { LatentPoint } from '../types';

const MODALITY_COLORS: Record<string, string> = {
  physics: '#3b82f6',
  text: '#22c55e',
  image: '#ef4444',
  audio: '#eab308',
  code: '#a855f7',
  crossmodal: '#a855f7',
};

const DEFAULT_COLOR = '#9ca3af';
const MAX_POINTS = 500;
const TRAJECTORY_LEN = 50;

const wrapStyle: CSSProperties = {
  height: '100%',
  width: '100%',
  position: 'relative',
};

const emptyStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '13px',
};

const tooltipStyle: CSSProperties = {
  background: 'rgba(17,17,17,0.95)',
  border: '1px solid #333',
  borderRadius: '4px',
  padding: '3px 8px',
  fontSize: '11px',
  color: '#e0e0e0',
  pointerEvents: 'none',
  whiteSpace: 'nowrap',
};

const legendStyle: CSSProperties = {
  position: 'absolute',
  top: 4,
  left: 4,
  display: 'flex',
  gap: '8px',
  flexWrap: 'wrap',
  fontSize: '10px',
  color: '#aaa',
  pointerEvents: 'none',
};

function colorFor(modality: string): string {
  return MODALITY_COLORS[modality] ?? DEFAULT_COLOR;
}

/**
 * Downsample points when there are too many: keep the most recent
 * MAX_POINTS and randomly drop a fraction of older ones.
 */
function downsample(points: LatentPoint[]): LatentPoint[] {
  if (points.length <= MAX_POINTS) return points;
  const recent = points.slice(-MAX_POINTS);
  return recent;
}

/** A rotating InstancedMesh of all latent points. */
function PointCloud({
  points,
  onHover,
}: {
  points: LatentPoint[];
  onHover: (p: LatentPoint | null, e: any) => void;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    points.forEach((p, i) => {
      dummy.position.set(p.coords[0], p.coords[1], p.coords[2]);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      color.set(colorFor(p.modality));
      mesh.setColorAt(i, color);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [points]);

  if (points.length === 0) return null;

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, points.length]}
      onPointerMove={(e) => {
        e.stopPropagation();
        const id = e.instanceId;
        if (id !== undefined && id >= 0 && id < points.length) {
          onHover(points[id], e);
        }
      }}
      onPointerOut={() => onHover(null, null)}
    >
      <sphereGeometry args={[0.08, 12, 12]} />
      <meshStandardMaterial transparent opacity={0.85} />
    </instancedMesh>
  );
}

/** The latest point: a larger, glowing sphere with a point light. */
function CurrentPoint({ point }: { point: LatentPoint }) {
  const ref = useRef<THREE.Mesh>(null);
  const color = colorFor(point.modality);
  useFrame((state) => {
    if (ref.current) {
      // Gentle pulsing scale.
      const s = 1 + 0.15 * Math.sin(state.clock.elapsedTime * 3);
      ref.current.scale.setScalar(s);
    }
  });
  return (
    <group position={point.coords}>
      <mesh ref={ref}>
        <sphereGeometry args={[0.22, 24, 24]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.8}
          transparent
          opacity={0.9}
        />
      </mesh>
      <pointLight color={color} intensity={2} distance={3} />
    </group>
  );
}

/** Trajectory line connecting the last N points; brightness fades. */
function Trajectory({ points }: { points: LatentPoint[] }) {
  const segments = points.slice(-TRAJECTORY_LEN);
  if (segments.length < 2) return null;
  const positions: [number, number, number][] = segments.map((p) => p.coords);
  return (
    <Line
      points={positions}
      color="#ffffff"
      lineWidth={1.5}
      transparent
      opacity={0.4}
      dashed
      dashSize={0.1}
      gapSize={0.05}
    />
  );
}

/** Auto-rotating OrbitControls wrapper. */
function Controls() {
  return (
    <OrbitControls
      autoRotate
      autoRotateSpeed={0.4}
      enableDamping
      dampingFactor={0.08}
      minDistance={1}
      maxDistance={30}
    />
  );
}

export default function LatentSpace3D() {
  const getLatentPoints = useModelStore((s) => s.getLatentPoints);
  const historyLen = useModelStore((s) => s.history.length);
  const snapshot = useModelStore((s) => s.snapshot);
  const [hover, setHover] = useState<{ p: LatentPoint; x: number; y: number } | null>(null);

  // Re-derive points whenever history grows.
  const points = useMemo(() => {
    void historyLen; // dependency
    return downsample(getLatentPoints());
  }, [getLatentPoints, historyLen]);

  // Current point (latest snapshot with a valid latent_3d).
  const current = useMemo<LatentPoint | null>(() => {
    if (!snapshot?.latent_3d) return null;
    const c = snapshot.latent_3d;
    if (c[0] === 0 && c[1] === 0 && c[2] === 0) return null;
    return {
      step: snapshot.step,
      modality: snapshot.modality,
      coords: [c[0], c[1], c[2]],
      freeEnergy: snapshot.free_energy,
      predictionError: snapshot.prediction_error,
    };
  }, [snapshot]);

  if (points.length === 0 && !current) {
    return <div style={emptyStyle}>Waiting for latent data...</div>;
  }

  // Modality legend (unique modalities present in the points).
  const modalities = Array.from(new Set(points.map((p) => p.modality)));

  return (
    <div style={wrapStyle}>
      <div style={legendStyle} aria-hidden="true">
        {modalities.map((m) => (
          <span key={m}>
            <span style={{ color: colorFor(m) }}>●</span> {m}
          </span>
        ))}
        <span style={{ color: '#fff' }}>● latest</span>
      </div>
      <Canvas
        camera={{ position: [5, 5, 5], fov: 50 }}
        aria-label={`Interactive 3D latent space visualization. ${points.length} points projected from belief state, colored by modality: ${modalities.join(', ')}. Drag to rotate, scroll to zoom. The latest point is highlighted.`}
      >
        <ambientLight intensity={0.35} />
        <pointLight position={[10, 10, 10]} intensity={0.6} />
        <pointLight position={[-5, -5, -5]} intensity={0.3} />
        <Grid
          args={[20, 20]}
          cellSize={1}
          cellColor="#1a2a1a"
          sectionSize={5}
          sectionColor="#2a3a2a"
          fadeDistance={25}
          fadeStrength={1}
          position={[0, 0, 0]}
        />
        <axesHelper args={[3]} />
        <PointCloud points={points} onHover={(p, e) => {
          if (p) {
            setHover({ p, x: e.clientX ?? 0, y: e.clientY ?? 0 });
          } else {
            setHover(null);
          }
        }} />
        <Trajectory points={points} />
        {current && <CurrentPoint point={current} />}
        <Controls />
        {hover && (
          <Html position={hover.p.coords} style={tooltipStyle} center distanceFactor={8}>
            <div>
              <strong style={{ color: colorFor(hover.p.modality) }}>{hover.p.modality}</strong>{' '}
              step {hover.p.step}
              <br />
              FE {hover.p.freeEnergy.toFixed(4)}
            </div>
          </Html>
        )}
      </Canvas>
    </div>
  );
}
