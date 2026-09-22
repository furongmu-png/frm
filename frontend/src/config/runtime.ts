/**
 * Runtime configuration — resolves API/WS endpoints based on environment.
 *
 * Development (vite dev server):
 *   Vite proxy forwards /api → localhost:8000, /ws → localhost:8765
 *   So we use relative paths /api/v1 and /ws.
 *
 * Production (served by nginx/ingress):
 *   Ingress forwards /api/* → backend:8000 (rewrite stripping /api),
 *   /ws → backend:8765.
 *   So we use relative paths /api/v1 and /ws.
 *
 * Direct mode (legacy/docker-compose without ingress):
 *   Set VITE_API_BASE and VITE_WS_BASE env vars to point to host:port directly.
 *   For the REST API, the base URL must include the version prefix, e.g.
 *   ``http://localhost:8000/v1``.
 */

function getEnv(key: string, fallback = ''): string {
  // Vite exposes VITE_ prefixed env vars at build time
  const val = import.meta.env[key];
  return typeof val === 'string' ? val : fallback;
}

const DIRECT_API_BASE = getEnv('VITE_API_BASE', '');
const DIRECT_WS_BASE = getEnv('VITE_WS_BASE', '');

export const runtimeConfig = {
  /** REST API base URL. Empty string = use relative /api path (proxied). */
  apiBase: DIRECT_API_BASE,
  /** WebSocket base URL. Empty string = use relative /ws path (proxied). */
  wsBase: DIRECT_WS_BASE,
  /** Whether to use direct connection (bypass proxy) */
  isDirectMode: DIRECT_API_BASE !== '' || DIRECT_WS_BASE !== '',
} as const;

/**
 * Build full REST API URL from a path like '/health' or '/think'.
 * Returns either '/api/v1/health' (proxied) or 'http://host:8000/v1/health'
 * (direct). The nginx ingress strips the ``/api`` segment, so the backend
 * receives ``/v1/health``. Callers pass an unversioned path (e.g. ``/think``);
 * the ``/v1`` version prefix is added here so the version lives in one place.
 */
export function apiUrl(path: string): string {
  const base = runtimeConfig.apiBase;
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  if (base) {
    // Direct mode: the base URL already carries the version prefix
    // (e.g. ``http://localhost:8000/v1``); just append the path.
    return `${base.replace(/\/$/, '')}${cleanPath}`;
  }
  // Proxied mode: nginx strips ``/api`` → backend receives ``/v1/<path>``.
  return `/api/v1${cleanPath}`;
}

/**
 * Build full WebSocket URL from a path like '/ws' or ''.
 * Returns either 'ws://host:8765/ws' (direct) or relative '/ws' (proxied).
 */
export function wsUrl(path = '/ws'): string {
  const base = runtimeConfig.wsBase;
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  if (base) {
    // Direct mode
    const scheme = base.startsWith('https') ? 'wss' : base.startsWith('http') ? 'ws' : 'ws';
    const host = base.replace(/^https?:\/\//, '').replace(/\/$/, '');
    return `${scheme}://${host}${cleanPath}`;
  }
  // Proxied mode: use relative path, browser resolves based on current protocol/host
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}${cleanPath}`;
}
