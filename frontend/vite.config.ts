import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: 'ws://localhost:8765',
        ws: true,
      },
    },
  },
  build: {
    // P2.11 性能优化: 默认 500KB 警告阈值改为 500KB（实际是 500KB，单位是 KB）。
    // 这里设为 500 以匹配优化目标——主 bundle 不应超过 500KB，重型 vendor
    // 库（three/cytoscape）应被拆分为独立 chunk 并懒加载。
    chunkSizeWarningLimit: 500,
    rollupOptions: {
      output: {
        // P2.11 性能优化: 手动分块——把重型第三方库按"功能域"拆分，配合
        // App.tsx 中的 React.lazy 懒加载（LatentSpace3D→three，
        // CausalGraph/KnowledgeGraphView→cytoscape），让首屏只下载
        // vendor-react + 主应用代码，其余 vendor chunk 在用户首次切换到
        // 对应 tab 时按需加载。
        // 用函数形式（而非对象形式）以兼容当前 Rollup 类型定义
        // (ManualChunksFunction)；按 node_modules 中的包名精确匹配，避免
        // ``includes('react')`` 误伤 react-* 系列包。
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined;
          // 提取包名：node_modules/pkg/... 或 node_modules/@scope/pkg/...
          const match = id.match(/node_modules\/((?:@[^/]+\/)?[^/]+)/);
          if (!match) return undefined;
          const pkg = match[1];
          // three.js + react-three-fiber/drei（仅 LatentSpace3D 使用）。
          if (
            pkg === 'three' ||
            pkg === '@react-three/fiber' ||
            pkg === '@react-three/drei'
          ) {
            return 'vendor-three';
          }
          // cytoscape 图引擎（CausalGraph / KnowledgeGraphView 使用）。
          if (pkg === 'cytoscape' || pkg === 'react-cytoscapejs') {
            return 'vendor-cytoscape';
          }
          // recharts 图表库（FreeEnergyChart / ConfidenceDashboard 等）。
          if (pkg === 'recharts') {
            return 'vendor-recharts';
          }
          // Markdown 渲染 + 代码高亮（TextExplorer / SelfAuthoring 使用）。
          if (pkg === 'react-markdown' || pkg === 'react-syntax-highlighter') {
            return 'vendor-markdown';
          }
          // React 核心 + 小型 react 生态库（常驻内存，首屏必需）。
          if (
            pkg === 'react' ||
            pkg === 'react-dom' ||
            pkg === 'react-resizable-panels' ||
            pkg === 'zustand'
          ) {
            return 'vendor-react';
          }
          return undefined;
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.ts'],
    // Exclude Playwright E2E tests (they use @playwright/test, not vitest)
    exclude: ['**/e2e/**', '**/node_modules/**', '**/dist/**'],
  },
})
