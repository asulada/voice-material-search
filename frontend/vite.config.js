import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 前端只做界面；/api 与 /api/jobs/*/ws 代理到 FastAPI（方案 §6.7）
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: false,
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
