import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: '/4G-Traffic-prediction-web/',
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: process.env.VITE_DEV_API_URL || 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
