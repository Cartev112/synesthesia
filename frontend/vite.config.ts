import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const normalizeHttpUrl = (url: string) => url.replace(/\/+$/, '');
const toWebSocketUrl = (url: string) => {
  try {
    const parsed = new URL(url);
    parsed.protocol = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
    parsed.pathname = '';
    parsed.search = '';
    parsed.hash = '';
    return normalizeHttpUrl(parsed.toString());
  } catch {
    return normalizeHttpUrl(url.replace(/^http/i, 'ws'));
  }
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const backendUrl = normalizeHttpUrl(env.BACKEND_URL || env.VITE_BACKEND_URL || 'http://localhost:8000');

  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      host: '0.0.0.0',
      port: parseInt(process.env.PORT || '5173'),
      allowedHosts: ['synesthesia.up.railway.app'],
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
        },
        '/ws': {
          target: toWebSocketUrl(backendUrl),
          ws: true,
          changeOrigin: true,
        }
      }
    }
  }
})
