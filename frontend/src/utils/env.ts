const trimTrailingSlash = (url: string) => url.replace(/\/+$/, '');

const pickEnv = (key: 'BACKEND_URL' | 'FRONTEND_URL') => {
  const direct = import.meta.env[key as keyof ImportMetaEnv];
  const vitePrefixed = import.meta.env[`VITE_${key}` as keyof ImportMetaEnv];
  return (direct as string | undefined) || (vitePrefixed as string | undefined);
};

export const getBackendUrl = (): string => {
  const envBackend = pickEnv('BACKEND_URL');
  if (envBackend) {
    return trimTrailingSlash(envBackend);
  }

  if (import.meta.env.DEV) {
    return 'http://localhost:8000';
  }

  if (typeof window !== 'undefined') {
    return trimTrailingSlash(window.location.origin);
  }

  return 'http://localhost:8000';
};

export const getFrontendUrl = (): string => {
  const envFrontend = pickEnv('FRONTEND_URL');
  if (envFrontend) {
    return trimTrailingSlash(envFrontend);
  }

  if (typeof window !== 'undefined') {
    return trimTrailingSlash(window.location.origin);
  }

  return 'http://localhost:5173';
};

export const buildApiUrl = (path: string): string => {
  const base = getBackendUrl();
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${base}${normalizedPath}`;
};

export const buildWebSocketUrl = (path: string): string => {
  const base = getBackendUrl();
  const url = new URL(base);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.pathname = path.startsWith('/') ? path : `/${path}`;
  url.search = '';
  url.hash = '';
  return url.toString();
};
