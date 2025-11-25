/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly BACKEND_URL?: string;
  readonly FRONTEND_URL?: string;
  readonly VITE_BACKEND_URL?: string;
  readonly VITE_FRONTEND_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
