/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MEMORY_SERVER_URL: string
  readonly VITE_OPENAI_API_KEY: string
  readonly VITE_OPENAI_MODEL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
