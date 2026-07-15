/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  /** "true" enables the static portfolio demo -- see src/lib/demo.ts. */
  readonly VITE_DEMO_MODE?: string
  /** Only read when VITE_DEMO_MODE=true -- called directly from the browser. */
  readonly VITE_GEMINI_BASE_URL?: string
  readonly VITE_GEMINI_API_KEY?: string
  readonly VITE_GEMINI_MODEL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
