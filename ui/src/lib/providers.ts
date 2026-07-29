import {
  supportsThinking,
  defaultThink,
  supportsMaxTokens,
} from "./models"

// ============================================================
// Types
// ============================================================

export interface ProviderConfig {
  readonly name: string
  readonly backend: string
  readonly baseUrl: string
  readonly apiKey: string
  readonly model: string
  readonly client: string
  readonly capabilities: string
}

// ============================================================
// Known roles
// ============================================================

const _KNOWN_ROLES = ["generate", "reformulate", "review", "supplement"] as const

// ============================================================
// Internal tables (mirrors Python _backend_url_table etc.)
// ============================================================

function _backendUrlTable(): Record<string, string> {
  return {
    flm: import.meta.env.VITE_LLM_FLM_URL ?? "",
    ollama: import.meta.env.VITE_LLM_OLLAMA_URL ?? "",
    gemini: import.meta.env.VITE_LLM_GEMINI_URL ?? import.meta.env.VITE_GEMINI_BASE_URL ?? "",
  }
}

function _backendApiKeyTable(): Record<string, string> {
  return {
    flm: "not-needed",
    ollama: "not-needed",
    gemini: import.meta.env.VITE_GEMINI_API_KEY ?? "",
  }
}

const _BACKEND_CLIENT: Record<string, string> = {
  flm: "openai_compat",
  ollama: "ollama_native",
  gemini: "openai_compat",
}

// ============================================================
// Role spec resolution (mirrors Python settings.role_spec)
// ============================================================

const _ENV_ROLE_MAP: Record<string, string> = {
  generate: "VITE_LLM_ROL_GENERATE",
  reformulate: "VITE_LLM_ROL_REFORMULATE",
  review: "VITE_LLM_ROL_REVIEW",
  supplement: "VITE_LLM_ROL_SUPPLEMENT",
}

function _roleSpec(role: string): [backend: string, model: string] {
  const envKey = _ENV_ROLE_MAP[role]
  if (!envKey) {
    throw new Error(`Unknown LLM role: '${role}'`)
  }
  const value = import.meta.env[envKey] ?? ""
  if (!value) {
    throw new Error(
      `${envKey} is not set. Use format "backend,model" (e.g. "flm,qwen3.5:9b").`,
    )
  }
  const parts = value.split(",")
  if (parts.length !== 2 || !parts[0] || !parts[1]) {
    throw new Error(
      `Invalid ${envKey}: '${value}'. Expected format "backend,model".`,
    )
  }
  return [parts[0], parts[1]]
}

// ============================================================
// Provider config builder (cached)
// ============================================================

let _configCache: Record<string, ProviderConfig> | null = null

function _buildProviderTable(): Record<string, ProviderConfig> {
  const table: Record<string, ProviderConfig> = {}
  const urlTable = _backendUrlTable()
  const keyTable = _backendApiKeyTable()

  for (const role of _KNOWN_ROLES) {
    const [backend, model] = _roleSpec(role)
    const envUrlKey = `VITE_LLM_${backend.toUpperCase()}_URL`

    if (!urlTable[backend]) {
      throw new Error(
        `Backend '${backend}' (role '${role}') has no URL configured -- ` +
        `${envUrlKey} is missing in .env`,
      )
    }

    table[role] = {
      name: role,
      backend,
      baseUrl: urlTable[backend],
      apiKey: keyTable[backend],
      model,
      client: _BACKEND_CLIENT[backend],
      capabilities: backend,
    }
  }

  return table
}

// ============================================================
// Public API
// ============================================================

export function listProviderConfigs(): Record<string, ProviderConfig> {
  if (!_configCache) {
    _configCache = _buildProviderTable()
  }
  return _configCache
}

export function getProviderConfig(name: string): ProviderConfig {
  const table = listProviderConfigs()
  const config = table[name]
  if (!config) {
    const valid = Object.keys(table).sort().join(", ")
    throw new Error(`Unknown LLM role: '${name}'. Valid: ${valid}`)
  }
  return config
}

// ============================================================
// ProviderInfo builder (mirrors src/api/routers/config.py logic)
// ============================================================

export interface ProviderInfo {
  name: string
  model: string
  supports: string[]
  defaultThink: boolean | null
}

export function listProviderInfos(): Record<string, ProviderInfo> {
  const configs = listProviderConfigs()
  const infos: Record<string, ProviderInfo> = {}

  for (const [role, config] of Object.entries(configs)) {
    const supports: string[] = ["extra"]
    if (supportsMaxTokens(config.capabilities, config.model)) {
      supports.push("max_tokens")
    }
    if (supportsThinking(config.capabilities, config.model)) {
      supports.push("think_mode")
    }

    infos[role] = {
      name: role,
      model: config.model,
      supports,
      defaultThink: defaultThink(config.capabilities, config.model),
    }
  }

  return infos
}

// ============================================================
// Cached client instances (per identity, not per role)
// ============================================================

export type ClientKind = "openai_compat" | "ollama_native"

export interface LLMClient {
  readonly kind: ClientKind
  readonly config: ProviderConfig
}

const _clientCache = new Map<string, LLMClient>()

function _clientCacheKey(config: ProviderConfig): string {
  return `${config.backend}|${config.baseUrl}|${config.apiKey}|${config.model}|${config.client}`
}

function _buildClient(config: ProviderConfig): LLMClient {
  return { kind: config.client as ClientKind, config }
}

export function getClient(name: string): LLMClient {
  const config = getProviderConfig(name)
  const key = _clientCacheKey(config)
  let client = _clientCache.get(key)
  if (!client) {
    client = _buildClient(config)
    _clientCache.set(key, client)
  }
  return client
}

// ============================================================
// Convenience: role -> capabilities via models.ts
// ============================================================

export function roleSupportsThinking(role: string): boolean {
  const config = getProviderConfig(role)
  return supportsThinking(config.capabilities, config.model)
}

export function roleDefaultThink(role: string): boolean | null {
  const config = getProviderConfig(role)
  return defaultThink(config.capabilities, config.model)
}

export function roleSupportsMaxTokens(role: string): boolean {
  const config = getProviderConfig(role)
  return supportsMaxTokens(config.capabilities, config.model)
}
