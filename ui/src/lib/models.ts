import models from "@models/models.json"

export interface ModelKwargs {
  timeout?: number
  temperature?: number
  max_tokens?: number
  top_p?: number
  presence_penalty?: number
  frequency_penalty?: number
  extra_body?: Record<string, unknown>
  options?: Record<string, unknown>
  think?: boolean
  model?: string
  messages?: unknown[]
}

export interface ModelEntry {
  context_window: number | null
  kwargs: ModelKwargs
}

export type ModelsData = Record<string, Record<string, ModelEntry>>

const _data: ModelsData = models as ModelsData

export function getBackendModels(backend: string): Record<string, ModelEntry> {
  const entry = _data[backend]
  if (!entry) {
    throw new Error(`Unknown capabilities backend: '${backend}'`)
  }
  return entry
}

function _lookup(backend: string, model: string): ModelEntry {
  const backends = getBackendModels(backend)
  const entry = backends[model]
  if (!entry) {
    throw new Error(`Unsupported model '${model}' for backend '${backend}'`)
  }
  return entry
}

export function supportsThinking(backend: string, model: string): boolean {
  const kwargs = _lookup(backend, model).kwargs
  const extraBody = kwargs.extra_body
  if (extraBody && "enable_thinking" in extraBody) {
    return true
  }
  const chatKwargs = extraBody?.chat_template_kwargs as Record<string, unknown> | undefined
  if (chatKwargs && "enable_thinking" in chatKwargs) {
    return true
  }
  if ("think" in kwargs) {
    return true
  }
  return false
}

export function defaultThink(backend: string, model: string): boolean | null {
  if (!supportsThinking(backend, model)) {
    return null
  }
  const kwargs = _lookup(backend, model).kwargs
  const extraBody = kwargs.extra_body
  if (extraBody && "enable_thinking" in extraBody) {
    return Boolean(extraBody.enable_thinking)
  }
  const chatKwargs = extraBody?.chat_template_kwargs as Record<string, unknown> | undefined
  if (chatKwargs && "enable_thinking" in chatKwargs) {
    return Boolean(chatKwargs.enable_thinking)
  }
  if ("think" in kwargs) {
    return Boolean(kwargs.think)
  }
  return null
}

export function supportsMaxTokens(backend: string, model: string): boolean {
  const kwargs = _lookup(backend, model).kwargs
  if ("max_tokens" in kwargs) {
    return true
  }
  const options = kwargs.options
  if (options && "num_predict" in options) {
    return true
  }
  return false
}

export function contextWindow(backend: string, model: string): number | null {
  return _lookup(backend, model).context_window
}
