import axios from "axios"
import type {
  CollectionsResponse,
  ContextLimitExceededDetail,
  DeleteResponse,
  EphemeralFilesResponse,
  FileUploadResponse,
  ProvidersResponse,
  QueryRequest,
  QueryResponse,
  TaskFilesResponse,
  TaskRequest,
  TaskResponse,
} from "@/types/api"

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
})

/**
 * Forma del `detail` cuando el backend necesita comunicar algo más que un
 * mensaje de texto (ver _no_context_detail en src/api/routers/chat.py).
 * Por ahora el único caso es "se agotó la cuota de Tavily", pero queda
 * genérico por si aparece otra señal estructurada más adelante.
 */
interface StructuredErrorDetail {
  message?: string
  web_search_quota_exceeded?: boolean
  /** Ver ContextLimitExceededDetail en types/api.ts -- mismo `detail`, forma más específica. */
  error?: string
  estimated_tokens?: number
  limit?: number
  model?: string
}

function getDetail(error: unknown): string | StructuredErrorDetail | undefined {
  if (!axios.isAxiosError(error)) return undefined
  return (error.response?.data as { detail?: string | StructuredErrorDetail } | undefined)
    ?.detail
}

/**
 * Extrae un mensaje legible del `detail` que devuelve FastAPI en 400/404/422
 * (ver HTTPException en los routers) -- si no hay detail estructurado, cae
 * al mensaje genérico de axios/red.
 */
export function apiErrorMessage(error: unknown): string {
  const detail = getDetail(error)
  if (typeof detail === "string") return detail
  if (detail && typeof detail === "object") {
    if (detail.message) return detail.message
    if (detail.error === "context_limit_exceeded") {
      return (
        `El mensaje es demasiado largo para el modelo activo ` +
        `(${detail.estimated_tokens} tokens estimados, límite ${detail.limit}). ` +
        `Acortá el mensaje o quitá algún archivo adjunto.`
      )
    }
  }

  if (axios.isAxiosError(error)) {
    if (error.code === "ERR_NETWORK") {
      return "No se pudo conectar con el backend. ¿Está corriendo uvicorn en " +
        `${import.meta.env.VITE_API_BASE_URL}?`
    }
    return error.message
  }
  return "Ocurrió un error inesperado."
}

/**
 * True si el error viene de que se agotó la cuota de la cuenta de Tavily
 * (ver web_search_quota_exceeded en el detail estructurado del 422 de
 * _answer_web_only, o en el 200 normal de /query y /query/agent cuando
 * el complemento web falló por esto -- ver QueryResponse.web_search_quota_exceeded).
 * Usado en ChatView.tsx para persistir el flag en el store y que
 * GenerationSection.tsx deshabilite el botón "Web".
 */
export function isWebSearchQuotaExceededError(error: unknown): boolean {
  const detail = getDetail(error)
  return typeof detail === "object" && detail?.web_search_quota_exceeded === true
}

/**
 * Detalle del 413 cuando el request no entra en la ventana de contexto
 * del modelo activo (ver check_context_fit en src/llm/context_guard.py).
 * undefined si el error no es de este tipo. Usado en ChatView.tsx para
 * mostrar un banner específico en vez del mensaje genérico de axios.
 */
export function getContextLimitDetail(error: unknown): ContextLimitExceededDetail | undefined {
  const detail = getDetail(error)
  if (
    typeof detail === "object" &&
    detail !== null &&
    "error" in detail &&
    (detail as { error?: string }).error === "context_limit_exceeded"
  ) {
    return detail as unknown as ContextLimitExceededDetail
  }
  return undefined
}

export async function getCollections(): Promise<CollectionsResponse> {
  const { data } = await api.get<CollectionsResponse>("/collections")
  return data
}

export async function getProviders(): Promise<ProvidersResponse> {
  const { data } = await api.get<ProvidersResponse>("/config/providers")
  return data
}

export async function postQuery(
  payload: QueryRequest,
  useAgent: boolean
): Promise<QueryResponse> {
  const { data } = await api.post<QueryResponse>(
    useAgent ? "/query/agent" : "/query",
    payload
  )
  return data
}

export async function uploadFile(params: {
  file: File
  conversationId: string
  attachToCollection: boolean
  collection?: string
}): Promise<FileUploadResponse> {
  const form = new FormData()
  form.append("file", params.file)
  form.append("conversation_id", params.conversationId)
  form.append("attach_to_collection", String(params.attachToCollection))
  if (params.collection) form.append("collection", params.collection)

  const { data } = await api.post<FileUploadResponse>("/files", form, {
    headers: { "Content-Type": "multipart/form-data" },
  })
  return data
}

export async function listEphemeralFiles(
  conversationId: string
): Promise<EphemeralFilesResponse> {
  const { data } = await api.get<EphemeralFilesResponse>(`/files/${conversationId}`)
  return data
}

export async function deleteEphemeralFile(
  conversationId: string,
  fileId: string
): Promise<DeleteResponse> {
  const { data } = await api.delete<DeleteResponse>(
    `/files/${conversationId}/${fileId}`
  )
  return data
}

export async function deleteEphemeralConversation(
  conversationId: string
): Promise<DeleteResponse> {
  const { data } = await api.delete<DeleteResponse>(`/files/${conversationId}`)
  return data
}

// ---------------------------------------------------------------
// Modo Task -- ver src/api/routers/task.py
// ---------------------------------------------------------------

export async function postTaskQuery(payload: TaskRequest): Promise<TaskResponse> {
  const { data } = await api.post<TaskResponse>("/task/query", payload)
  return data
}

export async function uploadTaskFile(params: {
  file: File
  conversationId: string
}): Promise<TaskFilesResponse["files"][number]> {
  const form = new FormData()
  form.append("file", params.file)
  form.append("conversation_id", params.conversationId)

  const { data } = await api.post<TaskFilesResponse["files"][number]>(
    "/task/files",
    form,
    { headers: { "Content-Type": "multipart/form-data" } }
  )
  return data
}

export async function listTaskFiles(conversationId: string): Promise<TaskFilesResponse> {
  const { data } = await api.get<TaskFilesResponse>(`/task/files/${conversationId}`)
  return data
}

export async function deleteTaskFile(
  conversationId: string,
  fileId: string
): Promise<DeleteResponse> {
  const { data } = await api.delete<DeleteResponse>(
    `/task/files/${conversationId}/${fileId}`
  )
  return data
}

export async function deleteTaskConversation(conversationId: string): Promise<DeleteResponse> {
  const { data } = await api.delete<DeleteResponse>(`/task/files/${conversationId}`)
  return data
}
