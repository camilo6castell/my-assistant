import axios from "axios"
import type {
  AttachmentsResponse,
  CollectionsResponse,
  ContextLimitExceededDetail,
  DeleteResponse,
  EphemeralFilesResponse,
  FileUploadResponse,
  ProvidersResponse,
  QueryRequest,
  QueryResponse,
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
 * al mensaje genérico de axios/red. También maneja Error planos (no-axios)
 * para el modo demo (ver lib/demo.ts -- askGeminiDemo lanza Error, no pega
 * contra el backend, así que nunca hay un AxiosError que inspeccionar).
 */
export function apiErrorMessage(error: unknown): string {
  const detail = getDetail(error)
  if (typeof detail === "string") return detail
  if (detail && typeof detail === "object") {
    if (detail.message) return detail.message
    if (detail.error === "context_limit_exceeded") {
      return (
        `The message is too long for the active model ` +
        `(${detail.estimated_tokens} estimated tokens, limit ${detail.limit}). ` +
        `Shorten the message or remove an attachment.`
      )
    }
  }

  if (axios.isAxiosError(error)) {
    if (error.code === "ERR_NETWORK") {
      return (
        "Couldn't connect to the backend. Is uvicorn running at " +
        `${import.meta.env.VITE_API_BASE_URL}?`
      )
    }
    return error.message
  }
  if (error instanceof Error) return error.message
  return "An unexpected error occurred."
}

/**
 * True si el error viene de que se agotó la cuota de la cuenta de Tavily
 * (ver web_search_quota_exceeded en el detail estructurado del 422 de
 * _answer_web_only, o en el 200 normal de /query y /query/agent cuando
 * el complemento web falló por esto -- ver QueryResponse.web_search_quota_exceeded).
 * Usado en ChatView.tsx para persistir el flag en el store y que
 * ResponseModeSection.tsx deshabilite el botón "Web".
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
// Archivos adjuntos ad-hoc -- ver src/api/routers/attachments.py.
// De un solo uso: se consumen en el backend después de una query que
// los incluyó (ver _consume_attachments en src/api/routers/chat.py),
// así que la UI los vuelve a listar (queryClient.invalidateQueries)
// después de cada envío para reflejar que la lista quedó vacía.
// ---------------------------------------------------------------

export async function uploadAttachment(params: {
  file: File
  conversationId: string
}): Promise<AttachmentsResponse["files"][number]> {
  const form = new FormData()
  form.append("file", params.file)
  form.append("conversation_id", params.conversationId)

  const { data } = await api.post<AttachmentsResponse["files"][number]>(
    "/attachments",
    form,
    { headers: { "Content-Type": "multipart/form-data" } }
  )
  return data
}

export async function listAttachments(conversationId: string): Promise<AttachmentsResponse> {
  const { data } = await api.get<AttachmentsResponse>(`/attachments/${conversationId}`)
  return data
}

export async function deleteAttachment(
  conversationId: string,
  fileId: string
): Promise<DeleteResponse> {
  const { data } = await api.delete<DeleteResponse>(
    `/attachments/${conversationId}/${fileId}`
  )
  return data
}

export async function deleteAllAttachments(conversationId: string): Promise<DeleteResponse> {
  const { data } = await api.delete<DeleteResponse>(`/attachments/${conversationId}`)
  return data
}
