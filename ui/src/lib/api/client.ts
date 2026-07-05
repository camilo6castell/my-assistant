import axios from "axios"
import type {
  CollectionsResponse,
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
 * Extrae un mensaje legible del `detail` que devuelve FastAPI en 400/404/422
 * (ver HTTPException en los routers) -- si no hay detail estructurado, cae
 * al mensaje genérico de axios/red.
 */
export function apiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = (error.response?.data as { detail?: unknown } | undefined)?.detail
    if (typeof detail === "string") return detail
    if (error.code === "ERR_NETWORK") {
      return "No se pudo conectar con el backend. ¿Está corriendo uvicorn en " +
        `${import.meta.env.VITE_API_BASE_URL}?`
    }
    return error.message
  }
  return "Ocurrió un error inesperado."
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
