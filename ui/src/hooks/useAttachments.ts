import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { nanoid } from "nanoid"
import {
  deleteAllAttachments,
  deleteAttachment,
  listAttachments,
  uploadAttachment,
} from "@/lib/api/client"
import { DEMO_MODE } from "@/lib/demo"
import { useDemoAttachmentsStore, selectDemoFiles } from "@/stores/demoAttachmentsStore"

/**
 * Adjuntos ad-hoc de una conversación -- ver src/context/attachments.py
 * en el backend. Mismo patrón que useEphemeralFiles.ts, pero contra
 * /api/v1/attachments: acá no hay attachToCollection/collection porque
 * los adjuntos nunca se indexan, solo se inyectan enteros como texto en
 * la próxima query y se consumen automáticamente después de enviarla
 * (ver _consume_attachments en src/api/routers/chat.py) -- por eso
 * ChatView.tsx invalida esta query después de cada envío exitoso.
 *
 * En VITE_DEMO_MODE, no hay backend al que subir nada: useDemoAttachments
 * más abajo reimplementa el mismo contrato (data.files, upload.mutate,
 * remove.mutate) leyendo el archivo con FileReader y guardándolo en
 * memoria (useDemoAttachmentsStore) -- ver lib/demo.ts para el porqué.
 * Como nunca se indexaban (siempre fue texto crudo inyectado en el
 * prompt), esto es funcionalmente equivalente, solo que el "backend" es
 * la pestaña del navegador.
 */
function useRealAttachments(conversationId: string | null) {
  const queryClient = useQueryClient()
  const queryKey = ["attachments", conversationId]

  const query = useQuery({
    queryKey,
    queryFn: () => listAttachments(conversationId!),
    enabled: conversationId !== null,
    staleTime: 10_000,
  })

  const upload = useMutation({
    mutationFn: (file: File) => uploadAttachment({ file, conversationId: conversationId! }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  })

  const remove = useMutation({
    mutationFn: (fileId: string) => deleteAttachment(conversationId!, fileId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  })

  const removeAll = useMutation({
    mutationFn: () => deleteAllAttachments(conversationId!),
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  })

  return { ...query, upload, remove, removeAll }
}

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result ?? ""))
    reader.onerror = () => reject(reader.error ?? new Error("Couldn't read the file."))
    reader.readAsText(file)
  })
}

function useDemoAttachments(conversationId: string | null) {
  const files = useDemoAttachmentsStore(selectDemoFiles(conversationId))
  const addFile = useDemoAttachmentsStore((s) => s.addFile)
  const removeFile = useDemoAttachmentsStore((s) => s.removeFile)
  const clearFiles = useDemoAttachmentsStore((s) => s.clearFiles)

  const upload = useMutation({
    mutationFn: async (file: File) => {
      const content = await readFileAsText(file)
      addFile(conversationId!, {
        file_id: nanoid(),
        filename: file.name,
        size_bytes: file.size,
        uploaded_at: new Date().toISOString(),
        content,
      })
    },
  })

  const remove = useMutation({
    mutationFn: (fileId: string) => {
      removeFile(conversationId!, fileId)
      return Promise.resolve()
    },
  })

  const removeAll = useMutation({
    mutationFn: () => {
      clearFiles(conversationId!)
      return Promise.resolve()
    },
  })

  return {
    data: { files },
    isLoading: false as const,
    isPending: false as const,
    isError: false as const,
    error: null,
    upload,
    remove,
    removeAll,
  }
}

export function useAttachments(conversationId: string | null) {
  // DEMO_MODE es una constante de build time (import.meta.env), nunca
  // cambia entre renders para una misma build -- por eso ramificar acá
  // qué hook llamar es seguro pese a que parezca romper rules-of-hooks.
  // eslint-disable-next-line react-hooks/rules-of-hooks
  return DEMO_MODE ? useDemoAttachments(conversationId) : useRealAttachments(conversationId)
}
