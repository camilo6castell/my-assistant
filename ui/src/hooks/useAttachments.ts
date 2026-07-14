import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  deleteAllAttachments,
  deleteAttachment,
  listAttachments,
  uploadAttachment,
} from "@/lib/api/client"

/**
 * Archivos adjuntos ad-hoc de una conversación -- ver
 * src/context/attachments.py en el backend. Mismo patrón que
 * useEphemeralFiles.ts, pero contra /api/v1/attachments: acá no hay
 * attachToCollection/collection porque los adjuntos nunca se indexan,
 * solo se inyectan enteros como texto en la próxima query y se
 * consumen automáticamente después de enviarla (ver
 * _consume_attachments en src/api/routers/chat.py) -- por eso
 * ChatView.tsx invalida esta query después de cada envío exitoso.
 */
export function useAttachments(conversationId: string | null) {
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
