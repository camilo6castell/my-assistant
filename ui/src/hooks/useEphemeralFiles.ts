import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  deleteEphemeralConversation,
  deleteEphemeralFile,
  listEphemeralFiles,
  uploadFile,
} from "@/lib/api/client"

export function useEphemeralFiles(conversationId: string | null) {
  const queryClient = useQueryClient()
  const queryKey = ["ephemeral-files", conversationId]

  const query = useQuery({
    queryKey,
    queryFn: () => listEphemeralFiles(conversationId!),
    enabled: conversationId !== null,
    staleTime: 10_000,
  })

  const upload = useMutation({
    mutationFn: (params: { file: File; attachToCollection: boolean; collection?: string }) =>
      uploadFile({ ...params, conversationId: conversationId! }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  })

  const remove = useMutation({
    mutationFn: (fileId: string) => deleteEphemeralFile(conversationId!, fileId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  })

  const removeAll = useMutation({
    mutationFn: () => deleteEphemeralConversation(conversationId!),
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  })

  return { ...query, upload, remove, removeAll }
}
