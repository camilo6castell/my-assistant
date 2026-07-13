import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  deleteTaskConversation,
  deleteTaskFile,
  listTaskFiles,
  uploadTaskFile,
} from "@/lib/api/client"

/**
 * Archivos crudos del modo Task -- ver src/context/task_files.py en el
 * backend. Mismo patrón que useEphemeralFiles.ts, pero contra
 * /api/v1/task/files en vez de /api/v1/files: acá no hay
 * attachToCollection/collection porque Task no indexa nada, solo
 * inyecta el archivo entero como texto en el prompt.
 */
export function useTaskFiles(conversationId: string | null) {
  const queryClient = useQueryClient()
  const queryKey = ["task-files", conversationId]

  const query = useQuery({
    queryKey,
    queryFn: () => listTaskFiles(conversationId!),
    enabled: conversationId !== null,
    staleTime: 10_000,
  })

  const upload = useMutation({
    mutationFn: (file: File) => uploadTaskFile({ file, conversationId: conversationId! }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  })

  const remove = useMutation({
    mutationFn: (fileId: string) => deleteTaskFile(conversationId!, fileId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  })

  const removeAll = useMutation({
    mutationFn: () => deleteTaskConversation(conversationId!),
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  })

  return { ...query, upload, remove, removeAll }
}
