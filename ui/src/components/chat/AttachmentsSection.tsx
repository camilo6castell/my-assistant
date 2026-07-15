import { FileCode2, Trash2, UploadCloud } from "lucide-react"
import { useRef } from "react"
import { Button } from "@/components/ui/button"
import { apiErrorMessage } from "@/lib/api/client"
import type { useAttachments } from "@/hooks/useAttachments"

// Espejo de SUPPORTED_SUFFIXES en src/context/attachments.py -- si
// agregás una extensión ahí, agregala acá también para que el selector
// de archivos del navegador no la oculte.
const ACCEPTED_EXTENSIONS =
  ".txt,.md,.json,.py,.js,.ts,.tsx,.jsx,.java,.yaml,.yml,.toml,.csv,.sql,.sh,.env,.cfg,.ini,.xml,.css,.html"

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

/**
 * Archivos que se inyectan enteros en la PRÓXIMA query -- ver
 * src/context/attachments.py. Se aplican sin importar el modo de
 * respuesta: con colecciones activas, sin ninguna (ver _answer_raw en
 * el backend), o con búsqueda web. Una vez enviados, el backend los
 * consume (los borra) y esta lista vuelve a estar vacía -- por eso
 * ChatView.tsx invalida la query de useAttachments después de cada
 * envío exitoso.
 */
export function AttachmentsSection({
  attachments,
}: {
  attachments: ReturnType<typeof useAttachments>
}) {
  const inputRef = useRef<HTMLInputElement>(null)

  const fileCount = attachments.data?.files.length ?? 0

  function handlePickFile() {
    inputRef.current?.click()
  }

  function handleFileSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (!file) return
    attachments.upload.mutate(file)
  }

  return (
    <div className="space-y-3 px-3">
      <p className="text-[11px] leading-snug text-muted-foreground/70">
        Sent whole with your next message -- no search, no summaries.
        Once sent, they're no longer attached.
      </p>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS}
        className="hidden"
        onChange={handleFileSelected}
      />
      <Button
        variant="secondary"
        size="sm"
        className="w-full gap-1.5"
        disabled={attachments.upload.isPending}
        onClick={handlePickFile}
      >
        <UploadCloud className="size-3.5" />
        {attachments.upload.isPending ? "Uploading..." : "Attach file"}
      </Button>

      {attachments.upload.isError && (
        <p className="text-xs text-destructive">
          {apiErrorMessage(attachments.upload.error)}
        </p>
      )}

      {fileCount === 0 ? (
        <p className="py-2 text-center text-xs text-muted-foreground">
          No pending attachments.
        </p>
      ) : (
        <ul className="space-y-1">
          {attachments.data?.files.map((f) => (
            <li
              key={f.file_id}
              className="flex items-center justify-between gap-2 rounded-md px-2 py-1.5 hover:bg-white/5"
            >
              <span className="flex min-w-0 items-center gap-1.5 truncate text-xs">
                <FileCode2 className="size-3.5 shrink-0 text-muted-foreground" />
                <span className="truncate">{f.filename}</span>
                <span className="shrink-0 text-muted-foreground">
                  ({formatSize(f.size_bytes)})
                </span>
              </span>
              <button
                type="button"
                aria-label={`Remove ${f.filename}`}
                onClick={() => attachments.remove.mutate(f.file_id)}
                className="shrink-0 rounded p-1 text-muted-foreground hover:bg-white/10 hover:text-destructive"
              >
                <Trash2 className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
