import { FileCode2, Trash2, UploadCloud } from "lucide-react"
import { useRef } from "react"
import { Button } from "@/components/ui/button"
import { apiErrorMessage } from "@/lib/api/client"
import type { useTaskFiles } from "@/hooks/useTaskFiles"

// Espejo de SUPPORTED_SUFFIXES en src/context/task_files.py -- si agregás
// una extensión ahí, agregala acá también para que el selector de
// archivos del navegador no la oculte.
const ACCEPTED_EXTENSIONS =
  ".txt,.md,.json,.py,.js,.ts,.tsx,.jsx,.java,.yaml,.yml,.toml,.csv,.sql,.sh,.env,.cfg,.ini,.xml,.css,.html"

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

/**
 * Panel de archivos del modo Task -- reemplaza a <FilesSection> mientras
 * conversation.taskModeActive es true (ver RightSidebar.tsx). A
 * diferencia de FilesSection, acá no hay toggle de "colección
 * permanente": los archivos de Task nunca se indexan, solo se inyectan
 * crudos en el prompt (ver src/context/task_files.py).
 */
export function TaskFilesSection({ files }: { files: ReturnType<typeof useTaskFiles> }) {
  const inputRef = useRef<HTMLInputElement>(null)

  const fileCount = files.data?.files.length ?? 0

  function handlePickFile() {
    inputRef.current?.click()
  }

  function handleFileSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (!file) return
    files.upload.mutate(file)
  }

  return (
    <div className="space-y-3 px-3">
      <p className="text-[11px] leading-snug text-muted-foreground/70">
        Estos archivos se pasan enteros como contexto al modelo -- sin
        búsqueda ni resúmenes. Ideal para pegarle un módulo puntual a
        refactorizar.
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
        disabled={files.upload.isPending}
        onClick={handlePickFile}
      >
        <UploadCloud className="size-3.5" />
        {files.upload.isPending ? "Subiendo..." : "Subir archivo"}
      </Button>

      {files.upload.isError && (
        <p className="text-xs text-destructive">{apiErrorMessage(files.upload.error)}</p>
      )}

      {fileCount === 0 ? (
        <p className="py-2 text-center text-xs text-muted-foreground">
          Sin archivos en esta tarea todavía.
        </p>
      ) : (
        <ul className="space-y-1">
          {files.data?.files.map((f) => (
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
                aria-label={`Borrar ${f.filename}`}
                onClick={() => files.remove.mutate(f.file_id)}
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
