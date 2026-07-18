import { Trash2, UploadCloud } from "lucide-react"
import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { apiErrorMessage } from "@/lib/api/client"
import type { useEphemeralFiles } from "@/hooks/useEphemeralFiles"

export function FilesSection({ files }: { files: ReturnType<typeof useEphemeralFiles> }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [attachToCollection, setAttachToCollection] = useState(false)
  const [collectionName, setCollectionName] = useState("")

  const fileCount = files.data?.files.length ?? 0

  function handlePickFile() {
    inputRef.current?.click()
  }

  function handleFileSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (!file) return
    files.upload.mutate({
      file,
      attachToCollection,
      collection: attachToCollection ? collectionName.trim() : undefined,
    })
  }

  return (
    <div className="space-y-3 px-3">
      <label className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Attach to permanent collection</span>
        <Switch checked={attachToCollection} onCheckedChange={setAttachToCollection} />
      </label>

      {attachToCollection && (
        <input
          value={collectionName}
          onChange={(e) => setCollectionName(e.target.value)}
          placeholder="namespace/collection"
          className="w-full rounded-lg border border-border bg-overlay px-2.5 py-1.5 text-xs outline-none placeholder:text-muted-foreground focus-visible:border-ring"
        />
      )}

      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.html,.txt"
        className="hidden"
        onChange={handleFileSelected}
      />
      <Button
        variant="secondary"
        size="sm"
        className="w-full gap-1.5"
        disabled={files.upload.isPending || (attachToCollection && !collectionName.trim())}
        onClick={handlePickFile}
      >
        <UploadCloud className="size-3.5" />
        {files.upload.isPending ? "Uploading..." : "Upload file"}
      </Button>

      {files.upload.isError && (
        <p className="text-xs text-destructive">{apiErrorMessage(files.upload.error)}</p>
      )}

      {fileCount === 0 ? (
        <p className="py-2 text-center text-xs text-muted-foreground">
          No ephemeral files in this conversation.
        </p>
      ) : (
        <ul className="space-y-1">
          {files.data?.files.map((f) => (
            <li
              key={f.file_id}
              className="flex items-center justify-between gap-2 rounded-md px-2 py-1.5 hover:bg-overlay-hover"
            >
              <span className="min-w-0 truncate text-xs">
                {f.filename}
                <span className="ml-1 text-muted-foreground">({f.chunk_count} chunks)</span>
              </span>
              <button
                type="button"
                aria-label={`Delete ${f.filename}`}
                onClick={() => files.remove.mutate(f.file_id)}
                className="shrink-0 rounded p-1 text-muted-foreground hover:bg-overlay-hover hover:text-destructive"
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
