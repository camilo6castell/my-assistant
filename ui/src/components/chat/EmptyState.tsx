import { Sparkles } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { useConversationsStore } from "@/stores/conversationsStore"

export function EmptyState() {
  const navigate = useNavigate()
  const createConversation = useConversationsStore((s) => s.createConversation)

  function handleStart() {
    const id = createConversation()
    navigate(`/c/${id}`)
  }

  return (
    <div className="flex h-full flex-1 flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="flex size-14 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.05] backdrop-blur-xl">
        <Sparkles className="size-6 text-primary" />
      </div>
      <div className="space-y-1">
        <h1 className="text-lg font-medium text-foreground">Tu asistente RAG</h1>
        <p className="max-w-sm text-sm text-muted-foreground">
          Elegí una conversación existente o empezá una nueva para consultar
          tus colecciones.
        </p>
      </div>
      <Button onClick={handleStart} className="mt-2">
        Nueva conversación
      </Button>
    </div>
  )
}
