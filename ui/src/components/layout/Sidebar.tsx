import { formatDistanceToNow } from "date-fns"
import { es } from "date-fns/locale"
import { MessageSquarePlus, Trash2 } from "lucide-react"
import { useNavigate, useParams } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useConversationsStore } from "@/stores/conversationsStore"

export function Sidebar() {
  const navigate = useNavigate()
  const { conversationId } = useParams<{ conversationId: string }>()
  const conversations = useConversationsStore((s) => s.conversations)
  const createConversation = useConversationsStore((s) => s.createConversation)
  const deleteConversation = useConversationsStore((s) => s.deleteConversation)

  function handleNewConversation() {
    const id = createConversation()
    navigate(`/c/${id}`)
  }

  function handleDelete(e: React.MouseEvent, id: string) {
    e.stopPropagation()
    e.preventDefault()
    const remaining = conversations.filter((c) => c.id !== id)
    deleteConversation(id)
    if (id === conversationId) {
      navigate(remaining[0] ? `/c/${remaining[0].id}` : "/")
    }
  }

  return (
    <aside className="relative z-10 flex h-full w-72 shrink-0 flex-col border-r border-white/10 bg-white/[0.03] backdrop-blur-2xl">
      <div className="flex items-center justify-between gap-2 px-4 py-4">
        <span className="text-sm font-medium tracking-tight text-foreground">
          MyAssistant
        </span>
      </div>

      <div className="px-3">
        <Button
          variant="secondary"
          className="w-full justify-start gap-2 bg-white/[0.06] hover:bg-white/[0.1]"
          onClick={handleNewConversation}
        >
          <MessageSquarePlus className="size-4" />
          Nueva conversación
        </Button>
      </div>

      <nav className="mt-3 flex-1 space-y-1 overflow-y-auto px-3 pb-4">
        {conversations.length === 0 && (
          <p className="px-2 py-6 text-center text-xs text-muted-foreground">
            Todavía no tenés conversaciones.
          </p>
        )}

        {conversations.map((conv) => {
          const isActive = conv.id === conversationId
          return (
            <button
              key={conv.id}
              type="button"
              onClick={() => navigate(`/c/${conv.id}`)}
              className={cn(
                "group flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2.5 text-left text-sm transition-colors",
                isActive
                  ? "bg-white/[0.08] text-foreground"
                  : "text-muted-foreground hover:bg-white/[0.05] hover:text-foreground"
              )}
            >
              <span className="min-w-0 flex-1">
                <span className="block truncate">{conv.title}</span>
                <span className="block truncate text-[11px] text-muted-foreground/70">
                  {formatDistanceToNow(conv.createdAt, { addSuffix: true, locale: es })}
                </span>
              </span>
              <span
                role="button"
                tabIndex={0}
                onClick={(e) => handleDelete(e, conv.id)}
                className="shrink-0 rounded-md p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-white/10 hover:text-destructive group-hover:opacity-100"
                aria-label="Eliminar conversación"
              >
                <Trash2 className="size-3.5" />
              </span>
            </button>
          )
        })}
      </nav>
    </aside>
  )
}
