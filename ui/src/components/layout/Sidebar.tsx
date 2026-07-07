import { formatDistanceToNow } from "date-fns"
import { es } from "date-fns/locale"
import { Check, Layers, MessageSquarePlus, MessagesSquare, Pencil, Trash2 } from "lucide-react"
import { useState } from "react"
import { useNavigate, useParams } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { useCollections } from "@/hooks/useCollections"
import { cn } from "@/lib/utils"
import { useConversationsStore } from "@/stores/conversationsStore"
import { useUiStore } from "@/stores/uiStore"
import { CollectionsPicker } from "./CollectionsPicker"
import { SidebarSectionHeader } from "./SidebarSectionHeader"
import { SidebarShell } from "./SidebarShell"

export function Sidebar() {
  const navigate = useNavigate()
  const { conversationId } = useParams<{ conversationId: string }>()
  const conversations = useConversationsStore((s) => s.conversations)
  const createConversation = useConversationsStore((s) => s.createConversation)
  const deleteConversation = useConversationsStore((s) => s.deleteConversation)
  const renameConversation = useConversationsStore((s) => s.renameConversation)
  const setActiveCollections = useConversationsStore((s) => s.setActiveCollections)
  const activeConversation = useConversationsStore((s) =>
    s.conversations.find((c) => c.id === conversationId)
  )

  const [editingId, setEditingId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState("")

  const leftWidth = useUiStore((s) => s.leftWidth)
  const leftCollapsed = useUiStore((s) => s.leftCollapsed)
  const setLeftWidth = useUiStore((s) => s.setLeftWidth)
  const toggleLeftCollapsed = useUiStore((s) => s.toggleLeftCollapsed)

  const { data: collectionsData } = useCollections()

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

  function handleStartEdit(e: React.MouseEvent, id: string, currentTitle: string) {
    e.stopPropagation()
    e.preventDefault()
    setEditingId(id)
    setEditValue(currentTitle)
  }

  function handleCommitEdit(id: string) {
    renameConversation(id, editValue)
    setEditingId(null)
  }

  const collapsedContent = (
    <>
      <button
        type="button"
        onClick={handleNewConversation}
        title="Nueva conversación"
        className="flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-white/10 hover:text-foreground"
      >
        <MessageSquarePlus className="size-4" />
      </button>
      <button
        type="button"
        onClick={toggleLeftCollapsed}
        title="Chats"
        className="flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-white/10 hover:text-foreground"
      >
        <MessagesSquare className="size-4" />
      </button>
      <button
        type="button"
        onClick={toggleLeftCollapsed}
        title="Colecciones"
        className="flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-white/10 hover:text-foreground"
      >
        <Layers className="size-4" />
      </button>
    </>
  )

  return (
    <SidebarShell
      side="left"
      width={leftWidth}
      collapsed={leftCollapsed}
      onToggleCollapsed={toggleLeftCollapsed}
      onResize={setLeftWidth}
      collapsedContent={collapsedContent}
    >
      <div className="px-3 pb-2">
        <Button
          variant="secondary"
          className="w-full justify-start gap-2 bg-white/[0.06] hover:bg-white/[0.1]"
          onClick={handleNewConversation}
        >
          <MessageSquarePlus className="size-4" />
          Nueva conversación
        </Button>
      </div>

      <section className="flex min-h-0 flex-grow-1 flex-col pt-1">
        <SidebarSectionHeader
          icon={MessagesSquare}
          label="Chats"
          count={conversations.length}
        />
        <nav className="min-h-0 flex-1 space-y-1 overflow-y-auto px-3 pb-2">
          {conversations.length === 0 && (
            <p className="px-2 py-4 text-center text-xs text-muted-foreground">
              Todavía no tenés conversaciones.
            </p>
          )}
          {conversations.map((conv) => {
            const isActive = conv.id === conversationId
            const isEditing = editingId === conv.id
            return (
              <div
                key={conv.id}
                role="button"
                tabIndex={0}
                onClick={() => !isEditing && navigate(`/c/${conv.id}`)}
                onKeyDown={(e) => {
                  if (!isEditing && (e.key === "Enter" || e.key === " ")) {
                    navigate(`/c/${conv.id}`)
                  }
                }}
                className={cn(
                  "group flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2.5 text-left text-sm transition-colors",
                  isActive
                    ? "bg-white/[0.08] text-foreground"
                    : "text-muted-foreground hover:bg-white/[0.05] hover:text-foreground"
                )}
              >
                <span className="min-w-0 flex-1">
                  {isEditing ? (
                    <input
                      autoFocus
                      value={editValue}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => setEditValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault()
                          handleCommitEdit(conv.id)
                        } else if (e.key === "Escape") {
                          e.preventDefault()
                          setEditingId(null)
                        }
                      }}
                      onBlur={() => handleCommitEdit(conv.id)}
                      className="block w-full rounded-md border border-white/10 bg-white/[0.06] px-1.5 py-0.5 text-sm text-foreground outline-none focus:border-primary/50"
                    />
                  ) : (
                    <span className="block truncate">{conv.title}</span>
                  )}
                  <span className="block truncate text-[11px] text-muted-foreground/70">
                    {formatDistanceToNow(conv.createdAt, { addSuffix: true, locale: es })}
                  </span>
                </span>
                <span className="flex shrink-0 items-center gap-0.5">
                  {isEditing ? (
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={(e) => {
                        e.stopPropagation()
                        e.preventDefault()
                        handleCommitEdit(conv.id)
                      }}
                      className="rounded-md p-1 text-muted-foreground hover:bg-white/10 hover:text-foreground"
                      aria-label="Guardar nombre"
                    >
                      <Check className="size-3.5" />
                    </span>
                  ) : (
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={(e) => handleStartEdit(e, conv.id, conv.title)}
                      className="rounded-md p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-white/10 hover:text-foreground group-hover:opacity-100"
                      aria-label="Renombrar conversación"
                    >
                      <Pencil className="size-3.5" />
                    </span>
                  )}
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => handleDelete(e, conv.id)}
                    className={cn(
                      "rounded-md p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-white/10 hover:text-destructive group-hover:opacity-100",
                      isEditing && "hidden"
                    )}
                    aria-label="Eliminar conversación"
                  >
                    <Trash2 className="size-3.5" />
                  </span>
                </span>
              </div>
            )
          })}
        </nav>
      </section>

      <div className="mx-3 border-t border-white/10" />

      <section className="flex min-h-0 flex-shrink-0 flex-grow-0 flex-col pt-3">
        <SidebarSectionHeader
          icon={Layers}
          label="Colecciones"
          count={collectionsData?.collections.length ?? 0}
        />
        <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-4">
          {activeConversation ? (
            <CollectionsPicker
              collections={collectionsData?.collections ?? []}
              active={activeConversation.activeCollections}
              onChange={(next) => setActiveCollections(activeConversation.id, next)}
            />
          ) : (
            <>
              <p className="px-2 pb-2 text-center text-xs text-muted-foreground">
                Elegí o creá una conversación para seleccionar colecciones.
              </p>
              <CollectionsPicker
                collections={collectionsData?.collections ?? []}
                active={[]}
                onChange={() => {}}
                disabled
              />
            </>
          )}
        </div>
      </section>
    </SidebarShell>
  )
}
