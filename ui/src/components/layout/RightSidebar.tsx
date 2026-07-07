import { Paperclip, Settings2 } from "lucide-react"
import { useParams } from "react-router-dom"
import { FilesSection } from "@/components/chat/FilesSection"
import { GenerationSection } from "@/components/chat/GenerationSection"
import { useEphemeralFiles } from "@/hooks/useEphemeralFiles"
import { useProviders } from "@/hooks/useProviders"
import { useConversationsStore } from "@/stores/conversationsStore"
import { useUiStore } from "@/stores/uiStore"
import { SidebarSectionHeader } from "./SidebarSectionHeader"
import { SidebarShell } from "./SidebarShell"

export function RightSidebar() {
  const { conversationId } = useParams<{ conversationId: string }>()
  const conversation = useConversationsStore((s) =>
    s.conversations.find((c) => c.id === conversationId)
  )

  const rightWidth = useUiStore((s) => s.rightWidth)
  const rightCollapsed = useUiStore((s) => s.rightCollapsed)
  const setRightWidth = useUiStore((s) => s.setRightWidth)
  const toggleRightCollapsed = useUiStore((s) => s.toggleRightCollapsed)

  const ephemeralFiles = useEphemeralFiles(conversationId ?? null)
  const { data: providersData } = useProviders()

  const fileCount = ephemeralFiles.data?.files.length ?? 0

  const collapsedContent = (
    <>
      <button
        type="button"
        onClick={toggleRightCollapsed}
        title="Archivos"
        className="relative flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-white/10 hover:text-foreground"
      >
        <Paperclip className="size-4" />
        {fileCount > 0 && (
          <span className="absolute -right-0.5 -top-0.5 flex size-3.5 items-center justify-center rounded-full bg-primary text-[9px] text-primary-foreground">
            {fileCount}
          </span>
        )}
      </button>
      <button
        type="button"
        onClick={toggleRightCollapsed}
        title="Generación"
        className="flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-white/10 hover:text-foreground"
      >
        <Settings2 className="size-4" />
      </button>
    </>
  )

  if (!conversation) {
    return (
      <SidebarShell
        side="right"
        width={rightWidth}
        collapsed={rightCollapsed}
        onToggleCollapsed={toggleRightCollapsed}
        onResize={setRightWidth}
        collapsedContent={collapsedContent}
      >
        <p className="px-3 py-6 text-center text-xs text-muted-foreground">
          Elegí o creá una conversación para ver sus archivos y opciones de generación.
        </p>
      </SidebarShell>
    )
  }

  return (
    <SidebarShell
      side="right"
      width={rightWidth}
      collapsed={rightCollapsed}
      onToggleCollapsed={toggleRightCollapsed}
      onResize={setRightWidth}
      collapsedContent={collapsedContent}
    >
      <section className="flex min-h-0 flex-grow-1 flex-col pt-1">
        <SidebarSectionHeader icon={Paperclip} label="Archivos" count={fileCount} />
        <div className="min-h-0 flex-1 overflow-y-auto pb-3">
          <FilesSection files={ephemeralFiles} />
        </div>
      </section>

      <div className="mx-3 border-t border-white/10" />

      <section className="flex min-h-0 flex-shrink-0 flex-grow-0 flex-col pt-3">
        <SidebarSectionHeader icon={Settings2} label="Generación" />
        <div className="min-h-0 flex-1 overflow-y-auto pb-3">
          <GenerationSection conversation={conversation} providers={providersData} />
        </div>
      </section>
    </SidebarShell>
  )
}
