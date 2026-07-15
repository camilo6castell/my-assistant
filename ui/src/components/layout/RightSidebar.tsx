import { Layers, Paperclip, Sparkles } from "lucide-react"
import { useParams } from "react-router-dom"
import { AttachmentsSection } from "@/components/chat/AttachmentsSection"
import { FilesSection } from "@/components/chat/FilesSection"
import { CollectionsPicker } from "@/components/layout/CollectionsPicker"
import { useAttachments } from "@/hooks/useAttachments"
import { useCollections } from "@/hooks/useCollections"
import { useEphemeralFiles } from "@/hooks/useEphemeralFiles"
import {
  DEMO_MODE,
  EPHEMERAL_COLLECTIONS_DEMO_EXPLANATION,
  SYSTEM_COLLECTIONS_DEMO_EXPLANATION,
} from "@/lib/demo"
import { useConversationsStore } from "@/stores/conversationsStore"
import { useUiStore } from "@/stores/uiStore"
import { SidebarSectionHeader } from "./SidebarSectionHeader"
import { SidebarShell } from "./SidebarShell"

/**
 * Sidebar derecho: todo lo relacionado con CONOCIMIENTO, de más
 * puntual a más permanente --
 *
 *   1. Archivos adjuntos   -- ad-hoc, de un solo envío (ver
 *                              src/context/attachments.py)
 *   2. Colecciones efímeras -- indexadas, viven mientras dure la
 *                              conversación (ver src/context/ephemeral.py)
 *   3. Colecciones de sistema -- indexadas con `python -m src.ingest`,
 *                              permanentes, compartidas entre conversaciones
 *
 * El sidebar izquierdo (ver LeftSidebar.tsx) es todo lo relacionado con
 * la CONVERSACIÓN en sí: historial de chats y modo de respuesta.
 *
 * No hay una sección de "ajustes avanzados" (Turnos de historial / Top
 * K): esos parámetros pasaron a ser exclusivamente de .env (ver
 * GenerationOptions en src/api/schemas/chat.py) -- sin override por
 * conversación, así que no hay nada que la UI necesite mostrar o dejar
 * tocar acá.
 *
 * En demo mode (ver lib/demo.ts), las secciones 2 y 3 dependen del
 * pipeline de indexado local (embeddings + FAISS) que no existe en un
 * deploy estático sin backend -- se mantiene el header de cada una
 * (nombre + ícono) pero el contenido se reemplaza por un aviso
 * explicando por qué, en vez de intentar listar algo que no puede
 * existir acá. La sección 1 (adjuntos) sigue funcionando igual: nunca
 * se indexó, siempre fue texto crudo inyectado en el prompt -- ver
 * useAttachments.ts.
 */
export function RightSidebar() {
  const { conversationId } = useParams<{ conversationId: string }>()
  const conversation = useConversationsStore((s) =>
    s.conversations.find((c) => c.id === conversationId)
  )
  const setActiveCollections = useConversationsStore((s) => s.setActiveCollections)

  const rightWidth = useUiStore((s) => s.rightWidth)
  const rightCollapsed = useUiStore((s) => s.rightCollapsed)
  const setRightWidth = useUiStore((s) => s.setRightWidth)
  const toggleRightCollapsed = useUiStore((s) => s.toggleRightCollapsed)

  const attachments = useAttachments(conversationId ?? null)
  const ephemeralFiles = useEphemeralFiles(conversationId ?? null)
  const { data: collectionsData } = useCollections()

  const attachmentCount = attachments.data?.files.length ?? 0
  const ephemeralCount = DEMO_MODE ? 0 : (ephemeralFiles.data?.files.length ?? 0)
  const systemCollectionCount = DEMO_MODE ? 0 : (collectionsData?.collections.length ?? 0)

  const collapsedContent = (
    <>
      <button
        type="button"
        onClick={toggleRightCollapsed}
        title="Attachments"
        className="relative flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-white/10 hover:text-foreground"
      >
        <Paperclip className="size-4" />
        {attachmentCount > 0 && (
          <span className="absolute -right-0.5 -top-0.5 flex size-3.5 items-center justify-center rounded-full bg-primary text-[9px] text-primary-foreground">
            {attachmentCount}
          </span>
        )}
      </button>
      <button
        type="button"
        onClick={toggleRightCollapsed}
        title="Ephemeral collections"
        className="flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-white/10 hover:text-foreground"
      >
        <Sparkles className="size-4" />
      </button>
      <button
        type="button"
        onClick={toggleRightCollapsed}
        title="System collections"
        className="flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-white/10 hover:text-foreground"
      >
        <Layers className="size-4" />
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
          Pick or create a conversation to see its files and collections.
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
      <div className="flex min-h-0 flex-1 flex-col">
        {/* Bloque superior: secciones 1 y 2 se reparten el espacio restante al 50/50 */}
        <div className="flex min-h-0 flex-1 flex-col">
          {/* 1. Archivos adjuntos -- ad-hoc, de un solo envío. Funciona igual en demo mode. */}
          <section className="flex min-h-0 flex-1 flex-col pt-2">
            <SidebarSectionHeader icon={Paperclip} label="Attachments" count={attachmentCount} />
            <div className="min-h-0 flex-1 overflow-y-auto pb-3">
              <AttachmentsSection attachments={attachments} />
            </div>
          </section>

          <div className="mx-3 shrink-0 border-t border-white/10" />

          {/* 2. Colecciones efímeras -- indexadas, alcance de esta conversación */}
          <section className="flex min-h-0 flex-1 flex-col pt-3">
            <SidebarSectionHeader
              icon={Sparkles}
              label="Ephemeral collections"
              count={ephemeralCount}
            />
            <div className="min-h-0 flex-1 overflow-y-auto pb-3">
              {DEMO_MODE ? (
                <DemoUnavailableNotice text={EPHEMERAL_COLLECTIONS_DEMO_EXPLANATION} />
              ) : (
                <FilesSection files={ephemeralFiles} />
              )}
            </div>
          </section>
        </div>

        <div className="mx-3 shrink-0 border-t border-white/10" />

        {/* 3. Colecciones de sistema -- pegada abajo, máx 50% de altura, scroll propio */}
        <section className="flex max-h-[50%] min-h-0 shrink-0 flex-col pt-3 pb-3">
          <SidebarSectionHeader
            icon={Layers}
            label="System collections"
            count={systemCollectionCount}
          />
          <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
            {DEMO_MODE ? (
              <DemoUnavailableNotice text={SYSTEM_COLLECTIONS_DEMO_EXPLANATION} />
            ) : (
              <CollectionsPicker
                collections={collectionsData?.collections ?? []}
                active={conversation.activeCollections}
                onChange={(next) => setActiveCollections(conversation.id, next)}
              />
            )}
          </div>
        </section>
      </div>
    </SidebarShell>
  )
}

function DemoUnavailableNotice({ text }: { text: string }) {
  return (
    <p className="rounded-lg border border-white/10 bg-white/[0.02] px-3 py-3 text-center text-[11px] leading-relaxed text-muted-foreground">
      {text}
    </p>
  )
}
