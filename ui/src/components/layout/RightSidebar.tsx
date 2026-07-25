import { Layers, Paperclip, Sparkles } from "lucide-react";
import { useParams } from "react-router-dom";
import { AttachmentsSection } from "@/components/chat/AttachmentsSection";
import { FilesSection } from "@/components/chat/FilesSection";
import { CollectionsPicker } from "@/components/layout/CollectionsPicker";
import { Skeleton } from "@/components/ui/skeleton";
import { useAttachments } from "@/hooks/useAttachments";
import { useCollections } from "@/hooks/useCollections";
import { useEphemeralFiles } from "@/hooks/useEphemeralFiles";
import {
  DEMO_MODE,
  EPHEMERAL_COLLECTIONS_DEMO_EXPLANATION,
  SYSTEM_COLLECTIONS_DEMO_EXPLANATION,
} from "@/lib/demo";
import { useConversationsStore } from "@/stores/conversationsStore";
import { useUiStore } from "@/stores/uiStore";
import { SidebarSectionHeader } from "./SidebarSectionHeader";
import { SidebarShell } from "./SidebarShell";

export function RightSidebar() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const conversation = useConversationsStore((s) =>
    s.conversations.find((c) => c.id === conversationId),
  );
  const setActiveCollections = useConversationsStore(
    (s) => s.setActiveCollections,
  );

  const rightWidth = useUiStore((s) => s.rightWidth);
  const rightCollapsed = useUiStore((s) => s.rightCollapsed);
  const setRightWidth = useUiStore((s) => s.setRightWidth);
  const toggleRightCollapsed = useUiStore((s) => s.toggleRightCollapsed);
  const rightMobileOpen = useUiStore((s) => s.rightMobileOpen);
  const closeMobileSidebars = useUiStore((s) => s.closeMobileSidebars);

  const attachments = useAttachments(conversationId ?? null);
  const ephemeralFiles = useEphemeralFiles(conversationId ?? null);
  const { data: collectionsData, isLoading: collectionsLoading } =
    useCollections();

  const attachmentCount = attachments.data?.files.length ?? 0;
  const ephemeralCount = DEMO_MODE
    ? 0
    : (ephemeralFiles.data?.files.length ?? 0);
  const systemCollectionCount = DEMO_MODE
    ? 0
    : (collectionsData?.collections.length ?? 0);

  const collapsedContent = (
    <>
      <button
        type="button"
        onClick={toggleRightCollapsed}
        title="Attachments"
        className="relative flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-overlay-hover hover:text-foreground transition-colors"
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
        className="flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-overlay-hover hover:text-foreground transition-colors"
      >
        <Sparkles className="size-4" />
      </button>
      <button
        type="button"
        onClick={toggleRightCollapsed}
        title="System collections"
        className="flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-overlay-hover hover:text-foreground transition-colors"
      >
        <Layers className="size-4" />
      </button>
    </>
  );

  if (!conversation) {
    return (
      <SidebarShell
        side="right"
        width={rightWidth}
        collapsed={rightCollapsed}
        onToggleCollapsed={toggleRightCollapsed}
        onResize={setRightWidth}
        mobileOpen={rightMobileOpen}
        onMobileClose={closeMobileSidebars}
        collapsedContent={collapsedContent}
      >
        <p className="px-4 py-8 text-center text-xs text-muted-foreground/60">
          Pick or create a conversation to see its files and collections.
        </p>
      </SidebarShell>
    );
  }

  return (
    <SidebarShell
      side="right"
      width={rightWidth}
      collapsed={rightCollapsed}
      onToggleCollapsed={toggleRightCollapsed}
      onResize={setRightWidth}
      mobileOpen={rightMobileOpen}
      onMobileClose={closeMobileSidebars}
      collapsedContent={collapsedContent}
    >
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="flex min-h-0 flex-1 flex-col">
          {/* 1. Attachments */}
          <section className="flex min-h-0 flex-1 flex-col pt-1">
            <SidebarSectionHeader
              icon={Paperclip}
              label="Attachments"
              count={attachmentCount}
            />
            <div className="min-h-0 flex-1 overflow-y-auto pb-3">
              {attachments.isLoading ? (
                <ListSkeleton />
              ) : (
                <AttachmentsSection attachments={attachments} />
              )}
            </div>
          </section>

          <div className="mx-3 shrink-0 border-t border-border" />

          {/* 2. Ephemeral collections */}
          <section className="flex min-h-0 flex-1 flex-col pt-2">
            <SidebarSectionHeader
              icon={Sparkles}
              label="Ephemeral collections"
              count={ephemeralCount}
            />
            <div className="min-h-0 flex-1 overflow-y-auto pb-3">
              {DEMO_MODE ? (
                <DemoUnavailableNotice
                  text={EPHEMERAL_COLLECTIONS_DEMO_EXPLANATION}
                />
              ) : ephemeralFiles.isLoading ? (
                <ListSkeleton />
              ) : (
                <FilesSection files={ephemeralFiles} />
              )}
            </div>
          </section>
        </div>

        <div className="mx-3 shrink-0 border-t border-border" />

        {/* 3. System collections */}
        <section className="flex max-h-[50%] min-h-0 shrink-0 flex-col pt-2 pb-3">
          <SidebarSectionHeader
            icon={Layers}
            label="System collections"
            count={systemCollectionCount}
          />
          <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
            {DEMO_MODE ? (
              <DemoUnavailableNotice
                text={SYSTEM_COLLECTIONS_DEMO_EXPLANATION}
              />
            ) : collectionsLoading ? (
              <ListSkeleton />
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
  );
}

function ListSkeleton() {
  return (
    <div className="space-y-2 px-3">
      <Skeleton className="h-8 w-full" />
      <Skeleton className="h-8 w-5/6" />
    </div>
  );
}

function DemoUnavailableNotice({ text }: { text: string }) {
  return (
    <p className="mx-3 rounded-lg border border-border bg-overlay px-3 py-3 text-center text-[11px] leading-relaxed text-muted-foreground/60">
      {text}
    </p>
  );
}
