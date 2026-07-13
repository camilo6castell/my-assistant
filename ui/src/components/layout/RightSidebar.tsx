import { Paperclip, Settings2, Wrench } from "lucide-react";
import { useParams } from "react-router-dom";
import { FilesSection } from "@/components/chat/FilesSection";
import { GenerationSection } from "@/components/chat/GenerationSection";
import { TaskFilesSection } from "@/components/chat/TaskFilesSection";
import { cn } from "@/lib/utils";
import { useEphemeralFiles } from "@/hooks/useEphemeralFiles";
import { useProviders } from "@/hooks/useProviders";
import { useTaskFiles } from "@/hooks/useTaskFiles";
import { useConversationsStore } from "@/stores/conversationsStore";
import { useUiStore } from "@/stores/uiStore";
import { SidebarSectionHeader } from "./SidebarSectionHeader";
import { SidebarShell } from "./SidebarShell";

const TASK_MODE_EXPLANATION =
  "Modo Task: tareas de desarrollo puntuales (funciones, refactors, " +
  "estructuras de datos) sin usar tus colecciones -- el modelo solo ve " +
  "los archivos que adjuntes acá. Mientras está activo, las colecciones " +
  "y las opciones de generación quedan deshabilitadas: son dos flujos " +
  "independientes.";

export function RightSidebar() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const conversation = useConversationsStore((s) =>
    s.conversations.find((c) => c.id === conversationId),
  );

  const rightWidth = useUiStore((s) => s.rightWidth);
  const rightCollapsed = useUiStore((s) => s.rightCollapsed);
  const setRightWidth = useUiStore((s) => s.setRightWidth);
  const toggleRightCollapsed = useUiStore((s) => s.toggleRightCollapsed);

  const ephemeralFiles = useEphemeralFiles(conversationId ?? null);
  const taskFiles = useTaskFiles(conversationId ?? null);
  const setTaskModeActive = useConversationsStore((s) => s.setTaskModeActive);
  const { data: providersData } = useProviders();

  const taskModeActive = conversation?.taskModeActive ?? false;
  const fileCount = taskModeActive
    ? (taskFiles.data?.files.length ?? 0)
    : (ephemeralFiles.data?.files.length ?? 0);

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
  );

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
          Elegí o creá una conversación para ver sus archivos y opciones de
          generación.
        </p>
      </SidebarShell>
    );
  }

  const activeProvider = providersData
    ? providersData.providers[providersData.active_generation_provider]
    : undefined;

  return (
    <SidebarShell
      side="right"
      width={rightWidth}
      collapsed={rightCollapsed}
      onToggleCollapsed={toggleRightCollapsed}
      onResize={setRightWidth}
      collapsedContent={collapsedContent}
    >
      <div className="px-3 pt-2">
        <button
          type="button"
          onClick={() => setTaskModeActive(conversation.id, !taskModeActive)}
          title={TASK_MODE_EXPLANATION}
          aria-pressed={taskModeActive}
          className={cn(
            "flex w-full items-center justify-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium transition-colors",
            taskModeActive
              ? "border-primary/40 bg-primary/15 text-primary"
              : "border-white/10 text-muted-foreground hover:bg-white/5 hover:text-foreground",
          )}
        >
          <Wrench className="size-4" />
          Task
        </button>
      </div>

      <section className="flex min-h-0 flex-grow-1 flex-col pt-1">
        <SidebarSectionHeader
          icon={Paperclip}
          label={taskModeActive ? "Archivos de la tarea" : "Archivos"}
          count={fileCount}
        />
        <div className="min-h-0 flex-1 overflow-y-auto pb-3">
          {taskModeActive ? (
            <TaskFilesSection files={taskFiles} />
          ) : (
            <FilesSection files={ephemeralFiles} />
          )}
        </div>
      </section>

      <div className="mx-3 border-t border-white/10" />

      <section className="flex min-h-0 flex-shrink-0 flex-grow-0 flex-col pt-3">
        <SidebarSectionHeader
          icon={Settings2}
          label="Generación"
          count={activeProvider?.name + " · " + activeProvider?.model}
        />
        <div className="min-h-0 flex-1 overflow-y-auto pb-3">
          <GenerationSection
            conversation={conversation}
            providers={providersData}
            disabled={taskModeActive}
          />
        </div>
      </section>
    </SidebarShell>
  );
}
