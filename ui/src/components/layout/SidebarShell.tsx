import { ChevronLeft, ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import { useResizableWidth } from "@/hooks/useResizableWidth";
import { cn } from "@/lib/utils";
import {
  SIDEBAR_COLLAPSED_WIDTH,
  SIDEBAR_MAX_WIDTH,
  SIDEBAR_MIN_WIDTH,
} from "@/stores/uiStore";

export function SidebarShell({
  side,
  width,
  collapsed,
  onToggleCollapsed,
  onResize,
  collapsedContent,
  children,
}: {
  side: "left" | "right";
  width: number;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onResize: (width: number) => void;
  /** Iconos a mostrar cuando está colapsada (cada uno debería expandir al hacer click). */
  collapsedContent: ReactNode;
  children: ReactNode;
}) {
  const onPointerDown = useResizableWidth({
    width,
    onChange: onResize,
    min: SIDEBAR_MIN_WIDTH,
    max: SIDEBAR_MAX_WIDTH,
    growDirection: side === "left" ? 1 : -1,
  });

  const CollapseIcon = side === "left" ? ChevronLeft : ChevronRight;
  const ExpandIcon = side === "left" ? ChevronRight : ChevronLeft;

  return (
    <aside
      style={{ width: collapsed ? SIDEBAR_COLLAPSED_WIDTH : width }}
      className={cn(
        "relative z-10 flex h-full shrink-0 flex-col bg-white/[0.03] backdrop-blur-2xl",
        side === "left"
          ? "border-r border-white/10"
          : "border-l border-white/10",
      )}
    >
      <div
        className={cn(
          "flex items-center px-2 py-2",
          side === "left" ? "justify-end " : "justify-start",
        )}
      >
        {side === "left" && (
          <>
            <img className="w-6 h-6 mx-1.5" src="/favicon.svg" alt="" />
            <div className="group flex w-full items-center justify-center gap-2 rounded-lg px-3 py-4 text-left text-sm transition-colors">
              <span className="font-bold text-foreground truncate text-sm">
                My assistant
              </span>
              <span className="text-xs font-normal text-muted-foreground/80 truncate">
                RAG by camilo6castell
              </span>
            </div>
          </>
        )}

        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-label={collapsed ? "Expandir panel" : "Colapsar panel"}
          className="rounded-md p-1.5 text-muted-foreground hover:bg-white/10 hover:text-foreground"
        >
          {collapsed ? (
            <ExpandIcon className="size-4" />
          ) : (
            <CollapseIcon className="size-4" />
          )}
        </button>
      </div>

      {collapsed ? (
        <div className="flex flex-1 flex-col items-center gap-1 overflow-y-auto px-1.5 pb-3">
          {collapsedContent}
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">{children}</div>
      )}

      {/* Handle de resize: franja invisible sobre todo el borde (fácil de
          agarrar), con un pill visible solo al hacer hover para indicar
          que se puede arrastrar. Oculto cuando está colapsada. */}
      {!collapsed && (
        <div
          onPointerDown={onPointerDown}
          className={cn(
            "group absolute top-0 h-full w-2.5 cursor-col-resize touch-none select-none",
            side === "left" ? "-right-1.5" : "-left-1.5",
          )}
        >
          <div className="absolute top-1/2 left-1/2 h-12 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-transparent transition-colors group-hover:bg-primary/50" />
        </div>
      )}
    </aside>
  );
}
