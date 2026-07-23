import { ChevronLeft, ChevronRight, X } from "lucide-react";
import type { ReactNode } from "react";
import { useIsDesktop } from "@/hooks/useMediaQuery";
import { useResizableWidth } from "@/hooks/useResizableWidth";
import { cn } from "@/lib/utils";
import {
  SIDEBAR_COLLAPSED_WIDTH,
  SIDEBAR_MAX_WIDTH,
  SIDEBAR_MIN_WIDTH,
} from "@/stores/uiStore";
import { ThemeToggle } from "./ThemeToggle";

export function SidebarShell({
  side,
  width,
  collapsed,
  onToggleCollapsed,
  onResize,
  mobileOpen,
  onMobileClose,
  collapsedContent,
  children,
}: {
  side: "left" | "right";
  width: number;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onResize: (width: number) => void;
  /** Drawer abierto/cerrado en mobile (< lg) -- independiente de `collapsed`, ver uiStore.ts. */
  mobileOpen: boolean;
  onMobileClose: () => void;
  /** Iconos a mostrar cuando está colapsada en desktop (cada uno debería expandir al hacer click). */
  collapsedContent: ReactNode;
  children: ReactNode;
}) {
  const isDesktop = useIsDesktop();

  const onPointerDown = useResizableWidth({
    width,
    onChange: onResize,
    min: SIDEBAR_MIN_WIDTH,
    max: SIDEBAR_MAX_WIDTH,
    growDirection: side === "left" ? 1 : -1,
  });

  const CollapseIcon = side === "left" ? ChevronLeft : ChevronRight;
  const ExpandIcon = side === "left" ? ChevronRight : ChevronLeft;

  // En mobile el panel siempre se muestra "expandido" (nunca el rail de
  // íconos colapsado: no tiene sentido dentro de un drawer a pantalla
  // completa que de por sí está oculto por completo cuando no se usa).
  const effectiveCollapsed = isDesktop && collapsed;

  return (
    <>
      {/* Backdrop del drawer -- solo existe en mobile y solo mientras está abierto. */}
      {mobileOpen && (
        <div
          aria-hidden
          onClick={onMobileClose}
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px] duration-200 animate-in fade-in lg:hidden"
        />
      )}

      <aside
        style={isDesktop ? { width: effectiveCollapsed ? SIDEBAR_COLLAPSED_WIDTH : width } : undefined}
        className={cn(
          "flex h-full shrink-0 flex-col bg-sidebar backdrop-blur-2xl transition-transform duration-300 ease-out lg:bg-overlay lg:transition-none",
          // Mobile: overlay fijo a pantalla completa (ancho propio, fuera
          // del flujo). Desktop (lg+): vuelve al comportamiento original,
          // estático dentro del flex row de AppShell.
          "fixed inset-y-0 z-50 w-[86vw] max-w-[320px] lg:static lg:z-10 lg:w-auto lg:max-w-none",
          side === "left"
            ? cn(
                "left-0 border-r border-border",
                mobileOpen ? "translate-x-0" : "-translate-x-full",
                "lg:translate-x-0"
              )
            : cn(
                "right-0 border-l border-border",
                mobileOpen ? "translate-x-0" : "translate-x-full",
                "lg:translate-x-0"
              )
        )}
      >
        <div
          className={cn(
            "flex items-center gap-1 px-2 py-2",
            side === "left" ? "justify-end" : "justify-start",
          )}
        >
          {side === "left" && !effectiveCollapsed && (
            <>
              <img className="mx-1.5 size-6" src="/favicon.svg" alt="" />
              <div className="group flex w-full items-center justify-center gap-2 rounded-lg px-3 py-4 text-left text-sm transition-colors">
                <span className="truncate text-sm font-bold text-foreground">
                  My assistant
                </span>
                <span className="truncate text-xs font-normal text-muted-foreground/80">
                  RAG by camilo6castell
                </span>
              </div>
              <ThemeToggle />
            </>
          )}

          {/* Desktop: colapsar a rail de íconos. Oculto en mobile -- ahí el
              control equivalente es cerrar el drawer por completo (botón X). */}
          <button
            type="button"
            onClick={onToggleCollapsed}
            aria-label={collapsed ? "Expand panel" : "Collapse panel"}
            className="hidden rounded-md p-1.5 text-muted-foreground hover:bg-overlay-hover hover:text-foreground lg:flex"
          >
            {collapsed ? (
              <ExpandIcon className="size-4" />
            ) : (
              <CollapseIcon className="size-4" />
            )}
          </button>

          {/* Mobile: cerrar el drawer. Oculto en desktop. */}
          <button
            type="button"
            onClick={onMobileClose}
            aria-label="Close panel"
            className="ml-auto flex rounded-md p-1.5 text-muted-foreground hover:bg-overlay-hover hover:text-foreground lg:hidden"
          >
            <X className="size-5" />
          </button>
        </div>

        {effectiveCollapsed ? (
          <div className="flex flex-1 flex-col items-center gap-1 overflow-y-auto px-1.5 pb-3">
            {side === "left" && (
              <img className="mb-1 size-6" src="/favicon.svg" alt="" />
            )}
            {collapsedContent}
            {side === "left" && <ThemeToggle className="mt-auto flex-col" />}
          </div>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col">{children}</div>
        )}

        {/* Handle de resize: franja invisible sobre todo el borde (fácil de
            agarrar), con un pill visible solo al hacer hover para indicar
            que se puede arrastrar. Solo desktop -- en mobile el ancho del
            drawer es fijo (86vw), no tiene sentido redimensionarlo. */}
        {!effectiveCollapsed && (
          <div
            onPointerDown={onPointerDown}
            className={cn(
              "group absolute top-0 hidden h-full w-2.5 cursor-col-resize touch-none select-none lg:block",
              side === "left" ? "-right-1.5" : "-left-1.5",
            )}
          >
            <div className="absolute top-1/2 left-1/2 h-12 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-transparent transition-colors group-hover:bg-primary/50" />
          </div>
        )}
      </aside>
    </>
  );
}
