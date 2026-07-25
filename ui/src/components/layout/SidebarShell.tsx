import { ChevronLeft, ChevronRight, X } from "lucide-react";
import type { ReactNode } from "react";
import { LogoIcon } from "./LogoIcon";
import { useIsDesktop } from "@/hooks/useMediaQuery";
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
  mobileOpen: boolean;
  onMobileClose: () => void;
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

  const effectiveCollapsed = isDesktop && collapsed;

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          aria-hidden
          onClick={onMobileClose}
          className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm duration-200 animate-in fade-in lg:hidden"
        />
      )}

      <aside
        style={
          isDesktop
            ? { width: effectiveCollapsed ? SIDEBAR_COLLAPSED_WIDTH : width }
            : undefined
        }
        className={cn(
          "flex h-full shrink-0 flex-col bg-sidebar transition-all duration-200 ease-out",
          "fixed inset-y-0 z-50 w-[86vw] max-w-[320px] lg:static lg:z-10 lg:w-auto lg:max-w-none lg:transition-none",
          side === "left"
            ? cn(
                "left-0 border-r border-border",
                mobileOpen ? "translate-x-0" : "-translate-x-full",
                "lg:translate-x-0",
              )
            : cn(
                "right-0 border-l border-border",
                mobileOpen ? "translate-x-0" : "translate-x-full",
                "lg:translate-x-0",
              ),
        )}
      >
        {/* Sidebar header */}
        <div
          className={cn(
            "flex shrink-0 items-center px-3 py-2.5",
            side === "left" ? "justify-between" : "justify-end",
          )}
        >
          {side === "left" && !effectiveCollapsed && (
            <a
              className="flex w-full items-center justify-center gap-2 rounded-lg py-1 text-left transition-opacity hover:opacity-80"
              href="https://github.com/camilo6castell"
              target="_blank"
              rel="noopener noreferrer"
            >
              <LogoIcon className="size-5 shrink-0" />
              <span className="text-sm font-semibold tracking-tight text-foreground">
                My assistant
              </span>
            </a>
          )}

          {/* Desktop collapse toggle */}
          <button
            type="button"
            onClick={onToggleCollapsed}
            aria-label={collapsed ? "Expand panel" : "Collapse panel"}
            className="hidden rounded-lg p-1.5 text-muted-foreground hover:bg-overlay-hover hover:text-foreground transition-colors lg:flex mr-auto"
          >
            {collapsed ? (
              <ExpandIcon className="size-4" />
            ) : (
              <CollapseIcon className="size-4" />
            )}
          </button>

          {/* Mobile close */}
          <button
            type="button"
            onClick={onMobileClose}
            aria-label="Close panel"
            className="flex rounded-lg p-1.5 text-muted-foreground hover:bg-overlay-hover hover:text-foreground transition-colors lg:hidden"
          >
            <X className="size-5" />
          </button>
        </div>

        {effectiveCollapsed ? (
          <div className="flex flex-1 flex-col items-center gap-1 overflow-y-auto px-1.5 pb-3">
            {collapsedContent}
            {side === "left" && effectiveCollapsed && (
              <a
                href="https://github.com/camilo6castell"
                target="_blank"
                rel="noopener noreferrer"
                className="flex mt-auto size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-overlay-hover hover:text-foreground transition-colors"
              >
                <LogoIcon className="size-8" />
              </a>
            )}
          </div>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col">{children}</div>
        )}

        {/* Resize handle -- desktop only */}
        {!effectiveCollapsed && (
          <div
            onPointerDown={onPointerDown}
            className={cn(
              "group absolute top-0 hidden h-full w-2.5 cursor-col-resize touch-none select-none lg:block",
              side === "left" ? "-right-1.5" : "-left-1.5",
            )}
          >
            <div className="absolute top-1/2 left-1/2 h-10 w-0.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-transparent transition-colors group-hover:bg-primary/40" />
          </div>
        )}
      </aside>
    </>
  );
}
