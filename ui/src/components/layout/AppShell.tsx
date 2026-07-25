import { Menu, PanelRight } from "lucide-react";
import { Outlet } from "react-router-dom";
import { Blaze } from "@/components/canvasui/Blaze";
import { Frost } from "@/components/canvasui/Frost";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { useThemeSync } from "@/hooks/useThemeSync";
import { useUiStore } from "@/stores/uiStore";
import { ConnectionStatus } from "./ConnectionStatus";
import { RightSidebar } from "./RightSidebar";
import { Sidebar } from "./LeftSidebar";

function useIsDarkTheme() {
  const theme = useUiStore((s) => s.theme);
  const prefersDark = useMediaQuery("(prefers-color-scheme: dark)");
  return theme === "dark" || (theme === "system" && prefersDark);
}

export function AppShell() {
  useThemeSync();
  const isDark = useIsDarkTheme();

  const openLeftMobile = useUiStore((s) => s.openLeftMobile);
  const openRightMobile = useUiStore((s) => s.openRightMobile);

  return (
    <div className="relative flex h-dvh w-full flex-col overflow-hidden bg-background text-foreground lg:flex-row">
      {/* Canvas background effect */}
      <div className="fixed inset-0 z-0">
        {isDark ? (
          <Frost className="pointer-events-auto h-full w-full" opacity={0.4}>
            <div className="h-full w-full bg-background" />
          </Frost>
        ) : (
          <Blaze
            className="h-full w-full"
            sparks={0.8}
            sparkDensity={1.8}
            smoke={0.7}
            glow={1.4}
          >
            <div className="h-full w-full bg-background" />
          </Blaze>
        )}
      </div>

      {/* Mobile top bar */}
      <header className="relative z-20 flex shrink-0 items-center justify-between gap-2 border-b border-border bg-background px-3 py-2.5 lg:hidden">
        <button
          type="button"
          onClick={openLeftMobile}
          aria-label="Open chats and settings"
          className="flex size-9 items-center justify-center rounded-lg text-foreground hover:bg-overlay-hover transition-colors"
        >
          <Menu className="size-5" />
        </button>

        <div className="flex min-w-0 items-center gap-1.5">
          <span className="truncate text-sm font-semibold tracking-tight">My assistant</span>
        </div>

        <div className="flex items-center gap-1">
          <ConnectionStatus className="mr-0.5 hidden sm:inline-flex" />
          <button
            type="button"
            onClick={openRightMobile}
            aria-label="Open files and collections"
            className="flex size-9 items-center justify-center rounded-lg text-foreground hover:bg-overlay-hover transition-colors"
          >
            <PanelRight className="size-5" />
          </button>
        </div>
      </header>

      <div className="relative flex min-h-0 flex-1">
        <Sidebar />

        <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
          <Outlet />
        </main>

        <RightSidebar />
      </div>
    </div>
  );
}
