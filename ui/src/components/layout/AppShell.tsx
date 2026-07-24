import { Menu, PanelRight } from "lucide-react";
import { LogoIcon } from "./LogoIcon";
import { Outlet } from "react-router-dom";
import { useThemeSync } from "@/hooks/useThemeSync";
import { useUiStore } from "@/stores/uiStore";
import { ConnectionStatus } from "./ConnectionStatus";
import { RightSidebar } from "./RightSidebar";
import { Sidebar } from "./LeftSidebar";

export function AppShell() {
  useThemeSync();

  const openLeftMobile = useUiStore((s) => s.openLeftMobile);
  const openRightMobile = useUiStore((s) => s.openRightMobile);

  return (
    <div className="relative flex h-dvh w-full flex-col overflow-hidden bg-background text-foreground lg:flex-row">
      {/* Glow de fondo sutil -- lo que le da el aire "glass" sin recargar la UI */}
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 left-1/3 h-[32rem] w-[32rem] rounded-full bg-primary/10 blur-[120px]"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute bottom-0 right-0 h-[24rem] w-[24rem] rounded-full bg-accent/20 blur-[100px]"
      />

      {/* Barra superior mobile -- reemplaza el acceso directo a los
          sidebars (visibles siempre en desktop) por dos botones que
          abren cada uno como drawer. Oculta desde `lg`. */}
      <header className="relative z-20 flex shrink-0 items-center justify-between gap-2 border-b border-border bg-overlay px-3 py-2.5 backdrop-blur-xl lg:hidden">
        <button
          type="button"
          onClick={openLeftMobile}
          aria-label="Open chats and settings"
          className="flex size-9 items-center justify-center rounded-lg text-foreground hover:bg-overlay-hover"
        >
          <Menu className="size-5" />
        </button>

        <div className="flex min-w-0 items-center gap-1.5">
          <LogoIcon className="size-5 shrink-0" />
          <span className="truncate text-sm font-bold">My assistant</span>
        </div>

        <div className="flex items-center gap-1">
          <ConnectionStatus className="mr-0.5 hidden sm:inline-flex" />
          <button
            type="button"
            onClick={openRightMobile}
            aria-label="Open files and collections"
            className="flex size-9 items-center justify-center rounded-lg text-foreground hover:bg-overlay-hover"
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
