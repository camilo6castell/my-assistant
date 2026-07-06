import { Outlet } from "react-router-dom"
import { RightSidebar } from "./RightSidebar"
import { Sidebar } from "./Sidebar"

export function AppShell() {
  return (
    <div className="relative flex h-dvh w-full overflow-hidden bg-background text-foreground">
      {/* Glow de fondo sutil -- lo que le da el aire "glass" sin recargar la UI */}
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 left-1/3 h-[32rem] w-[32rem] rounded-full bg-primary/10 blur-[120px]"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute bottom-0 right-0 h-[24rem] w-[24rem] rounded-full bg-accent/20 blur-[100px]"
      />

      <Sidebar />

      <main className="relative flex min-w-0 flex-1 flex-col">
        <Outlet />
      </main>

      <RightSidebar />
    </div>
  )
}
