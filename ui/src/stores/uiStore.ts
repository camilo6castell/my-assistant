import { create } from "zustand"
import { persist } from "zustand/middleware"

export const SIDEBAR_MIN_WIDTH = 260
export const SIDEBAR_MAX_WIDTH = 480
export const SIDEBAR_DEFAULT_WIDTH = 320
export const SIDEBAR_COLLAPSED_WIDTH = 56

export type Theme = "light" | "dark" | "system"

/**
 * Aplica la clase `.dark` al <html> según el theme elegido (resolviendo
 * "system" contra prefers-color-scheme). Se llama desde el store cada
 * vez que cambia el theme, y una vez al boot desde main.tsx -- ver ahí
 * por qué el boot no puede depender solo de la hidratación de zustand
 * (causaría un flash del tema equivocado).
 */
export function applyTheme(theme: Theme) {
  const isDark =
    theme === "dark" ||
    (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches)
  document.documentElement.classList.toggle("dark", isDark)
}

interface UiState {
  leftWidth: number
  leftCollapsed: boolean
  rightWidth: number
  rightCollapsed: boolean
  theme: Theme
  /**
   * Estado de los drawers en mobile (< lg): independiente de
   * leftCollapsed/rightCollapsed, que son un concepto de desktop (rail
   * de íconos vs. panel ancho). En mobile cada sidebar es un overlay a
   * pantalla completa que está abierto o cerrado, nunca "colapsado" --
   * ver SidebarShell.tsx. Deliberadamente NO persistido (ver
   * partialize abajo): reabrir la app en el celular siempre debe
   * arrancar con los drawers cerrados.
   */
  leftMobileOpen: boolean
  rightMobileOpen: boolean

  setLeftWidth: (width: number) => void
  toggleLeftCollapsed: () => void
  setRightWidth: (width: number) => void
  toggleRightCollapsed: () => void
  setTheme: (theme: Theme) => void
  openLeftMobile: () => void
  openRightMobile: () => void
  closeMobileSidebars: () => void
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      leftWidth: SIDEBAR_DEFAULT_WIDTH,
      leftCollapsed: false,
      rightWidth: SIDEBAR_DEFAULT_WIDTH,
      rightCollapsed: false,
      theme: "system",
      leftMobileOpen: false,
      rightMobileOpen: false,

      setLeftWidth: (width) => set({ leftWidth: width }),
      toggleLeftCollapsed: () => set((s) => ({ leftCollapsed: !s.leftCollapsed })),
      setRightWidth: (width) => set({ rightWidth: width }),
      toggleRightCollapsed: () => set((s) => ({ rightCollapsed: !s.rightCollapsed })),
      setTheme: (theme) => {
        // Transición suave solo para este cambio explícito (no en el
        // boot, para no animar el flash inicial -- ver main.tsx).
        document.documentElement.classList.add("theme-transition")
        applyTheme(theme)
        set({ theme })
        window.setTimeout(() => {
          document.documentElement.classList.remove("theme-transition")
        }, 200)
      },
      // Solo un drawer mobile a la vez (mismo patrón que apps de chat
      // de referencia): abrir uno cierra el otro en vez de apilarse.
      openLeftMobile: () => set({ leftMobileOpen: true, rightMobileOpen: false }),
      openRightMobile: () => set({ rightMobileOpen: true, leftMobileOpen: false }),
      closeMobileSidebars: () => set({ leftMobileOpen: false, rightMobileOpen: false }),
    }),
    {
      name: "myassistant-ui",
      // leftMobileOpen/rightMobileOpen quedan afuera a propósito -- ver
      // el comentario en la interfaz de arriba.
      partialize: (state) => ({
        leftWidth: state.leftWidth,
        leftCollapsed: state.leftCollapsed,
        rightWidth: state.rightWidth,
        rightCollapsed: state.rightCollapsed,
        theme: state.theme,
      }),
    }
  )
)
