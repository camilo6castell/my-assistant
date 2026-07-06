import { create } from "zustand"
import { persist } from "zustand/middleware"

export const SIDEBAR_MIN_WIDTH = 260
export const SIDEBAR_MAX_WIDTH = 480
export const SIDEBAR_DEFAULT_WIDTH = 320
export const SIDEBAR_COLLAPSED_WIDTH = 56

interface UiState {
  leftWidth: number
  leftCollapsed: boolean
  rightWidth: number
  rightCollapsed: boolean

  setLeftWidth: (width: number) => void
  toggleLeftCollapsed: () => void
  setRightWidth: (width: number) => void
  toggleRightCollapsed: () => void
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      leftWidth: SIDEBAR_DEFAULT_WIDTH,
      leftCollapsed: false,
      rightWidth: SIDEBAR_DEFAULT_WIDTH,
      rightCollapsed: false,

      setLeftWidth: (width) => set({ leftWidth: width }),
      toggleLeftCollapsed: () => set((s) => ({ leftCollapsed: !s.leftCollapsed })),
      setRightWidth: (width) => set({ rightWidth: width }),
      toggleRightCollapsed: () => set((s) => ({ rightCollapsed: !s.rightCollapsed })),
    }),
    { name: "myassistant-ui" }
  )
)
