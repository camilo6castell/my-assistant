import { useEffect } from "react"
import { applyTheme, useUiStore } from "@/stores/uiStore"

/**
 * Con theme === "system", si el usuario cambia el tema del SO mientras
 * la app está abierta, esto lo refleja sin necesitar un refresh.
 * Con "light"/"dark" explícitos no hace nada (ya está aplicado por
 * setTheme/el boot de main.tsx).
 */
export function useThemeSync() {
  const theme = useUiStore((s) => s.theme)

  useEffect(() => {
    if (theme !== "system") return
    const mql = window.matchMedia("(prefers-color-scheme: dark)")
    const onChange = () => applyTheme("system")
    mql.addEventListener("change", onChange)
    return () => mql.removeEventListener("change", onChange)
  }, [theme])
}
