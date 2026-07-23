import { useEffect, useState } from "react"

/**
 * Boolean reactivo para una media query. Usado para decidir el modo de
 * los sidebars (overlay a pantalla completa en mobile vs. panel
 * estático redimensionable en desktop) -- ver SidebarShell.tsx. No se
 * puede resolver esto solo con clases Tailwind responsive porque el
 * ancho inline (`style={{ width }}`) que ya usa SidebarShell para el
 * resize con mouse ganaría siempre por especificidad sobre una clase
 * `lg:w-*`.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia(query).matches : false
  )

  useEffect(() => {
    const mql = window.matchMedia(query)
    const onChange = () => setMatches(mql.matches)
    onChange()
    mql.addEventListener("change", onChange)
    return () => mql.removeEventListener("change", onChange)
  }, [query])

  return matches
}

/** Mismo breakpoint `lg` que usa Tailwind (1024px) para el layout del AppShell. */
export function useIsDesktop(): boolean {
  return useMediaQuery("(min-width: 1024px)")
}
