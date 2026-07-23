import { useEffect, useState } from "react"

/**
 * Estado de conectividad del navegador (online/offline). Deliberadamente
 * NO hace ping al backend -- eso implicaría un endpoint/comportamiento
 * nuevo del lado del servidor, y la restricción del pedido es cero
 * cambios de integración. navigator.onLine + los eventos del browser
 * son suficientes para la señal que le importa al usuario: "¿tengo
 * salida a internet ahora mismo?".
 */
export function useOnlineStatus(): boolean {
  const [online, setOnline] = useState(() =>
    typeof navigator !== "undefined" ? navigator.onLine : true
  )

  useEffect(() => {
    const goOnline = () => setOnline(true)
    const goOffline = () => setOnline(false)
    window.addEventListener("online", goOnline)
    window.addEventListener("offline", goOffline)
    return () => {
      window.removeEventListener("online", goOnline)
      window.removeEventListener("offline", goOffline)
    }
  }, [])

  return online
}
