import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter } from "react-router-dom"
import App from "./App.tsx"
import "./index.css"
import { applyTheme, type Theme } from "./stores/uiStore"

// Aplica el theme persistido (o "system" si es la primera visita) ANTES
// del primer render, leyendo directo de localStorage en vez de esperar
// a que zustand/persist hidrate -- si no, hay un flash del tema
// contrario mientras React monta. Misma key que uiStore ("myassistant-ui").
try {
  const raw = localStorage.getItem("myassistant-ui")
  const theme: Theme = raw ? (JSON.parse(raw).state?.theme ?? "system") : "system"
  applyTheme(theme)
} catch {
  applyTheme("system")
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
})

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>
)
