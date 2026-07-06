import { Navigate, Route, Routes } from "react-router-dom"
import { AppShell } from "@/components/layout/AppShell"
import { ChatView } from "@/components/chat/ChatView"
import { EmptyState } from "@/components/chat/EmptyState"

function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<EmptyState />} />
        <Route path="/c/:conversationId" element={<ChatView />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

export default App
