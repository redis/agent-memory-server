import { Routes, Route } from 'react-router-dom'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
import DashboardPage from '@/pages/DashboardPage'
import ChatPage from '@/pages/ChatPage'
import ScenariosPage from '@/pages/ScenariosPage'
import ScenarioDetailPage from '@/pages/ScenarioDetailPage'
import ExplorerPage from '@/pages/ExplorerPage'
import WorkingMemoryPage from '@/pages/WorkingMemoryPage'
import ArchitecturePage from '@/pages/ArchitecturePage'

function App() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <Header />
      <main className="flex-1 container py-6">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/scenarios" element={<ScenariosPage />} />
          <Route path="/scenarios/:scenarioId" element={<ScenarioDetailPage />} />
          <Route path="/explorer" element={<ExplorerPage />} />
          <Route path="/working-memory" element={<WorkingMemoryPage />} />
          <Route path="/architecture" element={<ArchitecturePage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

export default App
