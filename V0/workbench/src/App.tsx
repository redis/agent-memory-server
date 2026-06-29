import { Routes, Route } from 'react-router-dom'
import { Header } from '@/components/layout/Header'
import { BackendProvider } from '@/context/BackendContext'
import ExplorerPage from '@/pages/ExplorerPage'
import ChatPage from '@/pages/ChatPage'

function App() {
  return (
    <BackendProvider>
      <div className="flex min-h-screen flex-col bg-[#07151C]">
        <Header />
        <main className="flex-1 mx-auto w-full max-w-7xl px-6 py-6">
          <Routes>
            <Route path="/" element={<ExplorerPage />} />
            <Route path="/chat" element={<ChatPage />} />
          </Routes>
        </main>
      </div>
    </BackendProvider>
  )
}

export default App
