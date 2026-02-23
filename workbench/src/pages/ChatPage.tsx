import { useState } from 'react'
import { MessageSquare, Plus, Trash2, Send, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

interface Session {
  id: string
  name: string
  createdAt: Date
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>([
    { id: 'demo', name: 'Demo Session', createdAt: new Date() },
  ])
  const [currentSession, setCurrentSession] = useState<string>('demo')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Simulate AI response (in real implementation, this would call the backend)
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `This is a demo response. In the full implementation, I would:

1. Retrieve relevant memories from long-term storage
2. Access the current session's working memory
3. Generate a contextual response using an LLM
4. Optionally store new memories from our conversation

The memory server is running at http://localhost:28000 - you can interact with it via the API or through Claude MCP.`,
      }
      setMessages((prev) => [...prev, assistantMessage])
      setIsLoading(false)
    }, 1000)
  }

  const handleNewSession = () => {
    const id = Date.now().toString()
    setSessions((prev) => [
      { id, name: `Session ${prev.length + 1}`, createdAt: new Date() },
      ...prev,
    ])
    setCurrentSession(id)
    setMessages([])
  }

  const handleDeleteSession = (id: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== id))
    if (currentSession === id) {
      setCurrentSession(sessions[0]?.id || '')
      setMessages([])
    }
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4 animate-in">
      {/* Session Sidebar */}
      <div className="w-64 bg-card border rounded-lg flex flex-col">
        <div className="p-3 border-b flex items-center justify-between">
          <h2 className="font-semibold">Sessions</h2>
          <button
            onClick={handleNewSession}
            className="p-1.5 rounded hover:bg-muted transition-colors"
            title="New Session"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={cn(
                'flex items-center justify-between p-2 rounded-md cursor-pointer group',
                currentSession === session.id
                  ? 'bg-primary/10 text-primary'
                  : 'hover:bg-muted'
              )}
              onClick={() => setCurrentSession(session.id)}
            >
              <div className="flex items-center gap-2 min-w-0">
                <MessageSquare className="w-4 h-4 flex-shrink-0" />
                <span className="text-sm truncate">{session.name}</span>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  handleDeleteSession(session.id)
                }}
                className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 bg-card border rounded-lg flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <h3 className="font-medium">Start a conversation</h3>
                <p className="text-sm mt-1">
                  Your messages will be stored in working memory
                </p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  'flex',
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                )}
              >
                <div
                  className={cn(
                    'max-w-[80%] rounded-lg px-4 py-2',
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted'
                  )}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-muted rounded-lg px-4 py-2">
                <Loader2 className="w-4 h-4 animate-spin" />
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="p-4 border-t">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSend()
                }
              }}
              placeholder="Type a message..."
              className="flex-1 min-h-[44px] max-h-[200px] px-4 py-2 bg-muted border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary/50"
              rows={1}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  )
}
