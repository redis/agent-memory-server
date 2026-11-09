import { useState, useRef, useEffect } from 'react'
import {
  Send,
  Loader2,
  Plus,
  Search,
  Save,
  Brain,
  ChevronDown,
  ChevronRight,
  Database,
  Sparkles,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { generateSessionId } from '@/lib/utils'
import { personas, type Persona } from '@/config/personas'
import { sendMessage, type ChatMessage, type MemoryOperation } from '@/lib/chat'
import { useBackend } from '@/context/BackendContext'

export default function ChatPage() {
  const { backend, transport } = useBackend()
  const [currentPersona, setCurrentPersona] = useState<Persona>(personas[0])
  const [sessionId, setSessionId] = useState(generateSessionId())
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [memoryOps, setMemoryOps] = useState<MemoryOperation[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [memoryEnabled, setMemoryEnabled] = useState(true)
  const [showPersonaDropdown, setShowPersonaDropdown] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Close dropdown on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node)
      ) {
        setShowPersonaDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const handleNewSession = () => {
    setSessionId(generateSessionId())
    setMessages([])
    setMemoryOps([])
  }

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input,
    }

    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    try {
      const response = await sendMessage(
        userMsg.content,
        sessionId,
        currentPersona,
        messages,
        (op) => setMemoryOps((prev) => [...prev, op]),
        memoryEnabled,
        backend
      )

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response,
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `Error: ${err instanceof Error ? err.message : 'Failed to get response'}`,
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex gap-4 h-[calc(100vh-5rem)] animate-in">
      {/* Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <div className="flex items-center gap-3 mb-4">
          {/* Persona Selector */}
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setShowPersonaDropdown(!showPersonaDropdown)}
              className="flex items-center gap-2 px-3 py-2 bg-redis-dusk-08 border border-redis-dusk-07 rounded-[--radius-redis-sm] text-sm hover:bg-redis-dusk-07 transition-colors"
            >
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: currentPersona.color }}
              />
              {currentPersona.name}
              <ChevronDown className="w-3 h-3 text-redis-dusk-04" />
            </button>
            {showPersonaDropdown && (
              <div className="absolute top-full left-0 mt-1 w-48 bg-redis-midnight border border-redis-dusk-07 rounded-[--radius-redis-md] shadow-lg z-50 py-1">
                {personas.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => {
                      setCurrentPersona(p)
                      setShowPersonaDropdown(false)
                    }}
                    className={cn(
                      'w-full flex items-center gap-2 px-3 py-2 text-sm text-left hover:bg-redis-dusk-08 transition-colors',
                      currentPersona.id === p.id && 'bg-redis-dusk-08'
                    )}
                  >
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: p.color }}
                    />
                    {p.name}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Memory Toggle */}
          <button
            onClick={() => setMemoryEnabled(!memoryEnabled)}
            className={cn(
              'flex items-center gap-2 px-3 py-2 border rounded-[--radius-redis-sm] text-sm transition-colors',
              memoryEnabled
                ? 'bg-redis-blue-04 border-redis-blue-01 text-redis-blue-03'
                : 'bg-redis-dusk-08 border-redis-dusk-07 text-redis-dusk-05'
            )}
          >
            <Brain className="w-4 h-4" />
            Memory {memoryEnabled ? 'On' : 'Off'}
          </button>

          <button
            onClick={handleNewSession}
            className="flex items-center gap-2 px-3 py-2 bg-redis-dusk-08 border border-redis-dusk-07 rounded-[--radius-redis-sm] text-sm hover:bg-redis-dusk-07 transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Session
          </button>

          <span className="text-xs text-redis-dusk-05 font-mono ml-auto flex items-center gap-2">
            <span className="px-1.5 py-0.5 bg-redis-dusk-09 rounded text-redis-dusk-04 uppercase tracking-wider">
              {transport}
            </span>
            {sessionId.slice(0, 8)}...
          </span>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-4 scrollbar-thin pr-2">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center text-redis-dusk-05">
              <div className="text-center">
                <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <h3 className="font-medium text-redis-dusk-03">
                  Start a conversation
                </h3>
                <p className="text-sm mt-1">
                  {memoryEnabled
                    ? 'Messages are stored in working memory and enriched with long-term context'
                    : 'Memory is off — chat without memory server integration'}
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
                    'max-w-[80%] rounded-[--radius-redis-md] px-4 py-3 text-sm',
                    message.role === 'user'
                      ? 'text-white'
                      : 'bg-redis-dusk-08 text-redis-dusk-01'
                  )}
                  style={
                    message.role === 'user'
                      ? {
                          backgroundColor: `color-mix(in srgb, ${currentPersona.color} 15%, var(--color-redis-dusk-08))`,
                          borderLeft: `3px solid ${currentPersona.color}`,
                        }
                      : undefined
                  }
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-redis-dusk-08 rounded-[--radius-redis-md] px-4 py-3">
                <Loader2 className="w-4 h-4 animate-spin text-redis-dusk-04" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="mt-4">
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
              className="flex-1 min-h-[44px] max-h-[200px] px-4 py-2.5 bg-redis-dusk-08 border border-redis-dusk-07 rounded-[--radius-redis-md] text-sm text-redis-dusk-01 placeholder-redis-dusk-05 resize-none focus:outline-none focus:ring-2 focus:ring-redis-blue-04 focus:border-redis-blue-01"
              rows={1}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="px-4 py-2 bg-redis-blue-01 text-white rounded-[--radius-redis-md] hover:bg-redis-blue-02 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <p className="text-xs text-redis-dusk-05 mt-1.5">
            Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>

      {/* Memory Ops Panel */}
      <MemoryOpsPanel ops={memoryOps} />
    </div>
  )
}

function MemoryOpsPanel({ ops }: { ops: MemoryOperation[] }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null)
  const panelEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    panelEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [ops])

  const getIcon = (type: MemoryOperation['type']) => {
    switch (type) {
      case 'retrieve':
        return <Search className="w-3.5 h-3.5" />
      case 'persist':
        return <Save className="w-3.5 h-3.5" />
      case 'inference':
        return <Sparkles className="w-3.5 h-3.5" />
    }
  }

  const getColor = (type: MemoryOperation['type']) => {
    switch (type) {
      case 'retrieve':
        return 'text-redis-blue-03'
      case 'persist':
        return 'text-redis-green'
      case 'inference':
        return 'text-redis-yellow-300'
    }
  }

  const getLabel = (type: MemoryOperation['type']) => {
    switch (type) {
      case 'retrieve':
        return 'RETRIEVE'
      case 'persist':
        return 'PERSIST'
      case 'inference':
        return 'LLM'
    }
  }

  return (
    <div className="w-80 bg-redis-midnight border border-redis-dusk-08 rounded-[--radius-redis-md] flex flex-col">
      <div className="px-4 py-3 border-b border-redis-dusk-08 flex items-center gap-2">
        <Database className="w-4 h-4 text-redis-dusk-04" />
        <h3 className="text-sm font-medium text-redis-dusk-02">
          Memory Operations
        </h3>
      </div>
      <div className="flex-1 overflow-y-auto scrollbar-thin p-2 space-y-1">
        {ops.length === 0 ? (
          <div className="flex items-center justify-center h-full text-redis-dusk-06 text-sm">
            Operations will appear here
          </div>
        ) : (
          ops.map((op, idx) => (
            <div key={idx}>
              <button
                onClick={() =>
                  op.details &&
                  setExpandedIdx(expandedIdx === idx ? null : idx)
                }
                className={cn(
                  'w-full flex items-start gap-2 px-3 py-2 rounded-[--radius-redis-xs] text-left text-xs transition-colors',
                  op.details
                    ? 'hover:bg-redis-dusk-09 cursor-pointer'
                    : 'cursor-default'
                )}
              >
                <span className={cn('mt-0.5 flex-shrink-0', getColor(op.type))}>
                  {getIcon(op.type)}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span
                      className={cn(
                        'text-[10px] font-semibold tracking-wider opacity-70',
                        getColor(op.type)
                      )}
                    >
                      {getLabel(op.type)}
                    </span>
                  </div>
                  <p className="text-redis-dusk-03 mt-0.5">{op.description}</p>
                  <p className="text-redis-dusk-06 mt-0.5">
                    {op.timestamp.toLocaleTimeString()}
                  </p>
                </div>
                {op.details && (
                  <span className="text-redis-dusk-06 mt-0.5">
                    {expandedIdx === idx ? (
                      <ChevronDown className="w-3 h-3" />
                    ) : (
                      <ChevronRight className="w-3 h-3" />
                    )}
                  </span>
                )}
              </button>
              {expandedIdx === idx && op.details && (
                <div className="mx-3 mb-2 px-3 py-2 bg-redis-dusk-09 rounded-[--radius-redis-xs] text-xs text-redis-dusk-04 font-mono whitespace-pre-wrap">
                  {op.details}
                </div>
              )}
            </div>
          ))
        )}
        <div ref={panelEndRef} />
      </div>
    </div>
  )
}
