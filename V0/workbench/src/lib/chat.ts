import { getOpenAIClient, getModel } from './openai'
import type { Persona } from '@/config/personas'
import type { MemoryBackend } from '@/context/BackendContext'

export interface MemoryOperation {
  type: 'retrieve' | 'inference' | 'persist'
  description: string
  timestamp: Date
  details?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
}

const SYSTEM_PROMPT = `You are a helpful AI assistant. You have access to memories about the user from previous conversations. Use these memories to provide personalized and contextual responses. Be conversational and helpful.`

const SYSTEM_PROMPT_NO_MEMORY = `You are a helpful AI assistant. Be conversational and helpful.`

/**
 * The memory server returns content as either a plain string or
 * a structured object like { type: "text", text: "...", annotations, _meta }.
 * This helper normalises both forms to a plain string for OpenAI.
 */
function extractContent(content: unknown): string {
  if (typeof content === 'string') return content
  if (content && typeof content === 'object' && 'text' in content) {
    return (content as { text: string }).text
  }
  return String(content)
}

export async function sendMessage(
  userMessage: string,
  sessionId: string,
  persona: Persona,
  history: ChatMessage[],
  onMemoryOp: (op: MemoryOperation) => void,
  memoryEnabled: boolean = true,
  backend?: MemoryBackend
): Promise<string> {
  let memoryMessages: Array<{ role: string; content: unknown }> = []

  if (memoryEnabled && backend) {
    // Step 1: Retrieve memory context
    onMemoryOp({
      type: 'retrieve',
      description: 'Searching long-term memories...',
      timestamp: new Date(),
    })

    try {
      const promptResponse = await backend.getMemoryPrompt({
        query: userMessage,
        session: {
          session_id: sessionId,
          user_id: persona.id,
        },
        long_term_search: {
          text: userMessage,
          limit: 5,
        },
      })

      memoryMessages = promptResponse.messages || []
      const ltMemories = promptResponse.long_term_memories || []
      const ltmCount = ltMemories.length

      const typeCounts: Record<string, number> = {}
      for (const m of ltMemories) {
        const t = (m as { memory_type?: string }).memory_type || 'unknown'
        typeCounts[t] = (typeCounts[t] || 0) + 1
      }
      const typeDesc = Object.entries(typeCounts)
        .map(([type, count]) => `${count} ${type}`)
        .join(', ')

      onMemoryOp({
        type: 'retrieve',
        description: ltmCount > 0
          ? `Found ${ltmCount} long-term memories (${typeDesc})`
          : 'No matching long-term memories found',
        timestamp: new Date(),
        details: ltmCount > 0
          ? ltMemories
              .map(
                (m) =>
                  `[${(m as { memory_type?: string }).memory_type || 'unknown'}] ${(m as { text?: string }).text?.slice(0, 120) || ''}`
              )
              .join('\n')
          : undefined,
      })
    } catch (err) {
      onMemoryOp({
        type: 'retrieve',
        description: `Memory retrieval failed: ${err instanceof Error ? err.message : 'unknown error'}`,
        timestamp: new Date(),
      })
    }
  }

  // Step 2: Build messages and call OpenAI
  onMemoryOp({
    type: 'inference',
    description: `Sending to ${getModel()}...`,
    timestamp: new Date(),
  })

  const openaiMessages: Array<{
    role: 'system' | 'user' | 'assistant'
    content: string
  }> = [
    {
      role: 'system',
      content: memoryEnabled ? SYSTEM_PROMPT : SYSTEM_PROMPT_NO_MEMORY,
    },
  ]

  // Add memory context messages (normalise content to plain strings)
  for (const msg of memoryMessages) {
    if (msg.role === 'system' || msg.role === 'user' || msg.role === 'assistant') {
      const text = extractContent(msg.content)
      if (text) {
        openaiMessages.push({
          role: msg.role as 'system' | 'user' | 'assistant',
          content: text,
        })
      }
    }
  }

  // Add recent conversation history (last 20 messages)
  const recentHistory = history.slice(-20)
  for (const msg of recentHistory) {
    openaiMessages.push({ role: msg.role, content: msg.content })
  }

  // Add the new user message
  openaiMessages.push({ role: 'user', content: userMessage })

  const client = getOpenAIClient()
  const completion = await client.chat.completions.create({
    model: getModel(),
    messages: openaiMessages,
  })

  const assistantContent =
    completion.choices[0]?.message?.content || 'No response generated.'

  onMemoryOp({
    type: 'inference',
    description: `Response generated (${completion.usage?.total_tokens || '?'} tokens)`,
    timestamp: new Date(),
  })

  // Step 3: Persist conversation to session
  if (memoryEnabled && backend) {
    onMemoryOp({
      type: 'persist',
      description: 'Persisting conversation turn...',
      timestamp: new Date(),
    })

    try {
      const allMessages = [
        ...recentHistory.map((m) => ({ role: m.role, content: m.content })),
        { role: 'user', content: userMessage },
        { role: 'assistant', content: assistantContent },
      ]

      await backend.updateSession(sessionId, {
        messages: allMessages,
        user_id: persona.id,
      })

      onMemoryOp({
        type: 'persist',
        description: `Conversation persisted (triggers memory extraction)`,
        timestamp: new Date(),
      })
    } catch (err) {
      onMemoryOp({
        type: 'persist',
        description: `Persist failed: ${err instanceof Error ? err.message : 'unknown error'}`,
        timestamp: new Date(),
      })
    }
  }

  return assistantContent
}
