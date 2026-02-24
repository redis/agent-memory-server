import OpenAI from 'openai'

let client: OpenAI | null = null

export function getOpenAIClient(): OpenAI {
  if (!client) {
    const apiKey = import.meta.env.VITE_OPENAI_API_KEY
    if (!apiKey) {
      throw new Error(
        'VITE_OPENAI_API_KEY is not set. Add it to your .env file.'
      )
    }
    client = new OpenAI({
      apiKey,
      dangerouslyAllowBrowser: true,
    })
  }
  return client
}

export function getModel(): string {
  return import.meta.env.VITE_OPENAI_MODEL || 'o4-mini'
}
