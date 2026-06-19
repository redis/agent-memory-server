import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function truncateText(text: string, maxLength: number) {
  if (text.length <= maxLength) {
    return text
  }
  return `${text.slice(0, maxLength - 1)}…`
}

export function formatRelativeTime(value?: string | null) {
  if (!value) {
    return 'Unknown'
  }

  const timestamp = new Date(value)
  if (Number.isNaN(timestamp.getTime())) {
    return 'Unknown'
  }

  const diffMs = Date.now() - timestamp.getTime()
  const diffSeconds = Math.round(diffMs / 1000)
  const absSeconds = Math.abs(diffSeconds)

  if (absSeconds < 60) {
    return diffSeconds >= 0 ? 'just now' : 'in a few seconds'
  }

  const minutes = Math.round(absSeconds / 60)
  if (minutes < 60) {
    return diffSeconds >= 0 ? `${minutes}m ago` : `in ${minutes}m`
  }

  const hours = Math.round(minutes / 60)
  if (hours < 24) {
    return diffSeconds >= 0 ? `${hours}h ago` : `in ${hours}h`
  }

  const days = Math.round(hours / 24)
  if (days < 7) {
    return diffSeconds >= 0 ? `${days}d ago` : `in ${days}d`
  }

  return timestamp.toLocaleDateString()
}

export function generateSessionId() {
  return crypto.randomUUID()
}
