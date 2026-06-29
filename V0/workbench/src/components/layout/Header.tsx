import { Link, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { memoryApi } from '@/lib/api'
import { cn } from '@/lib/utils'
import { RedisLogo } from '@/components/RedisLogo'
import { Search, MessageSquare, Wifi, WifiOff } from 'lucide-react'
import { useBackend, type TransportMode } from '@/context/BackendContext'

const navItems = [
  { href: '/', label: 'Explorer', icon: Search },
  { href: '/chat', label: 'Chat', icon: MessageSquare },
]

export function Header() {
  const location = useLocation()
  const { transport, setTransport, mcpStatus } = useBackend()
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => memoryApi.health(),
    refetchInterval: 30000,
  })

  const isHealthy = !!health?.now

  const handleTransportToggle = async (mode: TransportMode) => {
    if (mode === transport) return
    await setTransport(mode)
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b border-redis-dusk-08 bg-redis-midnight/95 backdrop-blur supports-[backdrop-filter]:bg-redis-midnight/80">
      <div className="mx-auto flex h-14 max-w-7xl items-center px-6">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 mr-8">
          <RedisLogo className="w-7 h-7" />
          <span className="font-semibold text-lg text-redis-dusk-01">
            Memory Workbench
          </span>
        </Link>

        {/* Navigation */}
        <nav className="flex items-center gap-1">
          {navItems.map((item) => {
            const isActive =
              item.href === '/'
                ? location.pathname === '/'
                : location.pathname.startsWith(item.href)

            return (
              <Link
                key={item.href}
                to={item.href}
                className={cn(
                  'flex items-center gap-2 px-3 py-2 rounded-[--radius-redis-sm] text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-redis-blue-04 text-redis-dusk-01'
                    : 'text-redis-dusk-03 hover:text-redis-dusk-01 hover:bg-redis-dusk-08'
                )}
              >
                <item.icon className="w-4 h-4" />
                {item.label}
              </Link>
            )
          })}
        </nav>

        {/* Right side */}
        <div className="ml-auto flex items-center gap-4">
          {/* Transport toggle */}
          <div className="flex items-center bg-redis-dusk-08 rounded-[--radius-redis-sm] p-0.5">
            <button
              onClick={() => handleTransportToggle('rest')}
              className={cn(
                'px-2.5 py-1 text-xs font-medium rounded-[--radius-redis-xs] transition-colors flex items-center gap-1.5',
                transport === 'rest'
                  ? 'bg-redis-blue-01 text-white'
                  : 'text-redis-dusk-04 hover:text-redis-dusk-02'
              )}
            >
              REST
              {transport === 'rest' && (
                <span
                  className={cn(
                    'w-1.5 h-1.5 rounded-full',
                    isHealthy ? 'bg-redis-green' : 'bg-redis-red animate-pulse'
                  )}
                />
              )}
            </button>
            <button
              onClick={() => handleTransportToggle('mcp')}
              className={cn(
                'px-2.5 py-1 text-xs font-medium rounded-[--radius-redis-xs] transition-colors flex items-center gap-1.5',
                transport === 'mcp'
                  ? 'bg-redis-blue-01 text-white'
                  : 'text-redis-dusk-04 hover:text-redis-dusk-02'
              )}
            >
              MCP
              {transport === 'mcp' && (
                <span
                  className={cn(
                    'w-1.5 h-1.5 rounded-full',
                    mcpStatus === 'connected'
                      ? 'bg-redis-green'
                      : mcpStatus === 'connecting'
                        ? 'bg-redis-yellow-300 animate-pulse'
                        : 'bg-redis-red'
                  )}
                />
              )}
            </button>
          </div>

          {/* Health indicator */}
          <div className="flex items-center gap-2">
            {isHealthy ? (
              <Wifi className="w-3.5 h-3.5 text-redis-green" />
            ) : (
              <WifiOff className="w-3.5 h-3.5 text-redis-red animate-pulse" />
            )}
            <span className="text-xs text-redis-dusk-04">
              {isHealthy ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}
