import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  MessageSquare,
  PlayCircle,
  Search,
  Database,
  GitBranch,
} from 'lucide-react'

const navItems = [
  { href: '/', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/chat', label: 'Chat', icon: MessageSquare },
  { href: '/scenarios', label: 'Scenarios', icon: PlayCircle },
  { href: '/explorer', label: 'Explorer', icon: Search },
  { href: '/working-memory', label: 'Sessions', icon: Database },
  { href: '/architecture', label: 'Architecture', icon: GitBranch },
]

export function Header() {
  const location = useLocation()

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 mr-8">
          <div className="flex items-center justify-center w-8 h-8 rounded bg-primary text-primary-foreground font-bold text-sm">
            <svg viewBox="0 0 32 32" className="w-5 h-5" fill="currentColor">
              <path d="M16 2L4 8v16l12 6 12-6V8L16 2zm0 2.5L25.5 9 16 13.5 6.5 9 16 4.5zM6 10.8l9 4.5v11.4l-9-4.5V10.8zm20 0v11.4l-9 4.5V15.3l9-4.5z" />
            </svg>
          </div>
          <span className="font-semibold text-lg">Memory Workbench</span>
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
                  'flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-primary/10 text-primary'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
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
          <a
            href="http://localhost:28000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            API Docs
          </a>
          <a
            href="http://localhost:28001"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Redis Insight
          </a>
        </div>
      </div>
    </header>
  )
}
