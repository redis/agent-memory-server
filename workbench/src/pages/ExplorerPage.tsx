import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { memoryApi, MemoryRecordResult } from '@/lib/api'
import { toast } from 'sonner'
import {
  Search,
  Filter,
  Trash2,
  Copy,
  ChevronDown,
  ChevronRight,
  Loader2,
  Database,
} from 'lucide-react'
import { cn, formatRelativeTime, truncateText } from '@/lib/utils'

export default function ExplorerPage() {
  const queryClient = useQueryClient()
  const [searchText, setSearchText] = useState('')
  const [filters, setFilters] = useState({
    memory_type: '' as '' | 'semantic' | 'episodic' | 'message',
    limit: 25,
  })
  const [showFilters, setShowFilters] = useState(false)
  const [selectedMemory, setSelectedMemory] = useState<MemoryRecordResult | null>(
    null
  )

  const {
    data: searchResults,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ['memories', searchText, filters],
    queryFn: () =>
      memoryApi.search({
        text: searchText || undefined,
        memory_type: filters.memory_type
          ? { eq: filters.memory_type }
          : undefined,
        limit: filters.limit,
      }),
    enabled: true,
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => memoryApi.deleteMemory(id),
    onSuccess: () => {
      toast.success('Memory deleted')
      queryClient.invalidateQueries({ queryKey: ['memories'] })
      setSelectedMemory(null)
    },
    onError: () => {
      toast.error('Failed to delete memory')
    },
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    refetch()
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    toast.success('Copied to clipboard')
  }

  return (
    <div className="flex gap-4 h-[calc(100vh-8rem)] animate-in">
      {/* Main Panel */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Search */}
        <form onSubmit={handleSearch} className="mb-4">
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                placeholder="Search memories (semantic search)..."
                className="w-full pl-10 pr-4 py-2 bg-card border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className={cn(
                'flex items-center gap-2 px-4 py-2 border rounded-lg transition-colors',
                showFilters
                  ? 'bg-primary/10 border-primary/50 text-primary'
                  : 'bg-card hover:bg-muted'
              )}
            >
              <Filter className="w-4 h-4" />
              Filters
              {showFilters ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
            >
              Search
            </button>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="mt-3 p-4 bg-card border rounded-lg">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <label className="text-sm font-medium mb-1 block">
                    Memory Type
                  </label>
                  <select
                    value={filters.memory_type}
                    onChange={(e) =>
                      setFilters({
                        ...filters,
                        memory_type: e.target.value as '' | 'semantic' | 'episodic' | 'message',
                      })
                    }
                    className="w-full px-3 py-2 bg-muted border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
                  >
                    <option value="">All Types</option>
                    <option value="semantic">Semantic</option>
                    <option value="episodic">Episodic</option>
                    <option value="message">Message</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium mb-1 block">
                    Results Limit
                  </label>
                  <select
                    value={filters.limit}
                    onChange={(e) =>
                      setFilters({ ...filters, limit: parseInt(e.target.value) })
                    }
                    className="w-full px-3 py-2 bg-muted border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
                  >
                    <option value="10">10</option>
                    <option value="25">25</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                  </select>
                </div>
              </div>
            </div>
          )}
        </form>

        {/* Results */}
        <div className="flex-1 overflow-y-auto space-y-2">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-6 h-6 animate-spin text-primary" />
            </div>
          ) : searchResults?.memories?.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Database className="w-12 h-12 mb-4 opacity-50" />
              <p>No memories found</p>
              <p className="text-sm mt-1">
                Try adjusting your search or filters
              </p>
            </div>
          ) : (
            searchResults?.memories?.map((memory) => (
              <div
                key={memory.id}
                onClick={() => setSelectedMemory(memory)}
                className={cn(
                  'bg-card border rounded-lg p-4 cursor-pointer transition-colors',
                  selectedMemory?.id === memory.id
                    ? 'border-primary bg-primary/5'
                    : 'hover:border-muted-foreground/30'
                )}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <span
                        className={cn(
                          'px-1.5 py-0.5 text-xs font-medium rounded border',
                          memory.memory_type === 'semantic' && 'badge-semantic',
                          memory.memory_type === 'episodic' && 'badge-episodic',
                          memory.memory_type === 'message' && 'badge-message'
                        )}
                      >
                        {memory.memory_type}
                      </span>
                      {searchText && memory.dist !== undefined && (
                        <span className="text-xs text-muted-foreground">
                          Score: {(1 - memory.dist).toFixed(3)}
                        </span>
                      )}
                    </div>
                    <p className="text-sm">{truncateText(memory.text, 200)}</p>
                    {memory.topics && memory.topics.length > 0 && (
                      <div className="flex gap-1 mt-2 flex-wrap">
                        {memory.topics.map((topic) => (
                          <span
                            key={topic}
                            className="px-1.5 py-0.5 text-xs bg-muted rounded"
                          >
                            {topic}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground text-right flex-shrink-0">
                    <p>{formatRelativeTime(memory.created_at)}</p>
                    <p className="mt-1">
                      Accessed {memory.access_count || 0} times
                    </p>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Results count */}
        {searchResults && (
          <div className="mt-4 text-sm text-muted-foreground">
            Showing {searchResults.memories?.length || 0} of {searchResults.total}{' '}
            memories
          </div>
        )}
      </div>

      {/* Detail Panel */}
      <div className="w-96 bg-card border rounded-lg flex flex-col">
        {selectedMemory ? (
          <>
            <div className="p-4 border-b flex items-center justify-between">
              <h3 className="font-semibold">Memory Details</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => copyToClipboard(selectedMemory.id)}
                  className="p-1.5 rounded hover:bg-muted transition-colors"
                  title="Copy ID"
                >
                  <Copy className="w-4 h-4" />
                </button>
                <button
                  onClick={() => deleteMutation.mutate(selectedMemory.id)}
                  disabled={deleteMutation.isPending}
                  className="p-1.5 rounded hover:bg-destructive/10 hover:text-destructive transition-colors"
                  title="Delete"
                >
                  {deleteMutation.isPending ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div>
                <label className="text-xs text-muted-foreground uppercase tracking-wider">
                  ID
                </label>
                <p className="text-sm font-mono break-all">{selectedMemory.id}</p>
              </div>
              <div>
                <label className="text-xs text-muted-foreground uppercase tracking-wider">
                  Type
                </label>
                <p>
                  <span
                    className={cn(
                      'px-2 py-1 text-xs font-medium rounded border',
                      selectedMemory.memory_type === 'semantic' && 'badge-semantic',
                      selectedMemory.memory_type === 'episodic' && 'badge-episodic',
                      selectedMemory.memory_type === 'message' && 'badge-message'
                    )}
                  >
                    {selectedMemory.memory_type}
                  </span>
                </p>
              </div>
              <div>
                <label className="text-xs text-muted-foreground uppercase tracking-wider">
                  Content
                </label>
                <p className="text-sm mt-1 whitespace-pre-wrap">
                  {selectedMemory.text}
                </p>
              </div>
              {selectedMemory.topics && selectedMemory.topics.length > 0 && (
                <div>
                  <label className="text-xs text-muted-foreground uppercase tracking-wider">
                    Topics
                  </label>
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {selectedMemory.topics.map((topic) => (
                      <span
                        key={topic}
                        className="px-2 py-1 text-xs bg-primary/10 text-primary rounded"
                      >
                        {topic}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {selectedMemory.entities && selectedMemory.entities.length > 0 && (
                <div>
                  <label className="text-xs text-muted-foreground uppercase tracking-wider">
                    Entities
                  </label>
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {selectedMemory.entities.map((entity) => (
                      <span
                        key={entity}
                        className="px-2 py-1 text-xs bg-muted rounded"
                      >
                        {entity}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {selectedMemory.event_date && (
                <div>
                  <label className="text-xs text-muted-foreground uppercase tracking-wider">
                    Event Date
                  </label>
                  <p className="text-sm">{selectedMemory.event_date}</p>
                </div>
              )}
              <div>
                <label className="text-xs text-muted-foreground uppercase tracking-wider">
                  Created
                </label>
                <p className="text-sm">
                  {new Date(selectedMemory.created_at).toLocaleString()}
                </p>
              </div>
              <div>
                <label className="text-xs text-muted-foreground uppercase tracking-wider">
                  Last Accessed
                </label>
                <p className="text-sm">
                  {new Date(selectedMemory.last_accessed).toLocaleString()}
                </p>
              </div>
              <div>
                <label className="text-xs text-muted-foreground uppercase tracking-wider">
                  Access Count
                </label>
                <p className="text-sm">{selectedMemory.access_count || 0}</p>
              </div>
              {selectedMemory.namespace && (
                <div>
                  <label className="text-xs text-muted-foreground uppercase tracking-wider">
                    Namespace
                  </label>
                  <p className="text-sm">{selectedMemory.namespace}</p>
                </div>
              )}
              {selectedMemory.user_id && (
                <div>
                  <label className="text-xs text-muted-foreground uppercase tracking-wider">
                    User ID
                  </label>
                  <p className="text-sm font-mono">{selectedMemory.user_id}</p>
                </div>
              )}
              {selectedMemory.session_id && (
                <div>
                  <label className="text-xs text-muted-foreground uppercase tracking-wider">
                    Session ID
                  </label>
                  <p className="text-sm font-mono">{selectedMemory.session_id}</p>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Select a memory to view details</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
