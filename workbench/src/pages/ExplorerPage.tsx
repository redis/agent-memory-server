import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { type MemoryRecordResult } from '@/lib/api'
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
  Pin,
  Merge,
} from 'lucide-react'
import { cn, formatRelativeTime, truncateText } from '@/lib/utils'
import { useBackend } from '@/context/BackendContext'

export default function ExplorerPage() {
  const { backend, transport } = useBackend()
  const queryClient = useQueryClient()
  const [searchText, setSearchText] = useState('')
  const [submittedSearch, setSubmittedSearch] = useState('')
  const [filters, setFilters] = useState({
    search_mode: 'semantic' as 'semantic' | 'keyword' | 'hybrid',
    memory_type: '' as '' | 'semantic' | 'episodic' | 'message',
    limit: 25,
  })
  const [showFilters, setShowFilters] = useState(false)
  const [selectedMemory, setSelectedMemory] =
    useState<MemoryRecordResult | null>(null)

  const {
    data: searchResults,
    isLoading,
  } = useQuery({
    queryKey: ['memories', submittedSearch, filters, transport],
    queryFn: () =>
      backend.search({
        text: submittedSearch,
        search_mode: filters.search_mode,
        memory_type: filters.memory_type
          ? { eq: filters.memory_type }
          : undefined,
        limit: filters.limit,
      }),
  })

  const [confirmClearAll, setConfirmClearAll] = useState(false)

  const deleteMutation = useMutation({
    mutationFn: (id: string) => backend.deleteMemory(id),
    onSuccess: () => {
      toast.success('Memory deleted')
      queryClient.invalidateQueries({ queryKey: ['memories'] })
      setSelectedMemory(null)
    },
    onError: () => {
      toast.error('Failed to delete memory')
    },
  })

  const clearAllMutation = useMutation({
    mutationFn: async () => {
      const ids = searchResults?.memories?.map((m) => m.id) ?? []
      if (ids.length === 0) return
      await backend.deleteMemories(ids)
    },
    onSuccess: () => {
      toast.success('All displayed memories deleted')
      queryClient.invalidateQueries({ queryKey: ['memories'] })
      setSelectedMemory(null)
      setConfirmClearAll(false)
    },
    onError: () => {
      toast.error('Failed to clear memories')
      setConfirmClearAll(false)
    },
  })

  const compactMutation = useMutation({
    mutationFn: () => backend.compactMemories(),
    onSuccess: (data) => {
      toast.success(data.status)
      queryClient.invalidateQueries({ queryKey: ['memories'] })
      setSelectedMemory(null)
    },
    onError: () => {
      toast.error('Failed to compact memories')
    },
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    setSubmittedSearch(searchText)
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    toast.success('Copied to clipboard')
  }

  return (
    <div className="flex gap-4 h-[calc(100vh-5rem)] animate-in">
      {/* Main Panel */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Search */}
        <form onSubmit={handleSearch} className="mb-4">
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-redis-dusk-05" />
              <input
                type="text"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                placeholder="Search memories (semantic search)..."
                className="w-full pl-10 pr-4 py-2 bg-redis-dusk-08 border border-redis-dusk-07 rounded-[--radius-redis-md] text-sm text-redis-dusk-01 placeholder-redis-dusk-05 focus:outline-none focus:ring-2 focus:ring-redis-blue-04 focus:border-redis-blue-01"
              />
            </div>
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className={cn(
                'flex items-center gap-2 px-4 py-2 border rounded-[--radius-redis-md] text-sm transition-colors',
                showFilters
                  ? 'bg-redis-blue-04 border-redis-blue-01 text-redis-dusk-01'
                  : 'bg-redis-dusk-08 border-redis-dusk-07 hover:bg-redis-dusk-07 text-redis-dusk-03'
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
              className="px-4 py-2 bg-redis-blue-01 text-white rounded-[--radius-redis-md] text-sm hover:bg-redis-blue-02 transition-colors"
            >
              Search
            </button>
            <button
              type="button"
              onClick={() => compactMutation.mutate()}
              disabled={compactMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-redis-dusk-08 border border-redis-dusk-07 text-redis-lime rounded-[--radius-redis-md] text-sm hover:bg-redis-dusk-07 transition-colors disabled:opacity-40"
              title="Merge hash-based and semantic duplicates"
            >
              {compactMutation.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Merge className="w-4 h-4" />
              )}
              Deduplicate
            </button>
            {searchResults && searchResults.memories.length > 0 && (
              confirmClearAll ? (
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => clearAllMutation.mutate()}
                    disabled={clearAllMutation.isPending}
                    className="px-3 py-2 bg-redis-red text-white rounded-[--radius-redis-md] text-sm hover:bg-red-600 transition-colors"
                  >
                    {clearAllMutation.isPending ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      `Delete ${searchResults.memories.length}`
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => setConfirmClearAll(false)}
                    className="px-3 py-2 bg-redis-dusk-08 border border-redis-dusk-07 text-redis-dusk-03 rounded-[--radius-redis-md] text-sm hover:bg-redis-dusk-07 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setConfirmClearAll(true)}
                  className="flex items-center gap-2 px-4 py-2 bg-redis-dusk-08 border border-redis-dusk-07 text-redis-red rounded-[--radius-redis-md] text-sm hover:bg-redis-red-dark hover:border-redis-red transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear All
                </button>
              )
            )}
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="mt-3 p-4 bg-redis-midnight border border-redis-dusk-08 rounded-[--radius-redis-md]">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <label className="text-xs font-medium text-redis-dusk-04 mb-1 block">
                    Search Mode
                  </label>
                  <select
                    value={filters.search_mode}
                    onChange={(e) =>
                      setFilters({
                        ...filters,
                        search_mode: e.target.value as 'semantic' | 'keyword' | 'hybrid',
                      })
                    }
                    className="w-full px-3 py-2 bg-redis-dusk-08 border border-redis-dusk-07 rounded-[--radius-redis-sm] text-sm text-redis-dusk-01 focus:outline-none focus:ring-2 focus:ring-redis-blue-04"
                  >
                    <option value="semantic">Semantic</option>
                    <option value="keyword">Keyword</option>
                    <option value="hybrid">Hybrid</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium text-redis-dusk-04 mb-1 block">
                    Memory Type
                  </label>
                  <select
                    value={filters.memory_type}
                    onChange={(e) =>
                      setFilters({
                        ...filters,
                        memory_type: e.target.value as
                          | ''
                          | 'semantic'
                          | 'episodic'
                          | 'message',
                      })
                    }
                    className="w-full px-3 py-2 bg-redis-dusk-08 border border-redis-dusk-07 rounded-[--radius-redis-sm] text-sm text-redis-dusk-01 focus:outline-none focus:ring-2 focus:ring-redis-blue-04"
                  >
                    <option value="">All Types</option>
                    <option value="semantic">Semantic</option>
                    <option value="episodic">Episodic</option>
                    <option value="message">Message</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium text-redis-dusk-04 mb-1 block">
                    Results Limit
                  </label>
                  <select
                    value={filters.limit}
                    onChange={(e) =>
                      setFilters({
                        ...filters,
                        limit: parseInt(e.target.value),
                      })
                    }
                    className="w-full px-3 py-2 bg-redis-dusk-08 border border-redis-dusk-07 rounded-[--radius-redis-sm] text-sm text-redis-dusk-01 focus:outline-none focus:ring-2 focus:ring-redis-blue-04"
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
        <div className="flex-1 overflow-y-auto space-y-2 scrollbar-thin pr-1">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-6 h-6 animate-spin text-redis-blue-03" />
            </div>
          ) : searchResults?.memories?.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-redis-dusk-05">
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
                  'bg-redis-midnight border rounded-[--radius-redis-md] p-4 cursor-pointer transition-colors',
                  selectedMemory?.id === memory.id
                    ? 'border-redis-blue-01 bg-redis-dusk-09'
                    : 'border-redis-dusk-08 hover:bg-redis-dusk-09'
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
                      {memory.pinned && (
                        <Pin className="w-3 h-3 text-redis-yellow-300" />
                      )}
                      {searchText && (memory.score !== undefined || memory.dist !== undefined) && (
                        <span className="text-xs text-redis-dusk-05">
                          Score:{' '}
                          {(
                            memory.score ?? (memory.dist !== undefined ? 1 - memory.dist : 0)
                          ).toFixed(3)}
                          {memory.score_type ? ` (${memory.score_type})` : ''}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-redis-dusk-02">
                      {truncateText(memory.text, 200)}
                    </p>
                    {memory.topics && memory.topics.length > 0 && (
                      <div className="flex gap-1 mt-2 flex-wrap">
                        {memory.topics.map((topic) => (
                          <span
                            key={topic}
                            className="px-1.5 py-0.5 text-xs bg-redis-dusk-08 text-redis-dusk-04 rounded"
                          >
                            {topic}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-redis-dusk-05 text-right flex-shrink-0">
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
          <div className="mt-4 text-sm text-redis-dusk-05">
            Showing {searchResults.memories?.length || 0} of{' '}
            {searchResults.total} memories
          </div>
        )}
      </div>

      {/* Detail Panel */}
      <div className="w-96 bg-redis-midnight border border-redis-dusk-08 rounded-[--radius-redis-md] flex flex-col">
        {selectedMemory ? (
          <>
            <div className="p-4 border-b border-redis-dusk-08 flex items-center justify-between">
              <h3 className="font-semibold text-redis-dusk-01">
                Memory Details
              </h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => copyToClipboard(selectedMemory.id)}
                  className="p-1.5 rounded hover:bg-redis-dusk-08 text-redis-dusk-04 hover:text-redis-dusk-01 transition-colors"
                  title="Copy ID"
                >
                  <Copy className="w-4 h-4" />
                </button>
                <button
                  onClick={() => deleteMutation.mutate(selectedMemory.id)}
                  disabled={deleteMutation.isPending}
                  className="p-1.5 rounded hover:bg-redis-red-dark text-redis-dusk-04 hover:text-redis-red transition-colors"
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
            <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin">
              <DetailField label="ID">
                <p className="text-sm font-mono break-all text-redis-dusk-02">
                  {selectedMemory.id}
                </p>
              </DetailField>
              <DetailField label="Type">
                <span
                  className={cn(
                    'px-2 py-1 text-xs font-medium rounded border',
                    selectedMemory.memory_type === 'semantic' &&
                      'badge-semantic',
                    selectedMemory.memory_type === 'episodic' &&
                      'badge-episodic',
                    selectedMemory.memory_type === 'message' && 'badge-message'
                  )}
                >
                  {selectedMemory.memory_type}
                </span>
              </DetailField>
              <DetailField label="Content">
                <p className="text-sm mt-1 whitespace-pre-wrap text-redis-dusk-02">
                  {selectedMemory.text}
                </p>
              </DetailField>
              {selectedMemory.topics && selectedMemory.topics.length > 0 && (
                <DetailField label="Topics">
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {selectedMemory.topics.map((topic) => (
                      <span
                        key={topic}
                        className="px-2 py-1 text-xs bg-redis-blue-04 text-redis-blue-03 rounded"
                      >
                        {topic}
                      </span>
                    ))}
                  </div>
                </DetailField>
              )}
              {selectedMemory.entities &&
                selectedMemory.entities.length > 0 && (
                  <DetailField label="Entities">
                    <div className="flex gap-1 mt-1 flex-wrap">
                      {selectedMemory.entities.map((entity) => (
                        <span
                          key={entity}
                          className="px-2 py-1 text-xs bg-redis-dusk-08 text-redis-dusk-03 rounded"
                        >
                          {entity}
                        </span>
                      ))}
                    </div>
                  </DetailField>
                )}
              {selectedMemory.event_date && (
                <DetailField label="Event Date">
                  <p className="text-sm text-redis-dusk-02">
                    {selectedMemory.event_date}
                  </p>
                </DetailField>
              )}
              {selectedMemory.pinned !== undefined && (
                <DetailField label="Pinned">
                  <div className="flex items-center gap-1.5">
                    {selectedMemory.pinned ? (
                      <>
                        <Pin className="w-3.5 h-3.5 text-redis-yellow-300" />
                        <span className="text-sm text-redis-yellow-300">Yes</span>
                      </>
                    ) : (
                      <span className="text-sm text-redis-dusk-04">No</span>
                    )}
                  </div>
                </DetailField>
              )}
              {selectedMemory.extraction_strategy && (
                <DetailField label="Extraction Strategy">
                  <span className="text-sm text-redis-dusk-02 font-mono">
                    {selectedMemory.extraction_strategy}
                  </span>
                </DetailField>
              )}
              <DetailField label="Created">
                <p className="text-sm text-redis-dusk-02">
                  {new Date(selectedMemory.created_at).toLocaleString()}
                </p>
              </DetailField>
              <DetailField label="Last Accessed">
                <p className="text-sm text-redis-dusk-02">
                  {new Date(selectedMemory.last_accessed).toLocaleString()}
                </p>
              </DetailField>
              <DetailField label="Access Count">
                <p className="text-sm text-redis-dusk-02">
                  {selectedMemory.access_count || 0}
                </p>
              </DetailField>
              {selectedMemory.namespace && (
                <DetailField label="Namespace">
                  <p className="text-sm text-redis-dusk-02">
                    {selectedMemory.namespace}
                  </p>
                </DetailField>
              )}
              {selectedMemory.user_id && (
                <DetailField label="User ID">
                  <p className="text-sm font-mono text-redis-dusk-02">
                    {selectedMemory.user_id}
                  </p>
                </DetailField>
              )}
              {selectedMemory.session_id && (
                <DetailField label="Session ID">
                  <p className="text-sm font-mono text-redis-dusk-02">
                    {selectedMemory.session_id}
                  </p>
                </DetailField>
              )}
              {selectedMemory.memory_hash && (
                <DetailField label="Memory Hash">
                  <p className="text-sm font-mono break-all text-redis-dusk-04">
                    {selectedMemory.memory_hash}
                  </p>
                </DetailField>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-redis-dusk-06">
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

function DetailField({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div>
      <label className="text-xs text-redis-dusk-05 uppercase tracking-wider">
        {label}
      </label>
      {children}
    </div>
  )
}
