// Redis AI Hub: shared header injector
// Renders a persistent top bar on every per-library docs site.
// In production, this would be injected by the central host (CDN edge or reverse proxy).
// In the prototype, each site bundles this JS and injects on DOMContentLoaded.

(function () {
  const HUB_URL = (window.REDIS_AI_HUB_URL || '/').replace(/\/+$/, '/') || '/';
  const CURRENT = window.REDIS_AI_HUB_LIBRARY || '';

  const LIBRARIES = {
    core: [
      { slug: 'redisvl', name: 'RedisVL', path: 'redisvl/' },
      { slug: 'agent-memory', name: 'Agent Memory', path: 'agent-memory/' },
      { slug: 'agent-kit', name: 'Agent Kit', path: 'agent-kit/' },
      { slug: 'sre-agent', name: 'SRE Agent', path: 'sre-agent/' },
    ],
    integration: [
      { slug: 'adk', name: 'Google ADK', path: 'adk/' },
      { slug: 'langchain', name: 'LangChain', path: 'langchain/' },
      { slug: 'langgraph', name: 'LangGraph', path: 'langgraph/' },
    ],
    tool: [
      { slug: 'sql', name: 'SQL for Redis', path: 'sql/' },
      { slug: 'optimizer', name: 'Retrieval Optimizer', path: 'optimizer/' },
    ],
    content: [
      { slug: 'recipes', name: 'AI Recipes', path: 'recipes/' },
    ],
  };

  function dropdown(label, items) {
    const links = items.map(it => {
      const isCurrent = it.slug === CURRENT;
      return `<a href="${HUB_URL}${it.path}" class="rah-item${isCurrent ? ' rah-item--active' : ''}">${it.name}</a>`;
    }).join('');
    return `
      <div class="rah-dropdown">
        <button class="rah-dropdown-btn" aria-haspopup="true">${label} <span class="rah-caret">▾</span></button>
        <div class="rah-dropdown-menu">${links}</div>
      </div>`;
  }

  function siblingTabs() {
    let category = null;
    for (const cat of Object.keys(LIBRARIES)) {
      if (LIBRARIES[cat].some(l => l.slug === CURRENT)) { category = cat; break; }
    }
    if (!category) return '';
    const tabs = LIBRARIES[category].map(it => {
      const isCurrent = it.slug === CURRENT;
      return `<a href="${HUB_URL}${it.path}" class="rah-tab${isCurrent ? ' rah-tab--active' : ''}">${it.name}</a>`;
    }).join('');
    return `<div class="rah-tabs">${tabs}</div>`;
  }

  function inject() {
    if (document.getElementById('rah-header')) return;
    const header = document.createElement('div');
    header.id = 'rah-header';
    header.innerHTML = `
      <div class="rah-bar">
        <a href="${HUB_URL}" class="rah-brand">
          <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="11" fill="#FF4438"/>
            <path d="M7 9l5-3 5 3-5 3-5-3zm0 4l5 3 5-3M7 17l5 3 5-3" stroke="#fff" stroke-width="1.5" fill="none"/>
          </svg>
          <span>Redis AI</span>
        </a>
        <nav class="rah-nav">
          ${dropdown('Libraries', LIBRARIES.core)}
          ${dropdown('Integrations', LIBRARIES.integration)}
          ${dropdown('Tools', LIBRARIES.tool)}
          <a href="${HUB_URL}recipes/" class="rah-link">Recipes</a>
        </nav>
        <div class="rah-actions">
          <button class="rah-search" onclick="alert('Federated search panel — not wired in prototype')">⌘K Search</button>
          <a href="https://github.com/redis" target="_blank" class="rah-gh" aria-label="GitHub">★ GitHub</a>
        </div>
      </div>
      ${siblingTabs()}
    `;
    document.body.insertBefore(header, document.body.firstChild);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', inject);
  } else {
    inject();
  }
})();
