import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  BookOpen,
  Brain,
  Bot,
  ChevronDown,
  ChevronLeft,
  Check,
  CirclePlus,
  Copy,
  HeartPulse,
  Info,
  LoaderCircle,
  LogOut,
  MessageSquare,
  PanelRight,
  Pencil,
  RefreshCw,
  Reply,
  SendHorizontal,
  Settings2,
  ShieldCheck,
  Square,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Languages,
  Menu,
  SunMoon,
  X,
} from 'lucide-react'
import { ApiRequestError, apiFetch, csrfHeaders, readJson } from './api'
import { KnowledgePage, MemoryPage, MoodPage, OperationsPage, PrivacyPage } from './FeaturePages'
import { translate } from './i18n'
import './App.css'

const AssistantMarkdown = lazy(() => import('./AssistantMarkdown'))

const NAVIGATION = [
  { id: 'chat', label: 'chat', icon: MessageSquare },
  { id: 'memory', label: 'personalData', icon: Brain },
  { id: 'mood', label: 'mood', icon: HeartPulse },
  { id: 'knowledge', label: 'knowledge', icon: BookOpen },
  { id: 'operations', label: 'operations', icon: Activity, requiresOperationsAccess: true },
  { id: 'privacy', label: 'privacy', icon: ShieldCheck },
]

const ASSISTANT_NAME = 'Serenova'
const MODEL_PROFILES = {
  fast: { temperature: 0.4, maxNewTokens: 800 },
  balanced: { temperature: 0.7, maxNewTokens: 1600 },
  detailed: { temperature: 0.6, maxNewTokens: 2400 },
}
const FALLBACK_PROVIDER_CATALOG = [
  { id: 'openai_compatible', label: 'OpenAI-compatible', models: ['deepseek-chat'], default_model: 'deepseek-chat', default_base_url: '' },
  { id: 'anthropic', label: 'Anthropic (Claude)', models: ['claude-opus-5', 'claude-sonnet-5'], default_model: 'claude-opus-5', default_base_url: '' },
  { id: 'deepseek', label: 'DeepSeek', models: ['deepseek-chat', 'deepseek-reasoner'], default_model: 'deepseek-chat', default_base_url: 'https://api.deepseek.com' },
  { id: 'openai', label: 'OpenAI', models: ['gpt-4.1-mini', 'gpt-4.1', 'gpt-4o-mini', 'gpt-4o'], default_model: 'gpt-4.1-mini', default_base_url: '' },
  { id: 'openrouter', label: 'OpenRouter', models: ['openai/gpt-4.1-mini', 'openai/gpt-4o-mini'], default_model: 'openai/gpt-4.1-mini', default_base_url: 'https://openrouter.ai/api/v1' },
  { id: 'nvidia_nim', label: 'NVIDIA NIM', models: ['openai/gpt-oss-20b', 'meta/llama-3.1-8b-instruct'], default_model: 'openai/gpt-oss-20b', default_base_url: 'https://integrate.api.nvidia.com/v1' },
  { id: 'custom', label: 'Custom endpoint', models: ['deepseek-chat'], default_model: 'deepseek-chat', default_base_url: '' },
]

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('mindful-theme') || 'system')
  const [locale, setLocale] = useState(() => localStorage.getItem('mindful-locale') || 'en')
  const [session, setSession] = useState(null)
  const [activeView, setActiveView] = useState('chat')
  const [conversations, setConversations] = useState([])
  const [activeConversation, setActiveConversation] = useState(null)
  const [editingConversationId, setEditingConversationId] = useState(null)
  const [conversationTitleDraft, setConversationTitleDraft] = useState('')
  const [messages, setMessages] = useState([])
  const [quotedMessage, setQuotedMessage] = useState(null)
  const [draft, setDraft] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isSlow, setIsSlow] = useState(false)
  const [notice, setNotice] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [whyOpen, setWhyOpen] = useState(false)
  const [toneOpen, setToneOpen] = useState(false)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const [options, setOptions] = useState({
    provider: '', model: '', baseUrl: '', apiKey: '',
    useKnowledge: false, useStyle: true, stylePrefix: '', profile: 'balanced',
  })
  const [providerCatalog, setProviderCatalog] = useState(FALLBACK_PROVIDER_CATALOG)
  const [editableModelField, setEditableModelField] = useState({ baseUrl: false, apiKey: false })
  const [stylePrefixes, setStylePrefixes] = useState([])
  const messageEndRef = useRef(null)
  const activeRequestRef = useRef(null)
  const t = (key) => translate(locale, key)
  // 下拉框只翻译显示名；option 的 value 仍是文件名前缀，检索过滤依赖它。
  const styleName = (prefix) => t(`style_${prefix}`) === `style_${prefix}` ? prefix : t(`style_${prefix}`)
  const visibleStylePrefixes = stylePrefixes.filter((prefix) => prefix !== 'Gentle')
  const errorText = (code) => t(`error_${code}`) === `error_${code}` ? t('replyFailed') : t(`error_${code}`)
  const visibleNavigation = NAVIGATION.filter((item) => !item.requiresOperationsAccess || session?.can_access_operations)

  const hasMessages = messages.length > 0
  const activeTitle = activeConversation?.title || t('newConversation')
  const selectedProvider = useMemo(
    () => providerCatalog.find((provider) => provider.id === options.provider),
    [providerCatalog, options.provider],
  )
  const messagesById = useMemo(
    () => new Map(messages.filter((message) => message.id).map((message) => [message.id, message])),
    [messages],
  )
  const latestAssistantMessage = useMemo(
    () => [...messages].reverse().find((message) => message.role === 'assistant' && !message.pending && !message.failed),
    [messages],
  )
  const modelChoices = selectedProvider?.models || []
  const activeModelLabel = options.model || selectedProvider?.default_model || t('serverDefault')
  const activeToneLabel = options.stylePrefix ? styleName(options.stylePrefix) : t('toneDefault')

  function isLikelyBaseUrl(value) {
    const candidate = value.trim()
    if (!candidate) return true
    try {
      const url = new URL(candidate.includes('://') ? candidate : `https://${candidate}`)
      return Boolean(url.hostname) && (url.hostname.includes('.') || url.hostname === 'localhost' || url.hostname.includes(':') || Boolean(url.port))
    } catch {
      return false
    }
  }

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isSending])

  useEffect(() => {
    if (!isSending) { setIsSlow(false); return undefined }
    const timer = window.setTimeout(() => setIsSlow(true), 15_000)
    return () => window.clearTimeout(timer)
  }, [isSending])

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const applyTheme = () => { document.documentElement.dataset.theme = theme === 'system' ? (media.matches ? 'dark' : 'light') : theme }
    applyTheme()
    media.addEventListener('change', applyTheme)
    localStorage.setItem('mindful-theme', theme)
    return () => media.removeEventListener('change', applyTheme)
  }, [theme])

  useEffect(() => { document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en'; localStorage.setItem('mindful-locale', locale) }, [locale])

  useEffect(() => {
    if (activeView === 'operations' && !session?.can_access_operations) setActiveView('chat')
  }, [activeView, session])

  async function refreshConversations() {
    const response = await apiFetch('/api/v1/conversations')
    const data = await response.json()
    setConversations(data.conversations)
    return data.conversations
  }

  async function restoreSession() {
    try {
      const response = await apiFetch('/api/v1/auth/session')
      const data = await response.json()
      setSession(data)
      try {
        const preference = await readJson('/api/v1/style/preference')
        setStylePrefixes(preference.available || [])
        setOptions((current) => ({ ...current, stylePrefix: preference.style_prefix || '' }))
      } catch {
        // A missing style preference must not block sign-in.
      }
      await refreshConversations()
    } catch {
      setSession(null)
    }
  }

  useEffect(() => { restoreSession() }, [])

  // The catalog names the deployment's own endpoints, so it is session-scoped;
  // fetching it before sign-in would only 401.
  useEffect(() => {
    if (!session) return
    readJson('/api/v1/model/providers')
      .then((catalog) => {
        if (Array.isArray(catalog.providers) && catalog.providers.length) setProviderCatalog(catalog.providers)
      })
      .catch(() => {
        // Fallback catalog keeps the settings panel usable if the request fails.
      })
  }, [session])

  function changeProvider(providerId) {
    const provider = providerCatalog.find((item) => item.id === providerId)
    setOptions((current) => ({
      ...current,
      provider: providerId,
      model: providerId ? (provider?.default_model || '') : '',
      baseUrl: providerId ? (provider?.default_base_url || '') : '',
    }))
  }

  async function selectConversation(conversation) {
    setNotice('')
    try {
      const response = await apiFetch(`/api/v1/conversations/${conversation.id}`)
      const data = await response.json()
      setActiveConversation(data.conversation)
      setMessages(data.conversation.messages || [])
      setQuotedMessage(null)
    } catch (error) {
      setNotice(error.message)
    }
  }

  async function createConversation() {
    setNotice('')
    try {
      const response = await apiFetch('/api/v1/conversations', {
        method: 'POST', headers: csrfHeaders(), body: JSON.stringify({ title: t('newConversation') }),
      })
      const data = await response.json()
      const conversation = { ...data.conversation, messages: [] }
      setActiveConversation(conversation)
      setMessages([])
      setQuotedMessage(null)
      await refreshConversations()
    } catch (error) {
      setNotice(error.message)
    }
  }

  function beginConversationRename(event, conversation) {
    event.stopPropagation()
    setEditingConversationId(conversation.id)
    setConversationTitleDraft(conversation.title)
  }

  async function renameConversation(event, conversation) {
    event.preventDefault()
    const title = conversationTitleDraft.trim()
    if (!title) return
    setNotice('')
    try {
      const data = await (await apiFetch(`/api/v1/conversations/${conversation.id}`, {
        method: 'PUT', headers: csrfHeaders(), body: JSON.stringify({ title }),
      })).json()
      setConversations((current) => current.map((item) => item.id === conversation.id ? { ...item, ...data.conversation } : item))
      if (activeConversation?.id === conversation.id) setActiveConversation((current) => ({ ...current, ...data.conversation }))
      setEditingConversationId(null)
    } catch (error) {
      setNotice(error.message)
    }
  }

  async function removeConversation(event, conversation) {
    event.stopPropagation()
    if (isSending || !window.confirm(t('deleteConversationConfirm'))) return
    setNotice('')
    try {
      await apiFetch(`/api/v1/conversations/${conversation.id}`, { method: 'DELETE', headers: csrfHeaders() })
      setConversations((current) => current.filter((item) => item.id !== conversation.id))
      if (activeConversation?.id === conversation.id) {
        setActiveConversation(null)
        setMessages([])
        setQuotedMessage(null)
      }
    } catch (error) {
      setNotice(error.message)
    }
  }

  async function sendMessage(retryIndex = null, directMessage = null, moodCheckin = null, forceNewConversation = false) {
    const retryMessageIndex = Number.isInteger(retryIndex) ? retryIndex : null
    const moodCheckinContext = retryMessageIndex === null
      ? moodCheckin
      : messages[retryMessageIndex - 1]?.moodCheckin || null
    const text = retryMessageIndex === null
      ? String(directMessage ?? draft).trim()
      : String(messages[retryMessageIndex - 1]?.content || '').trim()
    const quoteForRequest = retryMessageIndex === null ? quotedMessage : null
    if (!text || isSending) return
    if (!isLikelyBaseUrl(options.baseUrl)) {
      setOptions((current) => ({ ...current, baseUrl: '', apiKey: '' }))
      setNotice(t('invalidModelCredentials'))
      return
    }
    if (retryMessageIndex === null) {
      setDraft('')
      setQuotedMessage(null)
    }
    setNotice('')
    setIsSending(true)
    const controller = new AbortController()
    activeRequestRef.current = controller

    let conversationId = forceNewConversation ? null : activeConversation?.id
    try {
      if (!conversationId) {
        const response = await apiFetch('/api/v1/conversations', {
          method: 'POST', headers: csrfHeaders(), body: JSON.stringify({ title: text.slice(0, 60) }),
        })
        const data = await response.json()
        conversationId = data.conversation.id
        setActiveConversation({ ...data.conversation, messages: [] })
      }

      if (retryMessageIndex === null) {
        setMessages((current) => [...current,
          {
            role: 'user', content: text, moodCheckin: moodCheckinContext,
            reply_to_message_id: quoteForRequest?.id || null, quotedMessage: quoteForRequest,
          },
          { role: 'assistant', content: '', pending: true },
        ])
      } else {
        setMessages((current) => current.map((item, index) => index === retryMessageIndex
          ? { ...item, content: '', pending: true, failed: false }
          : item))
      }
      const response = await apiFetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: csrfHeaders(),
        signal: controller.signal,
        body: JSON.stringify({
          message: text,
          conversation_id: conversationId,
          provider: options.provider || null,
          model: options.model || null,
          base_url: options.baseUrl || null,
          api_key: options.apiKey || null,
          temperature: MODEL_PROFILES[options.profile].temperature,
          max_new_tokens: MODEL_PROFILES[options.profile].maxNewTokens,
          use_knowledge: moodCheckinContext ? false : options.useKnowledge,
          use_style: options.useStyle,
          style_prefix: options.stylePrefix,
          show_memory_receipt: false,
          mood_checkin: moodCheckinContext ? {
            date: moodCheckinContext.date,
            mood: moodCheckinContext.mood,
            intensity: moodCheckinContext.intensity,
            note: moodCheckinContext.note || '',
          } : null,
          quoted_message_id: quoteForRequest?.id || null,
          retry_last_response: retryMessageIndex !== null,
        }),
      })
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let receivedError = ''
      let receivedText = false
      let receivedRagStatus = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop() || ''
        for (const event of events) {
          const eventName = event.match(/^event: (.+)$/m)?.[1]
          const data = event.match(/^data: (.+)$/m)?.[1]
          if (!eventName || !data) continue
          const payload = JSON.parse(data)
          if (eventName === 'chunk') {
            receivedText ||= Boolean(payload.text)
            setMessages((current) => current.map((item, index) => index === current.length - 1
              ? { ...item, content: item.content + payload.text, pending: false }
              : item))
          }
          if (eventName === 'archived') {
            setMessages((current) => current.map((item, index) => {
              if (index === current.length - 2) return { ...item, id: payload.user_message_id }
              if (index === current.length - 1) return { ...item, id: payload.assistant_message_id }
              return item
            }))
          }
          if (eventName === 'citations') {
            setMessages((current) => current.map((item, index) => index === current.length - 1
              ? { ...item, citations: payload.citations || [], citationTraceId: payload.trace_id || null }
              : item))
          }
          if (eventName === 'rag_status') {
            receivedRagStatus = Boolean(payload.enforced)
            setMessages((current) => current.map((item, index) => index === current.length - 1
              ? { ...item, pending: false, ragStatus: payload }
              : item))
          }
          if (eventName === 'error') receivedError = { code: payload.code || 'generation_failed', retryable: Boolean(payload.retryable), message: payload.message }
        }
      }
      if (receivedError) throw new ApiRequestError(receivedError.message || receivedError.code, receivedError)
      if (!receivedText && !receivedRagStatus) throw new Error(t('emptyResponse'))
      await refreshConversations()
    } catch (error) {
      const cancelled = error.name === 'AbortError'
      const errorCode = error.code || 'generation_failed'
      const retryable = Boolean(error.retryable) || errorCode === 'generation_failed'
      setNotice(cancelled ? t('requestCancelled') : errorText(errorCode))
      setMessages((current) => {
        const responseIndex = retryMessageIndex ?? current.length - 1
        return current.map((item, index) => index === responseIndex
          ? { ...item, content: item.content || (cancelled ? '' : errorText(errorCode)), pending: false, failed: !cancelled, retryable: !cancelled && retryable, errorCode }
          : item).filter((item, index) => !(cancelled && index === responseIndex && item.role === 'assistant' && !item.content))
      })
    } finally {
      setIsSending(false)
      activeRequestRef.current = null
    }
  }

  async function saveStylePrefix(stylePrefix) {
    // Update the UI immediately, then persist so the choice survives a reload.
    setOptions((current) => ({ ...current, stylePrefix }))
    try {
      await readJson('/api/v1/style/preference', {
        method: 'PUT', headers: csrfHeaders(), body: JSON.stringify({ style_prefix: stylePrefix }),
      })
    } catch (error) {
      setNotice(error.message)
    }
  }

  function cancelGeneration() { activeRequestRef.current?.abort() }

  async function copyReply(content) {
    try {
      await navigator.clipboard.writeText(content)
      setNotice(t('copied'))
    } catch {
      setNotice(t('copyFailed'))
    }
  }

  async function submitRagFeedback(index, helpful) {
    const message = messages[index]
    if (!message?.citationTraceId) return
    try {
      const response = await apiFetch('/api/v1/rag/feedback', {
        method: 'POST', headers: csrfHeaders(),
        body: JSON.stringify({ trace_id: message.citationTraceId, helpful }),
      })
      if (!response.ok) throw new Error('feedback failed')
      setMessages((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, ragFeedback: helpful } : item))
    } catch {
      setNotice(t('feedbackFailed'))
    }
  }

  async function logout() {
    try {
      await apiFetch('/api/v1/auth/logout', { method: 'POST', headers: csrfHeaders() })
    } catch {
      // Clear the local view even if a stale session has already expired.
    }
    setSession(null)
    setConversations([])
    setActiveConversation(null)
    setMessages([])
  }

  if (!session) return <LoginScreen onSuccess={restoreSession} t={t} />

  const activeNavigationItem = visibleNavigation.find((item) => item.id === activeView)
  const isChatView = activeView === 'chat'
  const quoteMessage = (message) => {
    if (!message?.id || isSending) return
    setQuotedMessage({ id: message.id, role: message.role, content: message.content })
  }
  const mobileTitle = activeView === 'chat' ? activeTitle : t(activeNavigationItem?.label || 'chat')
  const navigateWorkspace = (viewId) => {
    setActiveView(viewId)
    setMobileSidebarOpen(false)
    setSettingsOpen(false)
    setWhyOpen(false)
    setToneOpen(false)
  }
  const startConversation = async () => {
    await createConversation()
    setActiveView('chat')
    setMobileSidebarOpen(false)
  }
  const startMoodReflection = async (record) => {
    if (isSending) return
    setActiveView('chat')
    setMobileSidebarOpen(false)
    setActiveConversation(null)
    setMessages([])
    setQuotedMessage(null)
    await sendMessage(null, t('moodReflectionRequest'), record, true)
  }
  const openConversation = async (conversation) => {
    await selectConversation(conversation)
    setActiveView('chat')
    setMobileSidebarOpen(false)
  }

  return (
    <main className={`app-shell ${isChatView ? 'chat-workspace' : 'feature-workspace'} ${settingsOpen ? 'settings-open' : 'settings-closed'} ${whyOpen ? 'context-open' : ''} ${mobileSidebarOpen ? 'mobile-sidebar-open' : ''}`}>
      <NavigationRail activeView={activeView} navigateWorkspace={navigateWorkspace} t={t} visibleNavigation={visibleNavigation} />
      <section className="workspace-frame">
        <WorkspaceTopbar locale={locale} setLocale={setLocale} setTheme={setTheme} t={t} theme={theme} />
        <div className={`workspace-content ${isChatView ? 'chat-workspace-content' : 'feature-workspace-content'} ${whyOpen ? 'reply-context-open' : ''}`}>
      {isChatView && <WorkspaceSidebar
        className="sidebar desktop-sidebar conversation-sidebar"
        activeConversation={activeConversation}
        activeView={activeView}
        beginConversationRename={beginConversationRename}
        conversationTitleDraft={conversationTitleDraft}
        conversations={conversations}
        editingConversationId={editingConversationId}
        isSending={isSending}
        locale={locale}
        logout={logout}
        navigateWorkspace={navigateWorkspace}
        onCloseMobile={() => setMobileSidebarOpen(false)}
        removeConversation={removeConversation}
        renameConversation={renameConversation}
        session={session}
        setConversationTitleDraft={setConversationTitleDraft}
        setEditingConversationId={setEditingConversationId}
        setLocale={setLocale}
        setTheme={setTheme}
        startConversation={startConversation}
        openConversation={openConversation}
        t={t}
        theme={theme}
        visibleNavigation={visibleNavigation}
      />}
      {mobileSidebarOpen && isChatView && <>
        <button className="mobile-sidebar-backdrop" aria-label={t('closeSidebar')} onClick={() => setMobileSidebarOpen(false)} />
        <WorkspaceSidebar
          className="sidebar mobile-sidebar open"
          activeConversation={activeConversation}
          activeView={activeView}
          beginConversationRename={beginConversationRename}
          conversationTitleDraft={conversationTitleDraft}
          conversations={conversations}
          editingConversationId={editingConversationId}
          isSending={isSending}
          locale={locale}
          logout={logout}
          navigateWorkspace={navigateWorkspace}
          onCloseMobile={() => setMobileSidebarOpen(false)}
          removeConversation={removeConversation}
          renameConversation={renameConversation}
          session={session}
          setConversationTitleDraft={setConversationTitleDraft}
          setEditingConversationId={setEditingConversationId}
          setLocale={setLocale}
          setTheme={setTheme}
          startConversation={startConversation}
          openConversation={openConversation}
          t={t}
          theme={theme}
          visibleNavigation={visibleNavigation}
        />
      </>}

      <div className="mobile-app-bar">{isChatView && <button className="icon-button" title={t('openSidebar')} onClick={() => setMobileSidebarOpen(true)}><Menu size={19} /></button>}<span>{mobileTitle}</span></div>
      {isChatView ? <><section className="chat-panel">
        <header className="chat-header"><div className="chat-title-block"><h1>{activeTitle}</h1><div className="chat-summary-row"><div className="tone-control"><button className={`conversation-summary tone-summary ${toneOpen ? 'active' : ''}`} aria-expanded={toneOpen} onClick={() => { setSettingsOpen(false); setToneOpen((open) => !open) }}><Sparkles size={13} />{t('conversationTone')}：{activeToneLabel}<ChevronDown size={13} /></button>{toneOpen && <ToneMenu activeStyle={options.stylePrefix} onClose={() => setToneOpen(false)} onSelect={saveStylePrefix} styleName={styleName} styles={visibleStylePrefixes} t={t} />}</div><button className="conversation-summary model-summary" aria-label={t('modelResponseSettings')} onClick={() => { setToneOpen(false); setWhyOpen(false); setSettingsOpen(true) }}><Settings2 size={13} />{activeModelLabel} · {activeToneLabel}{options.useKnowledge && <> · {t('knowledgeShort')}</>}<ChevronDown size={13} /></button></div></div><div className="chat-header-actions"><button className={`icon-button context-toggle ${whyOpen ? 'active' : ''}`} title={t('replyContext')} aria-label={t('replyContext')} onClick={() => { setSettingsOpen(false); setToneOpen(false); setWhyOpen((open) => !open) }}><Info size={18} /></button><button className="icon-button settings-toggle" title={settingsOpen ? t('hidePreferences') : t('modelResponseSettings')} onClick={() => { setWhyOpen(false); setToneOpen(false); setSettingsOpen((open) => !open) }}>{settingsOpen ? <ChevronLeft size={18} /> : <Settings2 size={18} />}</button></div></header>
        <div className="messages" aria-live="polite">
          {!hasMessages && <Welcome onPrompt={sendMessage} t={t} />}
          {messages.map((message, index) => <Message key={message.id || `${message.role}-${index}`} message={message} index={index} quotedMessage={message.quotedMessage || messagesById.get(message.reply_to_message_id)} canRegenerate={message.role === 'assistant' && index === messages.length - 1 && messages[index - 1]?.role === 'user'} onCopy={copyReply} onQuote={quoteMessage} onRetry={sendMessage} onRagFeedback={submitRagFeedback} isSending={isSending} t={t} />)}
          <div ref={messageEndRef} />
        </div>
        {notice && <div className="notice" role="alert">{notice}<button onClick={() => setNotice('')} title="Dismiss"><X size={15} /></button></div>}
        {isSlow && <div className="slow-request" role="status">{t('stillWaiting')}</div>}
        <Composer draft={draft} setDraft={setDraft} sendMessage={sendMessage} isSending={isSending} cancelGeneration={cancelGeneration} quotedMessage={quotedMessage} clearQuote={() => setQuotedMessage(null)} t={t} />
      </section>

      {whyOpen && <ReplyContextPanel latestMessage={latestAssistantMessage} onClose={() => setWhyOpen(false)} options={options} tone={activeToneLabel} t={t} />}
      {settingsOpen && <ModelSettingsPanel activeModelLabel={activeModelLabel} changeProvider={changeProvider} editableModelField={editableModelField} modelChoices={modelChoices} onClose={() => setSettingsOpen(false)} options={options} providerCatalog={providerCatalog} saveStylePrefix={saveStylePrefix} setEditableModelField={setEditableModelField} setOptions={setOptions} styleName={styleName} stylePrefixes={visibleStylePrefixes} t={t} />}</> : <section className="feature-main">{activeView === 'memory' && <MemoryPage t={t} locale={locale} />}{activeView === 'mood' && <MoodPage t={t} onReflect={startMoodReflection} locale={locale} />}{activeView === 'knowledge' && <KnowledgePage t={t} canManageKnowledge={session?.can_manage_knowledge} />}{activeView === 'operations' && <OperationsPage t={t} />}{activeView === 'privacy' && <PrivacyPage t={t} onDeleted={logout} />}</section>}
        </div>
      </section>
    </main>
  )
}

function WorkspaceTopbar({ locale, setLocale, setTheme, t, theme }) {
  const activeTheme = theme === 'light' ? 'light' : 'dark'
  return <header className="workspace-topbar">
    <div className="workspace-brand" aria-label={ASSISTANT_NAME}><Sparkles size={18} aria-hidden="true" /><span>{ASSISTANT_NAME}</span></div>
    <div className="workspace-topbar-controls">
      <span>{t('language')}</span>
      <div className="workspace-segment" role="group" aria-label={t('language')}>
        <button className={locale === 'zh' ? 'selected' : ''} aria-pressed={locale === 'zh'} onClick={() => setLocale('zh')}>{t('chinese')}</button>
        <button className={locale === 'en' ? 'selected' : ''} aria-pressed={locale === 'en'} onClick={() => setLocale('en')}>{t('english')}</button>
      </div>
      <span>{t('theme')}</span>
      <div className="workspace-segment" role="group" aria-label={t('theme')}>
        <button className={activeTheme === 'dark' ? 'selected' : ''} aria-pressed={activeTheme === 'dark'} onClick={() => setTheme('dark')}>{t('dark')}</button>
        <button className={activeTheme === 'light' ? 'selected' : ''} aria-pressed={activeTheme === 'light'} onClick={() => setTheme('light')}>{t('light')}</button>
      </div>
    </div>
  </header>
}

function NavigationRail({ activeView, navigateWorkspace, t, visibleNavigation }) {
  return <aside className="navigation-rail" aria-label={t('workspaceNavigation')}>
    <div className="rail-brand" aria-label={ASSISTANT_NAME}><Sparkles size={19} /></div>
    <nav className="rail-nav">
      {visibleNavigation.map(({ id, label, icon: Icon }) => <button key={id} className={`rail-link ${activeView === id ? 'selected' : ''}`} title={t(label)} aria-label={t(label)} onClick={() => navigateWorkspace(id)}><Icon size={18} /><span>{t(label)}</span></button>)}
    </nav>
  </aside>
}

function ToneMenu({ activeStyle, onClose, onSelect, styleName, styles, t }) {
  const availableStyles = styles.length ? styles : ['温柔型', '专业型']
  const selectStyle = async (style) => { await onSelect(style); onClose() }
  return <div className="tone-menu" role="menu" aria-label={t('conversationTone')}>
    <p>{t('toneMenuHelp')}</p>
    {availableStyles.map((style) => <button key={style} className={activeStyle === style ? 'selected' : ''} role="menuitemradio" aria-checked={activeStyle === style} onClick={() => selectStyle(style)}>{styleName(style)}</button>)}
  </div>
}

function ModelSettingsPanel({ activeModelLabel, changeProvider, editableModelField, modelChoices, onClose, options, providerCatalog, saveStylePrefix, setEditableModelField, setOptions, styleName, stylePrefixes, t }) {
  return <aside className="settings-panel model-settings-panel open" aria-label={t('modelResponseSettings')}>
    <div className="settings-title"><div><PanelRight size={18} /><h2>{t('modelResponseSettings')}</h2></div><button className="icon-button" title={t('hidePreferences')} onClick={onClose}><X size={18} /></button></div>
    <p className="settings-copy">{t('modelSettingsCopy')}</p>
    <section className="settings-group"><h3>{t('provider')}</h3><div className="provider-choice-list"><button className={!options.provider ? 'selected' : ''} onClick={() => changeProvider('')}><span>{t('serverDefault')}</span><small>{activeModelLabel}</small></button>{providerCatalog.map((provider) => <button key={provider.id} className={options.provider === provider.id ? 'selected' : ''} onClick={() => changeProvider(provider.id)}><span>{provider.label}</span><small>{provider.models.length} {t('models')}</small></button>)}</div></section>
    <section className="settings-group"><h3>{t('model')}</h3><div className="model-choice-list">{modelChoices.length ? modelChoices.map((model) => <button key={model} className={options.model === model ? 'selected' : ''} onClick={() => setOptions((current) => ({ ...current, model }))}>{model}{options.model === model && <Check size={15} />}</button>) : <p>{t('serverDefault')}</p>}</div></section>
    <section className="settings-group"><h3>{t('responseProfile')}</h3><div className="profile-choice-list">{['fast', 'balanced', 'detailed'].map((profile) => <button key={profile} className={options.profile === profile ? 'selected' : ''} onClick={() => setOptions((current) => ({ ...current, profile }))}>{t(`profile${profile[0].toUpperCase()}${profile.slice(1)}`)}</button>)}</div></section>
    {stylePrefixes.length > 0 && <section className="settings-group"><h3>{t('conversationTone')}</h3><div className="profile-choice-list">{stylePrefixes.map((prefix) => <button key={prefix} className={options.stylePrefix === prefix ? 'selected' : ''} onClick={() => saveStylePrefix(prefix)}>{styleName(prefix)}</button>)}</div></section>}
    <Toggle label={t('knowledgeRetrieval')} description={t('knowledgeHint')} checked={options.useKnowledge} onChange={(useKnowledge) => setOptions((current) => ({ ...current, useKnowledge }))} />
    <Toggle label={t('styleReference')} description={t('styleHint')} checked={options.useStyle} onChange={(useStyle) => setOptions((current) => ({ ...current, useStyle }))} />
    <details className="advanced-model-settings"><summary>{t('advancedConnection')}</summary><label className="model-field">{t('baseUrl')}<input name="llm-base-url" autoComplete="new-password" data-1p-ignore="true" data-lpignore="true" readOnly={!editableModelField.baseUrl} value={options.baseUrl} placeholder={t('optionalEndpoint')} onFocus={() => setEditableModelField((current) => ({ ...current, baseUrl: true }))} onChange={(event) => setOptions((current) => ({ ...current, baseUrl: event.target.value }))} /></label><label className="model-field">{t('apiKey')}<input name="llm-api-key" type="password" autoComplete="new-password" data-1p-ignore="true" data-lpignore="true" readOnly={!editableModelField.apiKey} value={options.apiKey} placeholder={t('tabOnly')} onFocus={() => setEditableModelField((current) => ({ ...current, apiKey: true }))} onChange={(event) => setOptions((current) => ({ ...current, apiKey: event.target.value }))} /></label></details>
  </aside>
}

function ReplyContextPanel({ latestMessage, onClose, options, tone, t }) {
  const citations = latestMessage?.citations || []
  return <aside className="reply-context open" aria-label={t('replyContext')}>
    <div className="context-title"><div><Info size={18} /><h2>{t('replyContext')}</h2></div><button className="icon-button" title={t('closeContext')} onClick={onClose}><X size={18} /></button></div>
    <p className="context-copy">{t('replyContextDescription')}</p>
    <section className="context-section gentle-context"><h3>{t('responseApproach')}</h3><p>{t('responseApproachCopy').replace('{tone}', tone)}</p><p>{t('contextPrivacyNote')}</p></section>
    <section className="context-section"><h3>{t('sources')}</h3>{citations.length ? <div className="context-citations">{citations.map((citation, index) => <span key={`${citation.source || citation}-${index}`}>{citation.source || citation}</span>)}</div> : <p>{t('contextNoSources')}</p>}</section>
    <section className="context-section"><h3>{t('contextSettings')}</h3><dl className="context-settings"><div><dt>{t('knowledgeRetrieval')}</dt><dd>{options.useKnowledge ? t('contextOn') : t('contextOff')}</dd></div><div><dt>{t('styleReference')}</dt><dd>{options.useStyle ? t('contextOn') : t('contextOff')}</dd></div></dl></section>
  </aside>
}

function WorkspaceSidebar({
  activeConversation,
  activeView,
  beginConversationRename,
  className,
  conversations,
  conversationTitleDraft,
  editingConversationId,
  isSending,
  locale,
  logout,
  navigateWorkspace,
  onCloseMobile,
  openConversation,
  removeConversation,
  renameConversation,
  session,
  setConversationTitleDraft,
  setEditingConversationId,
  setLocale,
  setTheme,
  startConversation,
  t,
  theme,
  visibleNavigation,
}) {
  return (
    <aside className={className} aria-label={t('workspaceNavigation')}>
      <div className="sidebar-title-row"><button className="icon-button mobile-sidebar-close" title={t('closeSidebar')} onClick={onCloseMobile}><X size={17} /></button></div>
      <nav className="workspace-nav">
        {visibleNavigation.map(({ id, label, icon: Icon }) => (
          <button key={id} className={`workspace-link ${activeView === id ? 'selected' : ''}`} onClick={() => navigateWorkspace(id)}><Icon size={16} /><span>{t(label)}</span></button>
        ))}
      </nav>
      <button className="new-chat" onClick={startConversation}><CirclePlus size={18} />{t('newChat')}</button>
      <div className="conversation-list">
        <p className="section-label">{t('conversations')}</p>
        {conversations.length === 0 && <p className="empty-list">{t('noChats')}</p>}
        {conversations.map((conversation) => editingConversationId === conversation.id ? (
          <form className="conversation-edit" key={conversation.id} onSubmit={(event) => renameConversation(event, conversation)}><input value={conversationTitleDraft} onChange={(event) => setConversationTitleDraft(event.target.value)} aria-label={t('conversationTitle')} autoFocus /><button className="conversation-action" title={t('saveTitle')}><Check size={15} /></button><button className="conversation-action" type="button" title={t('cancel')} onClick={() => setEditingConversationId(null)}><X size={15} /></button></form>
        ) : (
          <div className={`conversation-row ${conversation.id === activeConversation?.id ? 'selected' : ''}`} key={conversation.id}>
            <button className="conversation" onClick={() => openConversation(conversation)}><MessageSquare size={15} /><span>{conversation.title}</span></button>
            <div className="conversation-actions"><button className="conversation-action" title={t('renameConversation')} onClick={(event) => beginConversationRename(event, conversation)}><Pencil size={15} /></button><button className="conversation-action delete-conversation" title={t('deleteConversation')} onClick={(event) => removeConversation(event, conversation)} disabled={isSending}><Trash2 size={15} /></button></div>
          </div>
        ))}
      </div>
      <PreferencesControls theme={theme} setTheme={setTheme} locale={locale} setLocale={setLocale} t={t} />
      <div className="sidebar-footer">
        <div className="identity"><span className="avatar">{session.user_id.slice(0, 1).toUpperCase()}</span><span>{session.user_id}</span></div>
        <button className="icon-button" title={t('logout')} onClick={logout}><LogOut size={17} /></button>
      </div>
    </aside>
  )
}

function LoginScreen({ onSuccess, t }) {
  const [userId, setUserId] = useState('')
  const [accessKey, setAccessKey] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function submit(event) {
    event.preventDefault()
    setLoading(true); setError('')
    try {
      await apiFetch('/api/v1/auth/login', { method: 'POST', body: JSON.stringify({ user_id: userId, access_key: accessKey }) })
      await onSuccess()
    } catch (requestError) { setError(requestError.message) } finally { setLoading(false) }
  }

  return <main className="login-page"><section className="login-intro"><div className="brand"><Sparkles size={20} /><span>{ASSISTANT_NAME}</span></div><div><h1>{t('loginTitle')}</h1><p>{t('loginText')}</p></div><div className="intro-mark"><Bot size={38} /></div></section><form className="login-form" onSubmit={submit}><h2>{t('welcomeBack')}</h2><p>{t('loginHelp')}</p><label>{t('userId')}<input value={userId} onChange={(event) => setUserId(event.target.value)} autoComplete="username" required /></label><label>{t('password')}<input type="password" value={accessKey} onChange={(event) => setAccessKey(event.target.value)} autoComplete="current-password" required /></label>{error && <div className="login-error">{error}</div>}<button className="login-button" disabled={loading}>{loading ? <LoaderCircle className="spin" size={18} /> : t('signIn')}</button></form></main>
}

function Welcome({ onPrompt, t }) { return <div className="welcome"><div className="welcome-icon"><Sparkles size={24} /></div><h2>{t('welcomeTitle')}</h2><p>{t('welcomeText')}</p><div className="prompt-row"><button onClick={() => onPrompt(null, t('talkPromptMessage'))}>{t('talkPrompt')}</button><button onClick={() => onPrompt(null, t('planPromptMessage'))}>{t('planPrompt')}</button></div></div> }

function Message({ message, index, quotedMessage, canRegenerate, onCopy, onQuote, onRetry, onRagFeedback, isSending, t }) {
  // Only an enforced refusal replaces the reply. When RAG_REQUIRE_EVIDENCE is off
  // the server still reports `insufficient`, but it also answers normally.
  const insufficientEvidence = Boolean(message.ragStatus?.enforced)
  const displayContent = insufficientEvidence ? '' : message.content
  const canRetry = !insufficientEvidence && (message.failed ? message.retryable : canRegenerate && Boolean(displayContent))

  return <article className={`message ${message.role} ${message.failed ? 'failed' : ''}`}>
    <div className="message-avatar">{message.role === 'assistant' ? <Sparkles size={15} /> : 'You'}</div>
    <div>
      <div className={`message-body ${message.role === 'assistant' && !message.failed ? 'assistant-markdown' : ''}`}>
        {message.pending && !displayContent ? <span className="typing"><i /><i /><i /></span>
          : insufficientEvidence ? <p className="rag-evidence-status" role="status">{t('ragInsufficientEvidence')}</p>
            : message.role === 'assistant' && !message.failed ? <Suspense fallback={displayContent}><AssistantMarkdown content={displayContent} /></Suspense>
              : displayContent}
      </div>
      {quotedMessage && <div className="message-quote"><Reply size={13} /><span>{quotedMessage.role === 'assistant' ? t('quotedAssistant') : t('quotedUser')}</span><p>{quotedMessage.content}</p></div>}
      {message.role === 'assistant' && message.citations?.length > 0 && <div className="rag-citations"><span>{t('sources')}</span>{message.citations.map((citation) => <span className="rag-citation" title={citation.excerpt} key={`${citation.source}-${citation.chunk_index}`}>{citation.source}</span>)}</div>}
      {!message.pending && <div className="message-actions">
        {message.id && <button className="message-action" title={t('quoteMessage')} disabled={isSending} onClick={() => onQuote(message)}><Reply size={14} /></button>}
        {displayContent && !message.failed && <button className="message-action" title={t('copyReply')} onClick={() => onCopy(displayContent)}><Copy size={14} /></button>}
        {message.role === 'assistant' && <>{message.citationTraceId && <><button className={`message-action ${message.ragFeedback === true ? 'selected' : ''}`} title={t('ragHelpful')} onClick={() => onRagFeedback(index, true)}><ThumbsUp size={14} /></button><button className={`message-action ${message.ragFeedback === false ? 'selected' : ''}`} title={t('ragUnhelpful')} onClick={() => onRagFeedback(index, false)}><ThumbsDown size={14} /></button></>}{canRetry && <button className="message-action" title={t('retryGeneration')} disabled={isSending} onClick={() => onRetry(index)}><RefreshCw size={14} /></button>}</>}
      </div>}
    </div>
  </article>
}

function Composer({ draft, setDraft, sendMessage, isSending, cancelGeneration, quotedMessage, clearQuote, t }) { return <div className="composer-wrap">{quotedMessage && <div className="composer-quote"><Reply size={14} /><div><span>{quotedMessage.role === 'assistant' ? t('quotedAssistant') : t('quotedUser')}</span><p>{quotedMessage.content}</p></div><button type="button" className="icon-button" title={t('removeQuote')} onClick={clearQuote}><X size={15} /></button></div>}<div className="composer"><textarea value={draft} rows="1" placeholder={t('composer')} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage() } }} />{isSending ? <button type="button" className="send-button stop-button" title={t('stopGenerating')} onClick={cancelGeneration}><Square size={15} fill="currentColor" /></button> : <button type="button" className="send-button" title={t('composer')} disabled={!draft.trim()} onClick={() => sendMessage()}><SendHorizontal size={18} /></button>}</div></div> }

function PreferencesControls({ theme, setTheme, locale, setLocale, t }) { return <div className="preferences-controls"><label><SunMoon size={15} /><span>{t('theme')}</span><select value={theme} onChange={(event) => setTheme(event.target.value)}><option value="light">{t('light')}</option><option value="dark">{t('dark')}</option><option value="system">{t('system')}</option></select></label><label><Languages size={15} /><span>{t('language')}</span><select value={locale} onChange={(event) => setLocale(event.target.value)}><option value="en">{t('english')}</option><option value="zh">{t('chinese')}</option></select></label></div> }

function Toggle({ label, description, checked, onChange }) { return <label className="toggle"><span><b>{label}</b><small>{description}</small></span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><i /></label> }

export default App
