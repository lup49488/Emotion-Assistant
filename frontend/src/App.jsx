import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  BookOpen,
  Brain,
  Bot,
  ChevronLeft,
  Check,
  CirclePlus,
  Copy,
  HeartPulse,
  LoaderCircle,
  LogOut,
  MessageSquare,
  PanelRight,
  Pencil,
  RefreshCw,
  SendHorizontal,
  Settings2,
  ShieldCheck,
  Square,
  Sparkles,
  Trash2,
  Languages,
  SunMoon,
  X,
} from 'lucide-react'
import { API_BASE_URL, ApiRequestError, apiFetch, csrfHeaders } from './api'
import { KnowledgePage, MemoryPage, MoodPage, OperationsPage, PrivacyPage } from './FeaturePages'
import { translate } from './i18n'
import './App.css'

const NAVIGATION = [
  { id: 'chat', label: 'chat', icon: MessageSquare },
  { id: 'memory', label: 'personalData', icon: Brain },
  { id: 'mood', label: 'mood', icon: HeartPulse },
  { id: 'knowledge', label: 'knowledge', icon: BookOpen },
  { id: 'operations', label: 'operations', icon: Activity, requiresOperationsAccess: true },
  { id: 'privacy', label: 'privacy', icon: ShieldCheck },
]

const ASSISTANT_NAME = 'Serenova'

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
  const [draft, setDraft] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isSlow, setIsSlow] = useState(false)
  const [notice, setNotice] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(() => window.innerWidth > 1060)
  const [options, setOptions] = useState({
    provider: '', model: '', baseUrl: '', apiKey: '',
    useKnowledge: false, useStyle: false, temperature: 0.8,
  })
  const messageEndRef = useRef(null)
  const activeRequestRef = useRef(null)
  const t = (key) => translate(locale, key)
  const errorText = (code) => t(`error_${code}`) === `error_${code}` ? t('replyFailed') : t(`error_${code}`)
  const visibleNavigation = NAVIGATION.filter((item) => !item.requiresOperationsAccess || session?.can_access_operations)

  const hasMessages = messages.length > 0
  const activeTitle = activeConversation?.title || t('newConversation')
  const apiLabel = useMemo(() => (API_BASE_URL || window.location.host).replace(/^https?:\/\//, ''), [])

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
      await refreshConversations()
    } catch {
      setSession(null)
    }
  }

  useEffect(() => { restoreSession() }, [])

  async function selectConversation(conversation) {
    setNotice('')
    try {
      const response = await apiFetch(`/api/v1/conversations/${conversation.id}`)
      const data = await response.json()
      setActiveConversation(data.conversation)
      setMessages(data.conversation.messages || [])
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
      }
    } catch (error) {
      setNotice(error.message)
    }
  }

  async function sendMessage(retryIndex = null) {
    const retryMessageIndex = Number.isInteger(retryIndex) ? retryIndex : null
    const text = retryMessageIndex === null ? draft.trim() : String(messages[retryMessageIndex - 1]?.content || '').trim()
    if (!text || isSending) return
    if (retryMessageIndex === null) setDraft('')
    setNotice('')
    setIsSending(true)
    const controller = new AbortController()
    activeRequestRef.current = controller

    let conversationId = activeConversation?.id
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
        setMessages((current) => [...current, { role: 'user', content: text }, { role: 'assistant', content: '', pending: true }])
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
          temperature: options.temperature,
          use_knowledge: options.useKnowledge,
          use_style: options.useStyle,
          show_memory_receipt: false,
          retry_last_response: retryMessageIndex !== null,
        }),
      })
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let receivedError = ''
      let receivedText = false

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
          if (eventName === 'error') receivedError = { code: payload.code || 'generation_failed', retryable: Boolean(payload.retryable), message: payload.message }
        }
      }
      if (receivedError) throw new ApiRequestError(receivedError.message || receivedError.code, receivedError)
      if (!receivedText) throw new Error(t('emptyResponse'))
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

  function cancelGeneration() { activeRequestRef.current?.abort() }

  async function copyReply(content) {
    try {
      await navigator.clipboard.writeText(content)
      setNotice(t('copied'))
    } catch {
      setNotice(t('copyFailed'))
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

  return (
    <main className={`app-shell ${settingsOpen ? 'settings-open' : 'settings-closed'}`}>
      <aside className="sidebar" aria-label="Workspace navigation">
        <div className="brand"><Sparkles size={18} aria-hidden="true" /><span>{ASSISTANT_NAME}</span></div>
        <nav className="workspace-nav">{visibleNavigation.map(({ id, label, icon: Icon }) => <button key={id} className={`workspace-link ${activeView === id ? 'selected' : ''}`} onClick={() => setActiveView(id)}><Icon size={16} />{t(label)}</button>)}</nav>
        {activeView === 'chat' && <><button className="new-chat" onClick={createConversation}><CirclePlus size={18} />{t('newChat')}</button><div className="conversation-list"><p className="section-label">{t('conversations')}</p>{conversations.length === 0 && <p className="empty-list">{t('noChats')}</p>}{conversations.map((conversation) => editingConversationId === conversation.id ? (
          <form className="conversation-edit" key={conversation.id} onSubmit={(event) => renameConversation(event, conversation)}><input value={conversationTitleDraft} onChange={(event) => setConversationTitleDraft(event.target.value)} aria-label={t('conversationTitle')} autoFocus /><button className="conversation-action" title={t('saveTitle')}><Check size={15} /></button><button className="conversation-action" type="button" title={t('cancel')} onClick={() => setEditingConversationId(null)}><X size={15} /></button></form>
        ) : <div className={`conversation-row ${conversation.id === activeConversation?.id ? 'selected' : ''}`} key={conversation.id}><button className="conversation" onClick={() => selectConversation(conversation)}><MessageSquare size={15} /><span>{conversation.title}</span></button><div className="conversation-actions"><button className="conversation-action" title={t('renameConversation')} onClick={(event) => beginConversationRename(event, conversation)}><Pencil size={14} /></button><button className="conversation-action delete-conversation" title={t('deleteConversation')} onClick={(event) => removeConversation(event, conversation)} disabled={isSending}><Trash2 size={14} /></button></div></div>
        )}</div></>}
        {activeView !== 'chat' && <div className="sidebar-space" />}
        <PreferencesControls theme={theme} setTheme={setTheme} locale={locale} setLocale={setLocale} t={t} />
        <div className="sidebar-footer">
          <div className="identity"><span className="avatar">{session.user_id.slice(0, 1).toUpperCase()}</span><span>{session.user_id}</span></div>
          <button className="icon-button" title={t('logout')} onClick={logout}><LogOut size={17} /></button>
        </div>
      </aside>

      <nav className="mobile-workspace-nav" aria-label="Workspace navigation"><div className="mobile-tab-list">{visibleNavigation.map(({ id, label, icon: Icon }) => <button key={id} className={activeView === id ? 'selected' : ''} onClick={() => setActiveView(id)}><Icon size={15} /><span>{t(label)}</span></button>)}</div><div className="mobile-preference-row"><select className="mobile-preference" aria-label={t('theme')} value={theme} onChange={(event) => setTheme(event.target.value)}><option value="light">{t('light')}</option><option value="dark">{t('dark')}</option><option value="system">{t('system')}</option></select><select className="mobile-preference" aria-label={t('language')} value={locale} onChange={(event) => setLocale(event.target.value)}><option value="en">EN</option><option value="zh">中文</option></select></div></nav>
      {activeView === 'chat' ? <><section className="chat-panel">
        <header className="chat-header"><div><h1>{activeTitle}</h1><p>{t('connected')} {apiLabel}</p></div><button className="icon-button settings-toggle" title={settingsOpen ? t('hidePreferences') : t('preferences')} onClick={() => setSettingsOpen((open) => !open)}>{settingsOpen ? <ChevronLeft size={18} /> : <Settings2 size={18} />}</button></header>
        <div className="messages" aria-live="polite">
          {!hasMessages && <Welcome onPrompt={setDraft} t={t} />}
          {messages.map((message, index) => <Message key={`${message.role}-${index}`} message={message} index={index} onCopy={copyReply} onRetry={sendMessage} isSending={isSending} t={t} />)}
          <div ref={messageEndRef} />
        </div>
        {notice && <div className="notice" role="alert">{notice}<button onClick={() => setNotice('')} title="Dismiss"><X size={15} /></button></div>}
        {isSlow && <div className="slow-request" role="status">{t('stillWaiting')}</div>}
        <Composer draft={draft} setDraft={setDraft} sendMessage={sendMessage} isSending={isSending} cancelGeneration={cancelGeneration} t={t} />
      </section>

      <aside className={`settings-panel ${settingsOpen ? 'open' : ''}`} aria-label="Chat preferences">
        <div className="settings-title"><div><PanelRight size={18} /><h2>{t('preferences')}</h2></div><button className="icon-button" title={t('hidePreferences')} onClick={() => setSettingsOpen(false)}><ChevronLeft size={18} /></button></div>
        <p className="settings-copy">{t('settingsCopy')}</p>
        <label className="model-field">{t('provider')}<select value={options.provider} onChange={(event) => setOptions((current) => ({ ...current, provider: event.target.value }))}><option value="">{t('serverDefault')}</option><option value="openai_compatible">OpenAI-compatible</option><option value="deepseek">DeepSeek</option><option value="openai">OpenAI</option><option value="openrouter">OpenRouter</option><option value="custom">Custom endpoint</option></select></label>
        <label className="model-field">{t('model')}<input value={options.model} placeholder={t('serverDefault')} onChange={(event) => setOptions((current) => ({ ...current, model: event.target.value }))} /></label>
        <label className="model-field">{t('baseUrl')}<input value={options.baseUrl} placeholder={t('optionalEndpoint')} onChange={(event) => setOptions((current) => ({ ...current, baseUrl: event.target.value }))} /></label>
        <label className="model-field">{t('apiKey')}<input type="password" value={options.apiKey} placeholder={t('tabOnly')} onChange={(event) => setOptions((current) => ({ ...current, apiKey: event.target.value }))} /></label>
        <Toggle label={t('knowledgeRetrieval')} description={t('knowledgeHint')} checked={options.useKnowledge} onChange={(useKnowledge) => setOptions((current) => ({ ...current, useKnowledge }))} />
        <Toggle label={t('styleReference')} description={t('styleHint')} checked={options.useStyle} onChange={(useStyle) => setOptions((current) => ({ ...current, useStyle }))} />
        <label className="range-control"><span>{t('temperature')} <b>{options.temperature.toFixed(1)}</b></span><input type="range" min="0" max="2" step="0.1" value={options.temperature} onChange={(event) => setOptions((current) => ({ ...current, temperature: Number(event.target.value) }))} /></label>
        <div className="contract-note"><span>API v1</span><p>Cookie session and CSRF protection are active.</p></div>
      </aside></> : <section className="feature-main">{activeView === 'memory' && <MemoryPage t={t} />}{activeView === 'mood' && <MoodPage t={t} />}{activeView === 'knowledge' && <KnowledgePage t={t} />}{activeView === 'operations' && <OperationsPage t={t} />}{activeView === 'privacy' && <PrivacyPage t={t} onDeleted={logout} />}</section>}
    </main>
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

function Welcome({ onPrompt, t }) { return <div className="welcome"><div className="welcome-icon"><Sparkles size={24} /></div><h2>{t('welcomeTitle')}</h2><p>{t('welcomeText')}</p><div className="prompt-row"><button onClick={() => onPrompt('我今天有一点焦虑，能陪我理一理吗？')}>{t('talkPrompt')}</button><button onClick={() => onPrompt('我想为未来做一点准备，可以从哪里开始？')}>{t('planPrompt')}</button></div></div> }

function Message({ message, index, onCopy, onRetry, isSending, t }) { return <article className={`message ${message.role} ${message.failed ? 'failed' : ''}`}><div className="message-avatar">{message.role === 'assistant' ? <Sparkles size={15} /> : 'You'}</div><div><div className="message-body">{message.pending && !message.content ? <span className="typing"><i /><i /><i /></span> : message.content}</div>{message.role === 'assistant' && !message.pending && <div className="message-actions">{message.content && !message.failed && <button className="message-action" title={t('copyReply')} onClick={() => onCopy(message.content)}><Copy size={14} /></button>}{message.failed && message.retryable && <button className="message-action" title={t('retryGeneration')} disabled={isSending} onClick={() => onRetry(index)}><RefreshCw size={14} /></button>}</div>}</div></article> }

function Composer({ draft, setDraft, sendMessage, isSending, cancelGeneration, t }) { return <div className="composer"><textarea value={draft} rows="1" placeholder={t('composer')} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage() } }} />{isSending ? <button type="button" className="send-button stop-button" title={t('stopGenerating')} onClick={cancelGeneration}><Square size={15} fill="currentColor" /></button> : <button type="button" className="send-button" title={t('composer')} disabled={!draft.trim()} onClick={() => sendMessage()}><SendHorizontal size={18} /></button>}</div> }

function PreferencesControls({ theme, setTheme, locale, setLocale, t }) { return <div className="preferences-controls"><label><SunMoon size={15} /><span>{t('theme')}</span><select value={theme} onChange={(event) => setTheme(event.target.value)}><option value="light">{t('light')}</option><option value="dark">{t('dark')}</option><option value="system">{t('system')}</option></select></label><label><Languages size={15} /><span>{t('language')}</span><select value={locale} onChange={(event) => setLocale(event.target.value)}><option value="en">{t('english')}</option><option value="zh">{t('chinese')}</option></select></label></div> }

function Toggle({ label, description, checked, onChange }) { return <label className="toggle"><span><b>{label}</b><small>{description}</small></span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><i /></label> }

export default App
