import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Bot,
  ChevronLeft,
  CirclePlus,
  LoaderCircle,
  LogOut,
  MessageSquare,
  PanelRight,
  SendHorizontal,
  Settings2,
  Sparkles,
  X,
} from 'lucide-react'
import './App.css'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    credentials: 'include',
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.message || `Request failed (${response.status})`)
  }
  return response
}

function csrfHeaders() {
  const cookieName = 'chatbot_csrf='
  const token = document.cookie.split('; ').find((item) => item.startsWith(cookieName))?.slice(cookieName.length)
  return token ? { 'X-CSRF-Token': decodeURIComponent(token) } : {}
}

function App() {
  const [session, setSession] = useState(null)
  const [conversations, setConversations] = useState([])
  const [activeConversation, setActiveConversation] = useState(null)
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [notice, setNotice] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(() => window.innerWidth > 1060)
  const [options, setOptions] = useState({ useKnowledge: false, useStyle: false, temperature: 0.8 })
  const messageEndRef = useRef(null)

  const hasMessages = messages.length > 0
  const activeTitle = activeConversation?.title || 'New conversation'
  const apiLabel = useMemo(() => API_BASE_URL.replace(/^https?:\/\//, ''), [])

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isSending])

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
        method: 'POST', headers: csrfHeaders(), body: JSON.stringify({ title: 'New conversation' }),
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

  async function sendMessage() {
    const text = draft.trim()
    if (!text || isSending) return
    setDraft('')
    setNotice('')
    setIsSending(true)

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

      setMessages((current) => [...current, { role: 'user', content: text }, { role: 'assistant', content: '', pending: true }])
      const response = await apiFetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: csrfHeaders(),
        body: JSON.stringify({
          message: text,
          conversation_id: conversationId,
          temperature: options.temperature,
          use_knowledge: options.useKnowledge,
          use_style: options.useStyle,
        }),
      })
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let receivedError = ''

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
            setMessages((current) => current.map((item, index) => index === current.length - 1
              ? { ...item, content: item.content + payload.text, pending: false }
              : item))
          }
          if (eventName === 'error') receivedError = payload.message || 'The model could not complete this reply.'
        }
      }
      if (receivedError) throw new Error(receivedError)
      await refreshConversations()
    } catch (error) {
      setNotice(error.message)
      setMessages((current) => current.map((item, index) => index === current.length - 1
        ? { ...item, content: item.content || 'Unable to generate a reply.', pending: false, failed: true }
        : item))
    } finally {
      setIsSending(false)
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

  if (!session) return <LoginScreen onSuccess={restoreSession} />

  return (
    <main className="app-shell">
      <aside className="sidebar" aria-label="Conversation navigation">
        <div className="brand"><Sparkles size={18} aria-hidden="true" /><span>Mindful</span></div>
        <button className="new-chat" onClick={createConversation}><CirclePlus size={18} />New chat</button>
        <div className="conversation-list">
          <p className="section-label">Conversations</p>
          {conversations.length === 0 && <p className="empty-list">No saved chats yet.</p>}
          {conversations.map((conversation) => (
            <button key={conversation.id} className={`conversation ${conversation.id === activeConversation?.id ? 'selected' : ''}`} onClick={() => selectConversation(conversation)}>
              <MessageSquare size={15} /><span>{conversation.title}</span>
            </button>
          ))}
        </div>
        <div className="sidebar-footer">
          <div className="identity"><span className="avatar">{session.user_id.slice(0, 1).toUpperCase()}</span><span>{session.user_id}</span></div>
          <button className="icon-button" title="Log out" onClick={logout}><LogOut size={17} /></button>
        </div>
      </aside>

      <section className="chat-panel">
        <header className="chat-header"><div><h1>{activeTitle}</h1><p>Connected to {apiLabel}</p></div><button className="icon-button mobile-settings" title="Toggle settings" onClick={() => setSettingsOpen((open) => !open)}><Settings2 size={18} /></button></header>
        <div className="messages" aria-live="polite">
          {!hasMessages && <Welcome onPrompt={setDraft} />}
          {messages.map((message, index) => <Message key={`${message.role}-${index}`} message={message} />)}
          <div ref={messageEndRef} />
        </div>
        {notice && <div className="notice" role="alert">{notice}<button onClick={() => setNotice('')} title="Dismiss"><X size={15} /></button></div>}
        <Composer draft={draft} setDraft={setDraft} sendMessage={sendMessage} isSending={isSending} />
      </section>

      <aside className={`settings-panel ${settingsOpen ? 'open' : ''}`} aria-label="Chat preferences">
        <div className="settings-title"><div><PanelRight size={18} /><h2>Preferences</h2></div><button className="icon-button" title="Hide preferences" onClick={() => setSettingsOpen(false)}><ChevronLeft size={18} /></button></div>
        <p className="settings-copy">These options are sent with each reply.</p>
        <Toggle label="Knowledge retrieval" description="Include matching RAG references" checked={options.useKnowledge} onChange={(useKnowledge) => setOptions((current) => ({ ...current, useKnowledge }))} />
        <Toggle label="Style reference" description="Apply the selected style corpus" checked={options.useStyle} onChange={(useStyle) => setOptions((current) => ({ ...current, useStyle }))} />
        <label className="range-control"><span>Temperature <b>{options.temperature.toFixed(1)}</b></span><input type="range" min="0" max="2" step="0.1" value={options.temperature} onChange={(event) => setOptions((current) => ({ ...current, temperature: Number(event.target.value) }))} /></label>
        <div className="contract-note"><span>API v1</span><p>Cookie session and CSRF protection are active.</p></div>
      </aside>
    </main>
  )
}

function LoginScreen({ onSuccess }) {
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

  return <main className="login-page"><section className="login-intro"><div className="brand"><Sparkles size={20} /><span>Mindful</span></div><div><h1>A quieter place to think.</h1><p>Continue your conversations with memory, knowledge, and personal context kept under your control.</p></div><div className="intro-mark"><Bot size={38} /></div></section><form className="login-form" onSubmit={submit}><h2>Welcome back</h2><p>Sign in to open your private workspace.</p><label>User ID<input value={userId} onChange={(event) => setUserId(event.target.value)} autoComplete="username" required /></label><label>Access password<input type="password" value={accessKey} onChange={(event) => setAccessKey(event.target.value)} autoComplete="current-password" required /></label>{error && <div className="login-error">{error}</div>}<button className="login-button" disabled={loading}>{loading ? <LoaderCircle className="spin" size={18} /> : 'Sign in'}</button></form></main>
}

function Welcome({ onPrompt }) { return <div className="welcome"><div className="welcome-icon"><Sparkles size={24} /></div><h2>How are you feeling today?</h2><p>You can start with a question, a reflection, or a small thing that is on your mind.</p><div className="prompt-row"><button onClick={() => onPrompt('我今天有一点焦虑，能陪我理一理吗？')}>Talk through a feeling</button><button onClick={() => onPrompt('我想为未来做一点准备，可以从哪里开始？')}>Plan the next step</button></div></div> }

function Message({ message }) { return <article className={`message ${message.role} ${message.failed ? 'failed' : ''}`}><div className="message-avatar">{message.role === 'assistant' ? <Sparkles size={15} /> : 'You'}</div><div className="message-body">{message.pending && !message.content ? <span className="typing"><i /><i /><i /></span> : message.content}</div></article> }

function Composer({ draft, setDraft, sendMessage, isSending }) { return <div className="composer"><textarea value={draft} rows="1" placeholder="Message Mindful" onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage() } }} /><button className="send-button" title="Send message" disabled={!draft.trim() || isSending} onClick={sendMessage}>{isSending ? <LoaderCircle className="spin" size={18} /> : <SendHorizontal size={18} />}</button></div> }

function Toggle({ label, description, checked, onChange }) { return <label className="toggle"><span><b>{label}</b><small>{description}</small></span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><i /></label> }

export default App
