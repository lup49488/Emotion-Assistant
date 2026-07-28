import http from 'node:http'
import { randomUUID } from 'node:crypto'

let loggedIn = false
const conversations = []
const memorySnapshot = {
  history: [],
  emotion_memory: [],
  long_memory: [],
  stable_profile: [],
  interest_memory: [],
  memory_events: [],
}

function resetState() {
  loggedIn = false
  conversations.splice(0, conversations.length)
  Object.assign(memorySnapshot, {
    history: [], emotion_memory: [], long_memory: [], stable_profile: [], interest_memory: [], memory_events: [],
  })
}

function send(response, status, body, headers = {}) {
  response.writeHead(status, { 'Content-Type': 'application/json', ...headers })
  response.end(JSON.stringify(body))
}

function conversation(id) {
  return conversations.find((item) => item.id === id)
}

const server = http.createServer(async (request, response) => {
  const origin = request.headers.origin
  if (origin === 'http://127.0.0.1:4174') {
    response.setHeader('Access-Control-Allow-Origin', origin)
    response.setHeader('Access-Control-Allow-Credentials', 'true')
    response.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-CSRF-Token')
    response.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
  }
  if (request.method === 'OPTIONS') return response.writeHead(204).end()

  const chunks = []
  for await (const chunk of request) chunks.push(chunk)
  const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {}
  const url = new URL(request.url, 'http://127.0.0.1:18000')

  if (request.method === 'POST' && url.pathname === '/__test/reset') {
    resetState()
    return send(response, 200, { status: 'reset' })
  }
  if (request.method === 'GET' && url.pathname === '/health') return send(response, 200, { status: 'ok' })
  if (request.method === 'POST' && url.pathname === '/api/v1/auth/login') {
    loggedIn = true
    return send(response, 200, { token_type: 'cookie', expires_in: 3600, user_id: 'e2e-user' }, {
      'Set-Cookie': ['chatbot_session=e2e-session; Path=/; HttpOnly; SameSite=Lax', 'chatbot_csrf=e2e-csrf; Path=/; SameSite=Lax'],
    })
  }
  if (request.method === 'GET' && url.pathname === '/api/v1/auth/session') return loggedIn
    ? send(response, 200, { user_id: 'e2e-user', authentication: 'signed_cookie', can_access_operations: false, can_manage_knowledge: false })
    : send(response, 401, { detail: 'Signed session cookie is required.' })
  if (request.method === 'GET' && url.pathname === '/api/v1/conversations') return send(response, 200, { conversations: conversations.map((item) => ({ id: item.id, title: item.title, created_at: item.created_at, updated_at: item.updated_at, message_count: item.message_count })) })
  if (request.method === 'POST' && url.pathname === '/api/v1/conversations') {
    const item = { id: randomUUID().replaceAll('-', ''), title: body.title || 'New conversation', created_at: '2026-07-23T00:00:00', updated_at: '2026-07-23T00:00:00', message_count: 0, messages: [] }
    conversations.push(item)
    return send(response, 200, { conversation: item })
  }
  if (request.method === 'PUT' && /^\/api\/v1\/conversations\/[^/]+$/.test(url.pathname)) {
    const item = conversation(url.pathname.split('/').at(-1))
    if (!item) return send(response, 404, { detail: 'Conversation was not found.' })
    item.title = body.title
    return send(response, 200, { conversation: item })
  }
  if (request.method === 'DELETE' && /^\/api\/v1\/conversations\/[^/]+$/.test(url.pathname)) {
    const index = conversations.findIndex((item) => item.id === url.pathname.split('/').at(-1))
    if (index === -1) return send(response, 404, { detail: 'Conversation was not found.' })
    conversations.splice(index, 1)
    response.writeHead(204)
    return response.end()
  }
  if (request.method === 'GET' && url.pathname === '/api/v1/memory') return send(response, 200, memorySnapshot)
  if (request.method === 'GET' && url.pathname === '/api/v1/memory/quality') return send(response, 200, { report: 'Memory quality is healthy.' })
  if (request.method === 'POST' && url.pathname === '/api/v1/chat/stream') {
    const item = conversation(body.conversation_id)
    if (body.message === 'Trigger retryable error' && !body.retry_last_response) {
      response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
      response.write('event: error\ndata: {"code":"provider_timeout","retryable":true}\n\n')
      return response.end('event: done\ndata: {}\n\n')
    }
    const reply = body.retry_last_response ? 'Regenerated answer' : body.message === 'Show markdown'
      ? 'First line<br>Second line\n\n\\[ P(C \\mid \\mathbf{x}) = \\frac{P(\\mathbf{x} \\mid C)P(C)}{P(\\mathbf{x})} \\]\n\n| 后验分布 | 公式 |\n| --- | --- |\n| Posterior | $$P(\\theta | x)=\\frac{P(x | \\theta)P(\\theta)}{P(x)}$$ |'
      : 'Grounded answer from the knowledge base.'
    item.messages = [{ role: 'user', content: body.message }, { role: 'assistant', content: reply }]
    item.message_count = 2
    memorySnapshot.history = [...item.messages]
    if (body.message === 'I feel happy today') {
      memorySnapshot.emotion_memory = [{ label: 'joy', score: 0.98, time: '2026-07-27T09:00:00' }]
    }
    response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
    response.write(`event: chunk\ndata: ${JSON.stringify({ text: reply.slice(0, 10) })}\n\n`)
    response.write(`event: chunk\ndata: ${JSON.stringify({ text: reply.slice(10) })}\n\n`)
    if (body.use_knowledge) response.write(`event: citations\ndata: ${JSON.stringify({ trace_id: 'e2e-trace', citations: [{ source: 'deployment_guide.md', chunk_index: 0, score: 0.93, excerpt: 'Deployment guide excerpt' }] })}\n\n`)
    return response.end('event: done\ndata: {}\n\n')
  }
  if (request.method === 'POST' && url.pathname === '/api/v1/rag/feedback') return send(response, 200, { trace_id: body.trace_id, helpful: Boolean(body.helpful), comment: body.comment || '', created_at: '2026-07-23T00:00:00' })
  return send(response, 404, { detail: 'Not found' })
})

server.listen(18000, '127.0.0.1')
