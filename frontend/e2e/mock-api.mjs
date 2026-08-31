import http from 'node:http'
import { randomUUID } from 'node:crypto'

let loggedIn = false
const conversations = []
const moodRecords = []
const testImage = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL5JwAAAABJRU5ErkJggg==', 'base64')
let memorySaveMode = 'confirm'
let stylePrefix = ''
const mockPort = Number(process.env.E2E_MOCK_PORT || 18000)
const frontendOrigin = process.env.E2E_FRONTEND_ORIGIN || 'http://127.0.0.1:4174'
const memorySnapshot = {
  history: [],
  emotion_memory: [],
  long_memory: [],
  stable_profile: [],
  interest_memory: [],
  memory_events: [],
  pending_memory: [],
}

function resetState() {
  loggedIn = false
  memorySaveMode = 'confirm'
  stylePrefix = ''
  conversations.splice(0, conversations.length)
  moodRecords.splice(0, moodRecords.length, {
    date: '2026-07-29', mood: 'calm', intensity: 3,
    note: '    Indented first line.\n\nSecond paragraph.', source: 'checkin',
    created_at: '2026-07-29T09:00:00', updated_at: '2026-07-29T09:00:00', images: [],
  })
  Object.assign(memorySnapshot, {
    history: [], emotion_memory: [], long_memory: [], stable_profile: [], interest_memory: [], memory_events: [], pending_memory: [],
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
  if (origin === frontendOrigin) {
    response.setHeader('Access-Control-Allow-Origin', origin)
    response.setHeader('Access-Control-Allow-Credentials', 'true')
    response.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-CSRF-Token')
    response.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
  }
  if (request.method === 'OPTIONS') return response.writeHead(204).end()

  const chunks = []
  for await (const chunk of request) chunks.push(chunk)
  const contentType = request.headers['content-type'] || ''
  const body = chunks.length && contentType.includes('application/json') ? JSON.parse(Buffer.concat(chunks).toString()) : {}
  const url = new URL(request.url, `http://127.0.0.1:${mockPort}`)

  if (request.method === 'POST' && url.pathname === '/__test/reset') {
    resetState()
    return send(response, 200, { status: 'reset' })
  }
  if (request.method === 'GET' && url.pathname === '/health') return send(response, 200, { status: 'ok' })
  if (request.method === 'GET' && url.pathname === '/api/v1/model/providers') return send(response, 200, {
    providers: [
      { id: 'nvidia_nim', label: 'NVIDIA NIM', kind: 'openai_compatible', models: ['openai/gpt-oss-20b', 'meta/llama-3.1-8b-instruct'], default_model: 'openai/gpt-oss-20b', default_base_url: 'https://integrate.api.nvidia.com/v1', api_key_envs: ['NVIDIA_NIM_API_KEY', 'LLM_API_KEY'] },
      { id: 'deepseek', label: 'DeepSeek', kind: 'openai_compatible', models: ['deepseek-chat', 'deepseek-reasoner'], default_model: 'deepseek-chat', default_base_url: 'https://api.deepseek.com', api_key_envs: ['DEEPSEEK_API_KEY', 'LLM_API_KEY'] },
      { id: 'custom', label: 'Custom endpoint', kind: 'openai_compatible', models: ['custom-model'], default_model: 'custom-model', default_base_url: '', api_key_envs: ['LLM_API_KEY'] },
    ],
  })
  if (request.method === 'GET' && url.pathname === '/api/v1/style/preference') return send(response, 200, { style_prefix: stylePrefix, available: ['温柔型', '专业型'] })
  if (request.method === 'PUT' && url.pathname === '/api/v1/style/preference') {
    stylePrefix = ['温柔型', '专业型'].includes(body.style_prefix) ? body.style_prefix : ''
    return send(response, 200, { style_prefix: stylePrefix, available: ['温柔型', '专业型'] })
  }
  if (request.method === 'POST' && url.pathname === '/api/v1/auth/login') {
    loggedIn = true
    return send(response, 200, { token_type: 'cookie', expires_in: 3600, user_id: 'e2e-user' }, {
      'Set-Cookie': ['chatbot_session=e2e-session; Path=/; HttpOnly; SameSite=Lax', 'chatbot_csrf=e2e-csrf; Path=/; SameSite=Lax'],
    })
  }
  if (request.method === 'GET' && url.pathname === '/api/v1/auth/session') return loggedIn
    ? send(response, 200, { user_id: 'e2e-user', authentication: 'signed_cookie', can_access_operations: false, can_manage_knowledge: false })
    : send(response, 401, { detail: 'Signed session cookie is required.' })
  if (request.method === 'GET' && url.pathname === '/api/v1/rag/status') return send(response, 200, {
    status: 'ready', documents: [], release: { enabled: true, state: 'passing' },
  })
  if (request.method === 'GET' && url.pathname === '/api/v1/rag/quality') return send(response, 200, {
    level: 'ready', documents: 12, chunks: 486, average_chunk_chars: 534, issues: [],
  })
  if (request.method === 'POST' && url.pathname === '/api/v1/rag/search') return send(response, 200, {
    query: body.query, results: [{ source: 'sleep_hygiene_cn.md', chunk_index: 11, score: 0.71, text: '固定起床时间和固定入睡时间，能帮助形成更稳定的睡眠节律。' }],
  })
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
  if (request.method === 'GET' && url.pathname === '/api/v1/memory/quality') return send(response, 200, { report: url.searchParams.get('locale') === 'en' ? 'Memory quality is healthy.' : '记忆质量良好。' })
  if (request.method === 'GET' && url.pathname === '/api/v1/memory/preference') return send(response, 200, { mode: memorySaveMode })
  if (request.method === 'PUT' && url.pathname === '/api/v1/memory/preference') {
    memorySaveMode = ['auto', 'confirm', 'off'].includes(body.mode) ? body.mode : 'confirm'
    return send(response, 200, { mode: memorySaveMode })
  }
  if (request.method === 'PUT' && url.pathname === '/api/v1/memory/interests') {
    memorySnapshot.interest_memory = Array.isArray(body.items) ? body.items : []
    return send(response, 200, memorySnapshot)
  }
  if (request.method === 'PUT' && url.pathname === '/api/v1/memory/long-term') {
    memorySnapshot.long_memory = Array.isArray(body.items) ? body.items : []
    return send(response, 200, memorySnapshot)
  }
  if (request.method === 'POST' && url.pathname === '/api/v1/import/external/preview') return send(response, 200, {
    source: 'chatgpt', conversations: 2, messages: 8, sample_titles: ['A useful imported conversation'], profile_fields: [{ key: 'name', label: 'Name', value: 'Example user' }],
  })
  if (request.method === 'POST' && url.pathname === '/api/v1/import/external') return send(response, 200, { mode: url.searchParams.get('mode') || 'merge', source: 'chatgpt', conversations: 2, memories: 1, mood_checkins: 0 })
  if (request.method === 'GET' && url.pathname === '/api/v1/privacy') return send(response, 200, { backend: 'json', conversation_count: conversations.length, message_count: 0, history_count: memorySnapshot.history.length, memory_count: memorySnapshot.long_memory.length, mood_count: moodRecords.length, api_request_count: 0 })
  if (request.method === 'GET' && url.pathname === '/api/v1/export') return send(response, 200, { schema_version: 4, exported_at: '2026-08-04T00:00:00', user_id: 'e2e-user', notes: [], ...memorySnapshot, conversations, mood_checkins: moodRecords })
  if (request.method === 'POST' && url.pathname === '/api/v1/import') return send(response, 200, { mode: url.searchParams.get('mode') || 'merge', conversations: 1, mood_checkins: 1, memories: 1 })
  if (request.method === 'GET' && url.pathname === '/api/v1/mood/checkins') return send(response, 200, { records: moodRecords })
  if (request.method === 'POST' && url.pathname === '/api/v1/mood/checkins') {
    const now = '2026-08-22T09:00:00'
    const item = {
      date: body.checkin_date || '2026-08-22', mood: body.mood, intensity: body.intensity,
      note: body.note || '', source: 'checkin', created_at: now, updated_at: now, images: [],
    }
    const index = moodRecords.findIndex((record) => record.date === item.date)
    if (index === -1) moodRecords.push(item)
    else {
      item.images = moodRecords[index].images || []
      moodRecords[index] = item
    }
    return send(response, 200, { record: item })
  }
  const moodImageMatch = url.pathname.match(/^\/api\/v1\/mood\/checkins\/([^/]+)\/images\/([^/]+)$/)
  if (request.method === 'GET' && moodImageMatch) {
    response.writeHead(200, { 'Content-Type': 'image/png', 'Cache-Control': 'private, no-store' })
    return response.end(testImage)
  }
  if (request.method === 'POST' && /^\/api\/v1\/mood\/checkins\/[^/]+\/images$/.test(url.pathname)) {
    const date = url.pathname.split('/')[5]
    const record = moodRecords.find((item) => item.date === date)
    if (!record) return send(response, 404, { detail: 'Mood Check-in was not found.' })
    const image = { id: randomUUID().replaceAll('-', ''), filename: 'mood-photo.png', content_type: 'image/png', size_bytes: testImage.length }
    record.images ||= []; record.images.push(image)
    return send(response, 201, image)
  }
  if (request.method === 'DELETE' && moodImageMatch) {
    const [, date, imageId] = moodImageMatch
    const record = moodRecords.find((item) => item.date === date)
    if (!record || !record.images?.some((image) => image.id === imageId)) return send(response, 404, { detail: 'Mood Check-in image was not found.' })
    record.images = record.images.filter((image) => image.id !== imageId)
    response.writeHead(204)
    return response.end()
  }
  if (request.method === 'GET' && url.pathname === '/api/v1/mood/weekly') return send(response, 200, url.searchParams.get('locale') === 'en'
    ? { points: [], summary: 'No weekly summary available.', analysis: '' }
    : { points: [], summary: '暂无一周心情总结。', analysis: '' })
  if (request.method === 'POST' && url.pathname === '/api/v1/chat/stream') {
    const item = conversation(body.conversation_id)
    if (body.message === 'Trigger retryable error' && !body.retry_last_response) {
      response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
      response.write('event: error\ndata: {"code":"provider_timeout","retryable":true}\n\n')
      return response.end('event: done\ndata: {}\n\n')
    }
    if (body.message === 'Ask without sources' && body.use_knowledge) {
      response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
      response.write('event: rag_status\ndata: {"status":"insufficient","code":"insufficient_evidence","reason":"no_relevant_sources","enforced":true}\n\n')
      response.write('event: chunk\ndata: {"text":"The knowledge base did not contain enough relevant information."}\n\n')
      return response.end('event: done\ndata: {}\n\n')
    }
    // Retrieval found nothing but RAG_REQUIRE_EVIDENCE is off, so the reply still stands.
    if (body.message === 'Ask without sources unenforced' && body.use_knowledge) {
      response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
      response.write('event: rag_status\ndata: {"status":"insufficient","code":"insufficient_evidence","reason":"no_relevant_sources","enforced":false}\n\n')
      response.write('event: chunk\ndata: {"text":"General answer without sources."}\n\n')
      return response.end('event: done\ndata: {}\n\n')
    }
    const reply = body.retry_last_response ? 'Regenerated answer' : body.quoted_message_id
      ? 'Quoted reply with the selected context.' : body.mood_checkin
      ? `Mood reflection: ${body.mood_checkin.mood} (${body.mood_checkin.intensity}/5) - ${body.mood_checkin.note}`
      : body.message === 'Show markdown'
      ? 'First line<br>Second line\n\n\\[ P(C \\mid \\mathbf{x}) = \\frac{P(\\mathbf{x} \\mid C)P(C)}{P(\\mathbf{x})} \\]\n\n| 后验分布 | 公式 |\n| --- | --- |\n| Posterior | $$P(\\theta | x)=\\frac{P(x | \\theta)P(\\theta)}{P(x)}$$ |'
      : 'Grounded answer from the knowledge base.'
    const userMessage = { id: randomUUID().replaceAll('-', ''), role: 'user', content: body.message, reply_to_message_id: body.quoted_message_id || null }
    const assistantMessage = { id: randomUUID().replaceAll('-', ''), role: 'assistant', content: reply }
    item.messages = [...item.messages, userMessage, assistantMessage]
    item.message_count = item.messages.length
    memorySnapshot.history = [...item.messages]
    if (body.message === 'I feel happy today') {
      memorySnapshot.emotion_memory = [{ label: 'joy', score: 0.98, time: '2026-07-27T09:00:00' }]
    }
    response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
    response.write(`event: chunk\ndata: ${JSON.stringify({ text: reply.slice(0, 10) })}\n\n`)
    response.write(`event: chunk\ndata: ${JSON.stringify({ text: reply.slice(10) })}\n\n`)
    response.write(`event: archived\ndata: ${JSON.stringify({ user_message_id: userMessage.id, assistant_message_id: assistantMessage.id })}\n\n`)
    if (body.use_knowledge) response.write(`event: citations\ndata: ${JSON.stringify({ trace_id: 'e2e-trace', citations: [{ source: 'deployment_guide.md', chunk_index: 0, score: 0.93, excerpt: 'Deployment guide excerpt' }] })}\n\n`)
    return response.end('event: done\ndata: {}\n\n')
  }
  if (request.method === 'POST' && url.pathname === '/api/v1/rag/feedback') return send(response, 200, { trace_id: body.trace_id, helpful: Boolean(body.helpful), comment: body.comment || '', created_at: '2026-07-23T00:00:00' })
  return send(response, 404, { detail: 'Not found' })
})

server.listen(mockPort, '127.0.0.1')
