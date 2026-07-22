// Local Vite development uses FastAPI on port 8000. Public deployments use
// same-origin /api routing so signed and CSRF cookies remain first-party.
const isLocalHost = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname)
const DEFAULT_API_BASE_URL = isLocalHost ? `${window.location.protocol}//${window.location.hostname}:8000` : ''
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, '')

export class ApiRequestError extends Error {
  constructor(message, { code = 'request_failed', retryable = false, status = 0 } = {}) {
    super(message)
    this.name = 'ApiRequestError'
    this.code = code
    this.retryable = retryable
    this.status = status
  }
}

export async function apiFetch(path, options = {}) {
  const isFormData = options.body instanceof FormData
  const response = await fetch(`${API_BASE_URL}${path}`, {
    credentials: 'include',
    ...options,
    headers: { ...(isFormData ? {} : { 'Content-Type': 'application/json' }), ...(options.headers || {}) },
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new ApiRequestError(error.message || error.detail || `Request failed (${response.status})`, { ...error, status: response.status })
  }
  return response
}

export function csrfHeaders() {
  const cookieName = `${import.meta.env.VITE_CSRF_COOKIE_NAME || 'chatbot_csrf'}=`
  const token = document.cookie.split('; ').find((item) => item.startsWith(cookieName))?.slice(cookieName.length)
  return token ? { 'X-CSRF-Token': decodeURIComponent(token) } : {}
}

export async function readJson(path, options) {
  return (await apiFetch(path, options)).json()
}
