import { spawn } from 'node:child_process'

const isWindows = process.platform === 'win32'
const npx = isWindows ? 'npx.cmd' : 'npx'
const children = []
const mockPort = 18100
const vitePort = 4180
const baseUrl = `http://127.0.0.1:${vitePort}`
const mockUrl = `http://127.0.0.1:${mockPort}`

function start(command, args, options = {}) {
  const child = spawn(command, args, { stdio: 'inherit', ...options })
  children.push(child)
  return child
}

async function waitFor(url, timeoutMs = 12_000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url)
      if (response.ok) return
    } catch {
      // The process may still be binding its loopback port.
    }
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  throw new Error(`E2E service did not become ready: ${url}`)
}

function waitForExit(child) {
  return new Promise((resolve, reject) => {
    child.once('error', reject)
    child.once('exit', (code) => resolve(code ?? 1))
  })
}

async function stop(child) {
  if (child.exitCode !== null || child.killed) return
  child.kill()
  await Promise.race([
    waitForExit(child),
    new Promise((resolve) => setTimeout(resolve, 3_000)),
  ])
}

start(process.execPath, ['e2e/mock-api.mjs'], { env: { ...process.env, E2E_MOCK_PORT: String(mockPort), E2E_FRONTEND_ORIGIN: baseUrl } })
start(process.execPath, [
  './node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', String(vitePort),
], { env: { ...process.env, VITE_API_BASE_URL: mockUrl } })

try {
  await waitFor(`${mockUrl}/health`)
  await waitFor(baseUrl)
  const playwright = start(npx, ['playwright', 'test'], { shell: isWindows, env: { ...process.env, E2E_BASE_URL: baseUrl, E2E_API_BASE_URL: mockUrl } })
  const exitCode = await waitForExit(playwright)
  process.exitCode = exitCode
} catch (error) {
  console.error(error)
  process.exitCode = 1
} finally {
  await Promise.all(children.slice(0, 2).map(stop))
}
