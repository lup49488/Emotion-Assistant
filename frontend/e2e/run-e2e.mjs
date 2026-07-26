import { spawn } from 'node:child_process'

const isWindows = process.platform === 'win32'
const npx = isWindows ? 'npx.cmd' : 'npx'
const children = []

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

start(process.execPath, ['e2e/mock-api.mjs'])
start(process.execPath, [
  './node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '4174',
], { env: { ...process.env, VITE_API_BASE_URL: 'http://127.0.0.1:18000' } })

try {
  await waitFor('http://127.0.0.1:18000/health')
  await waitFor('http://127.0.0.1:4174')
  const playwright = start(npx, ['playwright', 'test'], { shell: isWindows })
  const exitCode = await waitForExit(playwright)
  process.exitCode = exitCode
} catch (error) {
  console.error(error)
  process.exitCode = 1
} finally {
  await Promise.all(children.slice(0, 2).map(stop))
}
