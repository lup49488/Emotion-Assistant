import { expect, test } from '@playwright/test'

const api = process.env.E2E_API_BASE_URL || 'http://127.0.0.1:18000'

test.beforeEach(async ({ page, request }) => {
  await request.post(`${api}/__test/reset`)
  await page.goto('/')
  await page.getByLabel('User ID').fill('e2e-user')
  await page.getByLabel('Access password').fill('e2e-password')
  await page.getByRole('button', { name: 'Sign in / Register' }).click()
  await expect(page.getByPlaceholder('Message Serenova')).toBeVisible()
})

test('IME confirmation keeps the draft until a separate Enter submits it', async ({ page }) => {
  const composer = page.getByPlaceholder('Message Serenova')
  let submissions = 0
  page.on('request', (request) => {
    if (request.url().endsWith('/api/v1/chat/stream')) submissions += 1
  })
  await composer.fill('今天想聊聊')
  await composer.dispatchEvent('keydown', { key: 'Enter', code: 'Enter', isComposing: true })
  await expect(composer).toHaveValue('今天想聊聊')
  // Some IMEs identify the final composing Enter only by legacy keyCode 229.
  await composer.dispatchEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 229 })
  await expect(composer).toHaveValue('今天想聊聊')
  expect(submissions).toBe(0)
  await composer.press('Enter')
  await expect(page.locator('.message.user')).toContainText('今天想聊聊')
  await expect(page.locator('.message.assistant')).toContainText('Grounded answer')
  expect(submissions).toBe(1)
})

test('conversation controls cannot replace the active thread during generation', async ({ page }) => {
  let release
  const gate = new Promise((resolve) => { release = resolve })
  await page.route('**/api/v1/chat/stream', async (route) => {
    await gate
    await route.continue()
  })
  await page.getByRole('button', { name: 'New chat', exact: true }).first().click()
  const conversation = page.locator('.desktop-sidebar .conversation').first()
  await expect(conversation).toBeVisible()
  await page.getByPlaceholder('Message Serenova').fill('Keep this reply in its thread')
  await page.getByTitle('Message Serenova', { exact: true }).click()
  try {
    await expect(page.getByTitle('Stop generating')).toBeVisible()
    await expect(conversation).toBeDisabled()
    await expect(page.getByRole('button', { name: 'New chat', exact: true }).first()).toBeDisabled()
  } finally {
    release()
  }
  await expect(page.locator('.message.assistant')).toContainText('Grounded answer')
  await expect(conversation).toBeEnabled()
})

test('a late conversation response cannot overwrite the latest selection', async ({ page }) => {
  const threads = ['first-thread', 'second-thread'].map((id) => ({
    id, title: id, messages: [{ id: `${id}-message`, role: 'assistant', content: `Content of ${id}` }],
  }))
  await page.route('**/api/v1/conversations', (route) => route.fulfill({ json: { conversations: threads } }))
  let releaseFirst
  const firstGate = new Promise((resolve) => { releaseFirst = resolve })
  await page.route('**/api/v1/conversations/first-thread', async (route) => {
    await firstGate
    await route.fulfill({ json: { conversation: threads[0] } })
  })
  await page.route('**/api/v1/conversations/second-thread', (route) => route.fulfill({ json: { conversation: threads[1] } }))
  await page.reload()
  const firstRequest = page.waitForRequest('**/api/v1/conversations/first-thread')
  await page.locator('.desktop-sidebar .conversation').filter({ hasText: 'first-thread' }).click()
  await firstRequest
  await page.locator('.desktop-sidebar .conversation').filter({ hasText: 'second-thread' }).click()
  await expect(page.locator('.message.assistant')).toContainText('Content of second-thread')
  const firstResponse = page.waitForResponse('**/api/v1/conversations/first-thread')
  releaseFirst()
  await (await firstResponse).finished()
  // Let the browser process the older response and React commit its updates.
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  await expect(page.locator('.message.assistant')).toContainText('Content of second-thread')
})
