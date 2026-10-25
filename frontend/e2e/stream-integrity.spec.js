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

test('a partial stream without done keeps its text and offers retry', async ({ page }) => {
  const errors = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.route('**/api/v1/chat/stream', (route) => route.fulfill({
    contentType: 'text/event-stream',
    body: 'event: chunk\ndata: {"text":"Partial reply preserved"}\n\n',
  }))
  await page.getByPlaceholder('Message Serenova').fill('Test an interrupted stream')
  await page.getByRole('button', { name: 'Send message' }).click()
  await expect(page.getByRole('alert')).toContainText('connection ended before the reply was complete')
  const reply = page.locator('.message.assistant.failed')
  await expect(reply).toContainText('Partial reply preserved')
  await expect(reply.getByTitle('Regenerate')).toBeEnabled()
  await expect(page.getByTitle('Stop generating')).toHaveCount(0)
  expect(errors).toEqual([])
})

test('a failed regeneration restores the saved reply and its quote action', async ({ page }) => {
  await page.getByPlaceholder('Message Serenova').fill('Save an original reply')
  await page.getByRole('button', { name: 'Send message' }).click()
  const reply = page.locator('.message.assistant')
  await expect(reply.getByTitle('Reply to this message')).toBeVisible()
  const original = await reply.locator('.message-body').innerText()
  await page.route('**/api/v1/chat/stream', (route) => route.fulfill({
    contentType: 'text/event-stream',
    body: 'event: chunk\ndata: {"text":"Unfinished replacement"}\n\n',
  }))
  await reply.getByTitle('Regenerate').click()
  await expect(page.getByRole('alert')).toContainText('connection ended before the reply was complete')
  await expect(reply.locator('.message-body')).toHaveText(original)
  await expect(reply.getByTitle('Reply to this message')).toBeVisible()
  await expect(page.locator('.message.assistant.failed')).toHaveCount(0)
})

test('a sidebar refresh failure does not roll back a completed regeneration', async ({ page }) => {
  await page.getByPlaceholder('Message Serenova').fill('Save an original reply')
  await page.getByRole('button', { name: 'Send message' }).click()
  const reply = page.locator('.message.assistant')
  await expect(reply.getByTitle('Reply to this message')).toBeVisible()
  await page.route('**/api/v1/conversations', (route) => route.request().method() === 'GET'
    ? route.fulfill({ status: 503, json: { detail: 'Sidebar unavailable' } })
    : route.fallback())
  await page.route('**/api/v1/chat/stream', (route) => route.fulfill({
    contentType: 'text/event-stream',
    body: 'event: chunk\ndata: {"text":"Completed replacement"}\n\nevent: done\ndata: {}\n\n',
  }))
  await reply.getByTitle('Regenerate').click()
  await expect(reply).toContainText('Completed replacement')
  await expect(page.getByTitle('Stop generating')).toHaveCount(0)
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('an archive failure keeps the generated reply and offers an idempotent save retry', async ({ page }) => {
  await page.route('**/api/v1/chat/stream', (route) => route.fulfill({
    contentType: 'text/event-stream',
    body: 'event: chunk\ndata: {"text":"Generated but not yet saved"}\n\nevent: error\ndata: {"code":"conversation_save_failed","retryable":true}\n\n',
  }))
  await page.getByPlaceholder('Message Serenova').fill('Keep this generated reply')
  await page.getByRole('button', { name: 'Send message' }).click()
  const reply = page.locator('.message.assistant')
  await expect(reply).toContainText('Generated but not yet saved')
  await expect(reply).toContainText('This reply is not saved yet.')
  await expect(reply.getByTitle('Regenerate')).toHaveCount(0)
  await reply.getByRole('button', { name: 'Retry save' }).click()
  await expect(reply).not.toContainText('not saved yet')
  await expect(reply.getByTitle('Reply to this message')).toBeVisible()
  await expect(reply.getByTitle('Regenerate')).toBeVisible()
})

test('an error after archived does not restore the superseded reply', async ({ page }) => {
  await page.getByPlaceholder('Message Serenova').fill('Save an original reply')
  await page.getByRole('button', { name: 'Send message' }).click()
  const reply = page.locator('.message.assistant')
  await expect(reply.getByTitle('Reply to this message')).toBeVisible()
  await page.route('**/api/v1/chat/stream', (route) => route.fulfill({
    contentType: 'text/event-stream',
    body: [
      'event: chunk\ndata: {"text":"Replacement is saved"}',
      'event: archived\ndata: {"user_message_id":"new-user-id","assistant_message_id":"new-assistant-id"}',
      'event: error\ndata: {"code":"generation_failed","retryable":true}',
    ].join('\n\n') + '\n\n',
  }))
  await reply.getByTitle('Regenerate').click()
  await expect(reply).toContainText('Replacement is saved')
  await expect(reply).not.toContainText('Regenerated answer')
  await expect(page.getByRole('alert')).toContainText('The reply is saved, but some follow-up details could not be loaded.')
})
