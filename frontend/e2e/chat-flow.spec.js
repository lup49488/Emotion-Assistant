import { expect, test } from '@playwright/test'

async function signIn(page) {
  await page.goto('/')
  await page.getByLabel('User ID').fill('e2e-user')
  await page.getByLabel('Access password').fill('e2e-password')
  await page.getByRole('button', { name: 'Sign in / Register' }).click()
  await expect(page.getByPlaceholder('Message Serenova')).toBeVisible()
}

async function sendMessage(page, text) {
  await page.getByPlaceholder('Message Serenova').fill(text)
  await page.getByTitle('Message Serenova').click()
}

test.beforeEach(async ({ request }) => {
  await request.post('http://127.0.0.1:18000/__test/reset')
})

test('login, cited reply, feedback, and regeneration work together', async ({ page }) => {
  await signIn(page)

  await page.getByText('Knowledge retrieval').click()
  await sendMessage(page, 'How do I deploy this project?')
  await expect(page.getByText('Grounded answer from the knowledge base.')).toBeVisible()
  await expect(page.getByText('Sources')).toBeVisible()
  await expect(page.getByText('deployment_guide.md')).toBeVisible()

  await page.getByTitle('Helpful sources', { exact: true }).click()
  await expect(page.getByTitle('Helpful sources', { exact: true })).toHaveClass(/selected/)
  await page.getByTitle('Regenerate').click()
  await expect(page.getByText('Regenerated answer')).toBeVisible()
})

test('assistant replies render HTML line breaks and LaTeX delimiters safely', async ({ page }) => {
  await signIn(page)
  await sendMessage(page, 'Show markdown')

  await expect(page.getByText('First line')).toBeVisible()
  await expect(page.getByText('Second line')).toBeVisible()
  await expect(page.locator('.katex-display')).toHaveCount(1)
  await expect(page.getByText('Posterior')).toBeVisible()
  await expect(page.locator('.katex')).toHaveCount(2)
  await expect(page.getByText('$$', { exact: false })).toHaveCount(0)
  await expect(page.getByText('<br>', { exact: true })).toHaveCount(0)
})

test('conversation titles can be renamed and deleted', async ({ page }) => {
  await signIn(page)
  await sendMessage(page, 'A temporary conversation')
  await expect(page.getByText('Grounded answer from the knowledge base.')).toBeVisible()

  await page.getByTitle('Rename conversation').click()
  await page.getByLabel('Conversation title').fill('Renamed conversation')
  await page.getByTitle('Save title').click()
  await expect(page.getByRole('button', { name: 'Renamed conversation' })).toBeVisible()

  page.once('dialog', (dialog) => dialog.accept())
  await page.getByTitle('Delete conversation').click()
  await expect(page.getByRole('button', { name: 'Renamed conversation' })).toHaveCount(0)
})

test('retryable SSE errors display a retry action that regenerates the reply', async ({ page }) => {
  await signIn(page)
  await sendMessage(page, 'Trigger retryable error')

  await expect(page.getByRole('alert')).toHaveText(/The provider timed out\. Please retry\./)
  await page.getByTitle('Regenerate').click()
  await expect(page.getByText('Regenerated answer')).toBeVisible()
})

test('insufficient RAG evidence displays a grounded-response status without regeneration', async ({ page }) => {
  await signIn(page)
  await page.getByText('Knowledge retrieval').click()
  await sendMessage(page, 'Ask without sources')

  await expect(page.getByRole('status')).toHaveText(/does not contain enough relevant information/)
  await expect(page.getByTitle('Regenerate')).toHaveCount(0)
})

test('an unenforced insufficient-evidence status still shows the generated reply', async ({ page }) => {
  await signIn(page)
  await page.getByText('Knowledge retrieval').click()
  await sendMessage(page, 'Ask without sources unenforced')

  await expect(page.getByText('General answer without sources.')).toBeVisible()
  await expect(page.getByRole('status')).toHaveCount(0)
})

test('a reliable conversation emotion is visible in personal data', async ({ page }) => {
  await signIn(page)
  await sendMessage(page, 'I feel happy today')
  await expect(page.getByText('Grounded answer from the knowledge base.')).toBeVisible()

  await page.getByRole('button', { name: 'Personal data' }).click()
  await expect(page.getByText('Conversation emotion history')).toBeVisible()
  await expect(page.getByText('joy', { exact: true })).toBeVisible()
  await expect(page.getByText('0.98', { exact: false })).toBeVisible()
})

test('mood notes retain leading indentation and paragraph breaks in the check-in history', async ({ page }) => {
  await signIn(page)
  await page.getByRole('button', { name: 'Mood check-in' }).click()

  const note = page.locator('.record-note')
  // toHaveText trims, so assert on textContent to see the leading spaces.
  await expect(note).toHaveText(/Indented first line\./)
  expect(await note.textContent()).toBe('    Indented first line.\n\nSecond paragraph.')
  await expect(note).toHaveCSS('white-space', 'pre-wrap')
})

test('mobile navigation has no horizontal overflow and applies theme changes', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await signIn(page)

  await expect(page.locator('.mobile-workspace-nav')).toBeVisible()
  await page.locator('.mobile-preference').first().selectOption('dark')
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark')
  const widths = await page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth }))
  expect(widths.scrollWidth).toBeLessThanOrEqual(widths.clientWidth + 1)
})
