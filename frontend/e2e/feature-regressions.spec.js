import { expect, test } from '@playwright/test'
import os from 'node:os'
import path from 'node:path'

const apiUrl = process.env.E2E_API_BASE_URL || 'http://127.0.0.1:18000'
const imageBytes = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL5JwAAAABJRU5ErkJggg==', 'base64')
const exportFile = (name) => ({ name, mimeType: 'application/json', buffer: Buffer.from('[]') })

async function signIn(page) {
  await page.goto('/')
  await page.getByLabel('User ID').fill('e2e-user')
  await page.getByLabel('Access password').fill('e2e-password')
  await page.getByRole('button', { name: 'Sign in / Register' }).click()
  await expect(page.getByPlaceholder('Message Serenova')).toBeVisible()
}

async function managerSession(page) {
  await page.route('**/api/v1/auth/session', async (route) => {
    const response = await route.fetch()
    if (!response.ok()) return route.fulfill({ response })
    await route.fulfill({ response, json: { ...await response.json(), can_manage_knowledge: true, can_access_operations: true } })
  })
}

test.beforeEach(async ({ request }) => {
  await request.post(`${apiUrl}/__test/reset`)
})

test('knowledge uploads reset the file chooser after the asynchronous response', async ({ page }) => {
  const errors = []
  page.on('pageerror', (error) => errors.push(error.message))
  await managerSession(page)
  await page.route('**/api/v1/rag/evaluations/latest', (route) => route.fulfill({ json: { report: null } }))
  await page.route('**/api/v1/jobs?limit=20', (route) => route.fulfill({ json: { jobs: [] } }))
  await page.route('**/api/v1/rag/feedback/summary', (route) => route.fulfill({ json: {} }))
  let uploads = 0
  await page.route('**/api/v1/rag/documents', (route) => {
    uploads += 1
    return route.fulfill({ json: { job: { id: `upload-${uploads}`, kind: 'rag_upload', status: 'succeeded', progress: 100 } } })
  })
  await signIn(page)
  await page.getByRole('button', { name: 'Knowledge & RAG' }).click()
  const chooser = page.locator('.knowledge-upload-form input[type=file]')
  const document = { name: 'review.txt', mimeType: 'text/plain', buffer: Buffer.from('Synthetic document') }
  await chooser.setInputFiles(document)
  await expect.poll(() => uploads).toBe(1)
  await expect(chooser).toHaveValue('')
  await chooser.setInputFiles(document)
  await expect.poll(() => uploads).toBe(2)
  await expect(chooser).toHaveValue('')
  expect(errors).toEqual([])
})

test('partial photo upload failure retains the historical check-in and retries only failed photos', async ({ page, request }) => {
  const errors = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.clock.setFixedTime(new Date('2026-09-07T12:00:00Z'))
  const imageRequests = []
  let failRetryImage = true
  await page.route('**/api/v1/mood/checkins/*/images', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback()
    imageRequests.push(route.request().url())
    if (failRetryImage && route.request().postDataBuffer().includes(Buffer.from('retry.png'))) {
      return route.fulfill({ status: 503, json: { message: 'Photo upload temporarily unavailable' } })
    }
    await route.fallback()
  })
  await signIn(page)
  await page.getByRole('button', { name: 'Mood check-in' }).click()
  await page.getByTitle('Edit check-in 2026-07-29').click()
  await page.getByLabel('Photos').setInputFiles(['saved.png', 'retry.png'].map((name) => ({ name, mimeType: 'image/png', buffer: imageBytes })))
  await page.getByRole('button', { name: 'Update check-in' }).click()
  await expect(page.getByRole('alert')).toHaveText('Photo upload temporarily unavailable')
  await expect(page.getByRole('heading', { name: 'Editing the check-in for 2026-07-29' })).toBeVisible()
  await expect(page.locator('.mood-image-picker figure')).toHaveCount(1)
  await expect(page.getByRole('img', { name: 'retry.png', exact: true })).toBeVisible()
  await expect(page.locator('.mood-reflection-callout')).toHaveCount(0)
  await page.screenshot({ path: path.join(os.tmpdir(), 'serenova-review-photo-retry.png'), fullPage: false })
  failRetryImage = false
  await page.getByRole('button', { name: 'Update check-in' }).click()
  await expect(page.locator('.mood-reflection-callout')).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.locator('.mood-image-picker figure')).toHaveCount(0)
  expect(imageRequests).toHaveLength(3)
  expect(imageRequests.every((url) => url.endsWith('/2026-07-29/images'))).toBe(true)
  const { records } = await (await request.get(`${apiUrl}/api/v1/mood/checkins`)).json()
  expect(records).toHaveLength(1)
  expect(records[0].date).toBe('2026-07-29')
  expect(records[0].images).toHaveLength(2)
  await expect(page).toHaveTitle('Serenova')
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
  expect(errors).toEqual([])
})

test('external import preview ignores an older file response and clears when selection is removed', async ({ page }) => {
  let oldRoute
  await page.route('**/api/v1/import/external/preview', async (route) => {
    if (route.request().postDataBuffer().includes(Buffer.from('old.json'))) {
      oldRoute = route
      return
    }
    await route.fulfill({ json: { source: 'chatgpt', conversations: 1, messages: 2, sample_titles: ['Current file conversation'], profile_fields: [] } })
  })
  await signIn(page)
  await page.getByRole('button', { name: 'Privacy & export' }).click()
  const chooser = page.getByLabel('External AI export')
  await chooser.setInputFiles(exportFile('old.json'))
  await expect.poll(() => Boolean(oldRoute)).toBe(true)
  await chooser.setInputFiles(exportFile('current.json'))
  await expect(page.getByText('Current file conversation')).toBeVisible()
  await oldRoute.fulfill({ json: { source: 'claude', conversations: 99, messages: 200, sample_titles: ['Stale file conversation'], profile_fields: [] } })
  // Wait until the deferred response reached the page before checking it cannot replace the preview.
  await page.waitForLoadState('networkidle')
  await expect(page.getByText('Current file conversation')).toBeVisible()
  await expect(page.getByText('Stale file conversation')).toHaveCount(0)
  await chooser.setInputFiles([])
  await expect(page.getByRole('button', { name: 'Import reviewed data' })).toHaveCount(0)
})

test('operations window changes ignore slower responses for an older window', async ({ page }) => {
  await managerSession(page)
  let oldRoute
  const dashboard = (days) => ({ window_days: days, alerts: [], http: { requests: days, traffic: {}, top_paths: [], statuses: {} }, jobs: { counts: {}, recent_failures: [] }, provider_failures: {}, runtime: {} })
  await page.route('**/api/v1/operations/dashboard?*', async (route) => {
    const days = Number(new URL(route.request().url()).searchParams.get('days'))
    if (days === 30) { oldRoute = route; return }
    await route.fulfill({ json: dashboard(days) })
  })
  await signIn(page)
  await page.getByRole('button', { name: 'Operations', exact: true }).first().click()
  const displayedWindow = page.locator('.stat').filter({ has: page.getByText('Selected window', { exact: true }) }).locator('strong')
  await expect(displayedWindow).toHaveText('7d')
  await page.getByLabel('Selected window').selectOption('30')
  await expect.poll(() => Boolean(oldRoute)).toBe(true)
  await page.getByLabel('Selected window').selectOption('1')
  await expect(displayedWindow).toHaveText('1d')
  await oldRoute.fulfill({ json: dashboard(30) })
  await page.waitForLoadState('networkidle')
  await expect(displayedWindow).toHaveText('1d')
  await expect(page.getByLabel('Selected window')).toHaveValue('1')
})
