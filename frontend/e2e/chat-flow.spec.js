import { expect, test } from '@playwright/test'

const testApiBaseUrl = process.env.E2E_API_BASE_URL || 'http://127.0.0.1:18000'

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

async function enableKnowledge(page) {
  await page.getByRole('button', { name: 'Model & reply style' }).first().click()
  const settings = page.getByRole('complementary', { name: 'Model & reply style' })
  await settings.getByText('Knowledge retrieval').click()
  await settings.getByRole('button', { name: 'Hide preferences' }).click()
}

test.beforeEach(async ({ request }) => {
  await request.post(`${testApiBaseUrl}/__test/reset`)
})

test('login, cited reply, feedback, and regeneration work together', async ({ page }) => {
  await signIn(page)

  await enableKnowledge(page)
  await sendMessage(page, 'How do I deploy this project?')
  await expect(page.getByText('Grounded answer from the knowledge base.')).toBeVisible()
  await expect(page.getByText('Sources')).toBeVisible()
  await expect(page.getByText('deployment_guide.md')).toBeVisible()

  await page.getByTitle('Helpful sources', { exact: true }).click()
  await expect(page.getByTitle('Helpful sources', { exact: true })).toHaveClass(/selected/)
  await page.getByTitle('Regenerate').click()
  await expect(page.getByText('Regenerated answer')).toBeVisible()
})

test('English welcome prompts submit their localized message immediately', async ({ page }) => {
  await signIn(page)

  await page.getByRole('button', { name: 'Talk through a feeling' }).click()
  await expect(page.getByText('I feel a little anxious today. Could you help me sort through it?')).toBeVisible()
  await expect(page.getByText('Grounded answer from the knowledge base.')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Talk through a feeling' })).toHaveCount(0)
})

test('a saved message can be quoted for a focused follow-up', async ({ page }) => {
  await signIn(page)
  await sendMessage(page, 'Explain a deployment step')
  await expect(page.getByText('Grounded answer from the knowledge base.')).toBeVisible()

  await page.getByTitle('Reply to this message').last().click()
  await expect(page.getByText('Replying to Serenova')).toBeVisible()
  await sendMessage(page, 'Could you expand that?')
  await expect(page.getByText('Quoted reply with the selected context.')).toBeVisible()
  await expect(page.getByText('Replying to Serenova')).toHaveCount(1)
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
  await enableKnowledge(page)
  await sendMessage(page, 'Ask without sources')

  await expect(page.getByRole('status')).toHaveText(/does not contain enough relevant information/)
  await expect(page.getByTitle('Regenerate')).toHaveCount(0)
})

test('an unenforced insufficient-evidence status still shows the generated reply', async ({ page }) => {
  await signIn(page)
  await enableKnowledge(page)
  await sendMessage(page, 'Ask without sources unenforced')

  await expect(page.getByText('General answer without sources.')).toBeVisible()
  await expect(page.getByRole('status')).toHaveCount(0)
})

test('conversation expression history is not shown in personal data', async ({ page }) => {
  await signIn(page)
  await sendMessage(page, 'I feel happy today')
  await expect(page.getByText('Grounded answer from the knowledge base.')).toBeVisible()

  await page.getByRole('button', { name: 'Personal data' }).click()
  await expect(page.getByText('Conversation emotion history')).toHaveCount(0)
  await expect(page.getByText('Conversation expression references')).toHaveCount(0)
})

test('knowledge workspace presents searchable source material', async ({ page }) => {
  await signIn(page)

  await page.getByRole('button', { name: 'Knowledge & RAG' }).click()
  await expect(page.getByRole('heading', { name: 'Knowledge & RAG' })).toBeVisible()
  await expect(page.getByText('486', { exact: true })).toBeVisible()

  await page.getByPlaceholder('Ask a question to test retrieval').fill('What can help with sleep?')
  await page.getByRole('button', { name: 'Retrieval check' }).click()
  await expect(page.getByText('sleep_hygiene_cn.md · #12')).toBeVisible()
  await expect(page.getByText('固定起床时间和固定入睡时间，能帮助形成更稳定的睡眠节律。')).toBeVisible()
})

test('interest memories can be added, edited, and deleted in personal data', async ({ page }) => {
  await signIn(page)
  await page.getByRole('button', { name: 'Personal data' }).click()

  const interests = page.locator('.memory-v2-primary .data-panel').filter({ has: page.getByRole('heading', { name: 'Interests' }) })
  await interests.getByRole('button', { name: 'Add memory' }).click()
  await interests.getByLabel('Memory text').fill('I enjoy hiking')
  await interests.getByRole('button', { name: 'Save memory' }).click()
  await expect(interests.getByText('I enjoy hiking', { exact: true })).toBeVisible()

  await interests.getByRole('button', { name: 'Edit memory' }).click()
  await interests.getByLabel('Memory text').fill('I enjoy hiking with friends')
  await interests.getByRole('button', { name: 'Save memory' }).click()
  await expect(interests.getByText('I enjoy hiking with friends', { exact: true })).toBeVisible()

  await interests.getByRole('button', { name: 'Delete memory' }).click()
  await expect(interests.getByText('I enjoy hiking with friends', { exact: true })).toHaveCount(0)
})

test('memory saving policy can be changed from personal data', async ({ page }) => {
  await signIn(page)
  await page.getByRole('button', { name: 'Personal data' }).click()

  const policy = page.getByRole('radio', { name: /Ask before saving/ })
  await expect(policy).toBeChecked()
  await page.getByRole('radio', { name: /Do not save automatically/ }).check()
  await expect(page.getByRole('radio', { name: /Do not save automatically/ })).toBeChecked()
  await expect(page.getByText('Conversation memory extraction is disabled. Manual changes remain available.')).toBeVisible()
})

test('chat header controls set a gentle tone and open the model reply window', async ({ page }) => {
  await signIn(page)

  await page.getByRole('button', { name: /Conversation tone/ }).click()
  await expect(page.getByRole('menu', { name: 'Conversation tone' })).toBeVisible()
  await expect(page.getByRole('menuitemradio', { name: 'Gentle' })).toHaveCount(1)
  await page.getByRole('menuitemradio', { name: 'Gentle' }).click()
  await expect(page.getByRole('button', { name: /Conversation tone/ })).toContainText('Gentle')

  await page.getByRole('button', { name: 'Model & reply style' }).first().click()
  const settings = page.getByRole('complementary', { name: 'Model & reply style' })
  await expect(settings).toBeVisible()
  await settings.getByRole('button', { name: /NVIDIA NIM/ }).click()
  await expect(settings.getByRole('button', { name: 'openai/gpt-oss-20b', exact: true })).toHaveClass(/selected/)
  await expect(settings.getByText('Reply profile')).toBeVisible()
})

test('feature workspace uses the V2 topbar without the chat conversation sidebar', async ({ page }) => {
  await page.setViewportSize({ width: 1366, height: 768 })
  await signIn(page)
  await page.getByRole('button', { name: 'Personal data' }).click()

  await expect(page.locator('.workspace-topbar')).toBeVisible()
  await expect(page.getByText('Serenova', { exact: true })).toBeVisible()
  await expect(page.locator('.conversation-sidebar')).toHaveCount(0)
  await page.getByRole('group', { name: 'Language' }).getByRole('button', { name: 'Chinese' }).click()
  await expect(page.getByRole('button', { name: '个人数据' })).toBeVisible()
  await page.getByRole('group', { name: '语言' }).getByRole('button', { name: 'English' }).click()
  await expect(page.getByRole('button', { name: 'Personal data' })).toBeVisible()
})

test('desktop reply context uses a dedicated right workspace column', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 900 })
  await signIn(page)
  await page.getByRole('button', { name: 'Reply context' }).click()

  await expect(page.getByRole('complementary', { name: 'Reply context' })).toBeVisible()
  const columns = await page.locator('.chat-workspace-content').evaluate((element) => getComputedStyle(element).gridTemplateColumns)
  expect(columns.trim().split(/\s+/)).toHaveLength(3)

  const sidebarTypeScale = await page.getByRole('complementary', { name: 'Reply context' }).evaluate((panel) => ({
    heading: Number.parseFloat(getComputedStyle(panel.querySelector('h2')).fontSize),
    copy: Number.parseFloat(getComputedStyle(panel.querySelector('.context-copy')).fontSize),
  }))
  expect(sidebarTypeScale.heading).toBeGreaterThanOrEqual(22)
  expect(sidebarTypeScale.copy).toBeGreaterThanOrEqual(15)
})

test('a Serenova export file can be selected and imported', async ({ page }) => {
  await signIn(page)
  await page.getByRole('button', { name: 'Privacy & export' }).click()
  await page.getByLabel('Export JSON file').setInputFiles({ name: 'serenova-export.json', mimeType: 'application/json', buffer: Buffer.from('{"schema_version":4}') })
  await page.getByRole('button', { name: 'Import data' }).click()

  await expect(page.getByText('Imported 1 conversations, 1 memories, and 1 Mood Check-ins.')).toBeVisible()
})

test('an external AI export is previewed before its selected profile fields are imported', async ({ page }) => {
  await signIn(page)
  await page.getByRole('button', { name: 'Privacy & export' }).click()

  const chooser = page.getByLabel('External AI export')
  await chooser.setInputFiles({ name: 'conversations.json', mimeType: 'application/json', buffer: Buffer.from('[]') })
  await expect(page.getByText('Detected:')).toBeVisible()
  await expect(page.getByText('A useful imported conversation')).toBeVisible()
  await page.getByRole('checkbox', { name: 'Name: Example user' }).check()
  await page.getByRole('button', { name: 'Import reviewed data' }).click()
  await expect(page.getByText('Imported from chatgpt: 2 conversations and 1 selected profile fields.')).toBeVisible()
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

test('a Mood Check-in can retain and remove a private image attachment', async ({ page }) => {
  await signIn(page)
  await page.getByRole('button', { name: 'Mood check-in' }).click()
  await page.getByRole('textbox', { name: 'Mood' }).fill('hopeful')
  await page.getByLabel('Photos').setInputFiles({ name: 'mood.png', mimeType: 'image/png', buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL5JwAAAABJRU5ErkJggg==', 'base64') })
  await page.getByRole('button', { name: 'Save check-in' }).click()

  const image = page.getByRole('img', { name: 'Mood Check-in photo' })
  await expect(image).toBeVisible()
  await page.getByTitle('Remove photo').click()
  await expect(image).toHaveCount(0)
})

test('a user-selected Mood Check-in is sent to a new chat as bounded context', async ({ page }) => {
  await signIn(page)
  await page.getByRole('button', { name: 'Mood check-in' }).click()
  await page.getByRole('textbox', { name: 'Mood' }).fill('anxious')
  await page.getByLabel('Note').fill('Interview tomorrow')
  await page.getByRole('button', { name: 'Save check-in' }).click()

  const savedCheckin = page.locator('.mood-reflection-callout')
  await expect(savedCheckin).toBeVisible()
  await savedCheckin.getByRole('button', { name: 'Discuss record and trend' }).click()
  await expect(page.locator('.message.user .message-body').last()).toHaveText("I'd like to talk about this mood check-in and its recent trend.")
  await expect(page.locator('.message.assistant .message-body').last()).toHaveText('Mood reflection: anxious (3/5) - Interview tomorrow')
})

test('a historical Mood Check-in can be discussed from its record action', async ({ page }) => {
  await signIn(page)
  await page.getByRole('button', { name: 'Mood check-in' }).click()

  await page.locator('.record-actions').first().getByTitle('Discuss record and trend').click()
  await expect(page.locator('.message.user .message-body').last()).toHaveText("I'd like to talk about this mood check-in and its recent trend.")
  await expect(page.locator('.message.assistant .message-body').last()).toContainText('Mood reflection: calm (3/5)')
})

test('tablet Mood Check-in keeps form controls within their panel', async ({ page }) => {
  await page.setViewportSize({ width: 1366, height: 768 })
  await signIn(page)
  await page.getByRole('button', { name: 'Mood check-in' }).click()
  await expect(page.getByRole('heading', { name: 'One note for today is enough' })).toBeVisible()

  const bounds = await page.locator('.mood-page').evaluate((pageElement) => {
    const form = pageElement.querySelector('.checkin-form')
    const moodInput = form?.querySelector("input:not([type='range']):not([type='file'])")
    if (!form || !moodInput) return null
    const formBounds = form.getBoundingClientRect()
    const inputBounds = moodInput.getBoundingClientRect()
    return {
      formRight: formBounds.right,
      inputRight: inputBounds.right,
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }
  })
  expect(bounds).not.toBeNull()
  expect(bounds.inputRight).toBeLessThanOrEqual(bounds.formRight + 1)
  expect(bounds.scrollWidth).toBeLessThanOrEqual(bounds.clientWidth + 1)
})

test('mobile sidebar navigation has no horizontal overflow and applies theme changes', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await signIn(page)

  await expect(page.locator('.mobile-app-bar')).toBeVisible()
  await page.getByTitle('Open sidebar').click()
  await expect(page.locator('.mobile-sidebar')).toBeVisible()
  await expect(page.locator('.mobile-sidebar').getByRole('button', { name: 'Personal data' })).toBeVisible()
  await expect(page.locator('.mobile-sidebar').getByText('Conversations')).toBeVisible()
  await page.locator('.mobile-sidebar .preferences-controls select').first().selectOption('dark')
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark')
  await page.locator('.mobile-sidebar').getByRole('button', { name: 'Personal data' }).click()
  await expect(page.locator('.mobile-sidebar')).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'What your assistant remembers about you' })).toBeVisible()
  const widths = await page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth }))
  expect(widths.scrollWidth).toBeLessThanOrEqual(widths.clientWidth + 1)
})
