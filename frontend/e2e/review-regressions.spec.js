import { expect, test } from '@playwright/test'

// Regressions found by reviewing the deployed site as a user.

const testApiBaseUrl = process.env.E2E_API_BASE_URL || 'http://127.0.0.1:18000'

async function signIn(page) {
  await page.goto('/')
  await page.getByLabel('User ID').fill('e2e-user')
  await page.getByLabel('Access password').fill('e2e-password')
  await page.getByRole('button', { name: 'Sign in / Register' }).click()
  await expect(page.getByPlaceholder('Message Serenova')).toBeVisible()
}

test.beforeEach(async ({ request }) => {
  await request.post(`${testApiBaseUrl}/__test/reset`)
})

test('a reply cut at the length limit says so and can be continued', async ({ page }) => {
  await signIn(page)
  await page.getByPlaceholder('Message Serenova').fill('Trigger truncated reply')
  await page.getByRole('button', { name: 'Send message' }).click()

  const notice = page.getByRole('status').filter({ hasText: 'reached the length limit' })
  await expect(notice).toBeVisible()
  await notice.getByRole('button', { name: 'Continue' }).click()
  await expect(page.getByText('Please continue from where you stopped.')).toBeVisible()
})

test('the send button and the message box have different names', async ({ page }) => {
  await signIn(page)
  await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Message Serenova' })).toHaveCount(0)
})

test('the English tone label uses English punctuation', async ({ page }) => {
  await signIn(page)
  const tone = page.getByRole('button', { name: /Conversation tone/ }).first()
  await expect(tone).toContainText('Conversation tone: ')
  await expect(tone).not.toContainText('：')
})

for (const width of [800, 600]) {
  test.describe(`a ${width}px window`, () => {
    test.use({ viewport: { width, height: 900 } })

    test('the menu button opens the conversation drawer', async ({ page }) => {
      await signIn(page)
      await page.getByRole('button', { name: 'Open sidebar' }).click()

      // Between 721 and 900px this button used to set the state and show nothing.
      const newChat = page.locator('.mobile-sidebar').getByRole('button', { name: 'New chat' })
      await expect(newChat).toBeVisible()
      await expect(newChat).toBeInViewport()
    })
  })
}

async function createCheckin(page) {
  const csrf = (await page.context().cookies()).find((cookie) => cookie.name === 'chatbot_csrf')?.value || ''
  await page.request.post(`${testApiBaseUrl}/api/v1/mood/checkins`, {
    headers: { 'X-CSRF-Token': csrf },
    data: { mood: 'overwhelmed but hopeful', intensity: 4, note: 'A long note that wraps across several lines on a narrow phone.' },
  })
}

test.describe('a 320px phone', () => {
  test.use({ viewport: { width: 320, height: 640 } })

  test('mood record actions stay on screen instead of being clipped', async ({ page }) => {
    await signIn(page)
    await createCheckin(page)
    await page.getByRole('button', { name: 'Mood check-in' }).click()
    await page.locator('.mood-period-tabs').getByRole('button', { name: 'All' }).click()

    const actions = page.locator('.record-row .record-actions button')
    await expect(actions.first()).toBeVisible()
    for (const box of await actions.evaluateAll((buttons) => buttons.map((button) => button.getBoundingClientRect().right))) {
      expect(box).toBeLessThanOrEqual(320)
    }
  })
})

test.describe('a touch screen', () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true })

  test('message actions are large enough for a fingertip', async ({ page }) => {
    await signIn(page)
    await page.getByPlaceholder('Message Serenova').fill('Hello there')
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect(page.getByText('Grounded answer from the knowledge base.')).toBeVisible()

    const sizes = await page.locator('.message-action').evaluateAll((buttons) => buttons.map((button) => {
      const box = button.getBoundingClientRect()
      return Math.min(box.width, box.height)
    }))
    expect(sizes.length).toBeGreaterThan(0)
    for (const size of sizes) expect(size).toBeGreaterThanOrEqual(44)
  })
})
