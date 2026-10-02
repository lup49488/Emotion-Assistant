import { expect, test } from '@playwright/test'

const EMAIL_AUTH_CONFIG = {
  email_auth_enabled: true,
  legacy_login_enabled: true,
  turnstile_required: false,
  turnstile_site_key: '',
  turnstile_action: 'email-auth',
}

async function withEmailAuth(page) {
  await page.route('**/api/v1/auth/config', (route) => route.fulfill({ json: EMAIL_AUTH_CONFIG }))
}

test('each sign-in tab explains itself instead of always saying welcome back', async ({ page }) => {
  await withEmailAuth(page)
  await page.goto('/')

  await expect(page.getByRole('heading', { name: 'Welcome back' })).toBeVisible()
  await page.getByRole('tab', { name: 'Create account' }).click()
  await expect(page.getByRole('heading', { name: 'Create your account' })).toBeVisible()
  await page.getByRole('tab', { name: 'Migrate legacy account' }).click()
  await expect(page.getByRole('heading', { name: 'Move an older account' })).toBeVisible()
  await page.getByRole('tab', { name: 'Use legacy sign-in' }).click()
  await expect(page.getByRole('heading', { name: 'Sign in with a user ID' })).toBeVisible()
  // With email sign-up available the legacy form only signs in.
  await expect(page.getByRole('button', { name: 'Sign in', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Sign in / Register' })).toHaveCount(0)
})

test('the legacy form still offers registration when email sign-up is off', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Sign in / Register' })).toBeVisible()
})

test('a password can be revealed and hidden again', async ({ page }) => {
  await withEmailAuth(page)
  await page.goto('/')
  const field = page.getByLabel('Password', { exact: true })
  await field.fill('secret-value')

  await expect(field).toHaveAttribute('type', 'password')
  await page.getByRole('button', { name: 'Show password' }).click()
  await expect(field).toHaveAttribute('type', 'text')
  await page.getByRole('button', { name: 'Hide password' }).click()
  await expect(field).toHaveAttribute('type', 'password')
})

test('the language can be switched before signing in', async ({ page }) => {
  await withEmailAuth(page)
  await page.goto('/')
  await page.getByRole('button', { name: '中文' }).click()

  await expect(page.getByRole('heading', { name: '欢迎回来' })).toBeVisible()
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN')
})

test.describe('a Chinese-language browser', () => {
  test.use({ locale: 'zh-CN' })

  test('opens in Chinese on a first visit', async ({ page }) => {
    await withEmailAuth(page)
    await page.goto('/')
    await expect(page.getByRole('heading', { name: '欢迎回来' })).toBeVisible()
  })
})

test('a first visit does not probe for a session that cannot exist', async ({ page }) => {
  const probes = []
  page.on('request', (request) => {
    if (request.url().includes('/api/v1/auth/session')) probes.push(request.url())
  })
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Sign in / Register' })).toBeVisible()

  expect(probes).toEqual([])
})

test.describe('on a phone', () => {
  test.use({ viewport: { width: 375, height: 812 } })

  test('the form fits above the fold with zoom-safe fields and finger-sized tabs', async ({ page }) => {
    await withEmailAuth(page)
    await page.goto('/')
    const button = page.locator('.login-button')
    await expect(button).toBeVisible()

    const layout = await page.evaluate(() => ({
      buttonBottom: document.querySelector('.login-button').getBoundingClientRect().bottom,
      fontSizes: [...document.querySelectorAll('.login-form input')].map((input) => parseFloat(getComputedStyle(input).fontSize)),
      tabHeights: [...document.querySelectorAll('.login-mode-tabs button')].map((tab) => tab.getBoundingClientRect().height),
    }))

    // The submit button is reachable without scrolling.
    expect(layout.buttonBottom).toBeLessThanOrEqual(812)
    // iOS Safari zooms into any field smaller than 16px.
    for (const size of layout.fontSizes) expect(size).toBeGreaterThanOrEqual(16)
    for (const height of layout.tabHeights) expect(height).toBeGreaterThanOrEqual(44)
  })
})
