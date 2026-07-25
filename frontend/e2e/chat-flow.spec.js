import { expect, test } from '@playwright/test'

test('login, cited reply, feedback, and regeneration work together', async ({ page }) => {
  await page.goto('/')
  await page.getByLabel('User ID').fill('e2e-user')
  await page.getByLabel('Access password').fill('e2e-password')
  await page.getByRole('button', { name: 'Sign in / Register' }).click()
  await expect(page.getByPlaceholder('Message Serenova')).toBeVisible()

  await page.getByText('Knowledge retrieval').click()
  await page.getByPlaceholder('Message Serenova').fill('How do I deploy this project?')
  await page.getByTitle('Message Serenova').click()
  await expect(page.getByText('Grounded answer from the knowledge base.')).toBeVisible()
  await expect(page.getByText('Sources')).toBeVisible()
  await expect(page.getByText('deployment_guide.md')).toBeVisible()

  await page.getByTitle('Helpful sources', { exact: true }).click()
  await expect(page.getByTitle('Helpful sources', { exact: true })).toHaveClass(/selected/)
  await page.getByTitle('Regenerate').click()
  await expect(page.getByText('Regenerated answer')).toBeVisible()
})

test('assistant replies render HTML line breaks and LaTeX delimiters safely', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByPlaceholder('Message Serenova')).toBeVisible()
  await page.getByPlaceholder('Message Serenova').fill('Show markdown')
  await page.getByTitle('Message Serenova').click()

  await expect(page.getByText('First line')).toBeVisible()
  await expect(page.getByText('Second line')).toBeVisible()
  await expect(page.locator('.katex-display')).toHaveCount(1)
  await expect(page.getByText('<br>', { exact: true })).toHaveCount(0)
})
