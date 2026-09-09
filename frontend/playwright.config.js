import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  // One mock API process serves every spec file and each file's beforeEach
  // resets its global state, so parallel workers sign each other out.
  workers: 1,
  use: { baseURL: process.env.E2E_BASE_URL || 'http://127.0.0.1:4174', screenshot: 'only-on-failure', trace: 'retain-on-failure' },
})
