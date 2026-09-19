# Email Authentication Migration Checklist

## Purpose

Move regular Serenova users from the current `user_id` plus access-key entry
flow to an email-verified account and password. Existing users must be able to
bind an email to their existing account without copying, renaming, or otherwise
moving their conversations, memories, uploads, or mood check-ins.

This document is an implementation contract and rollout checklist. It does not
change the current authentication behavior by itself.

## Scope and invariants

- [ ] Keep `user_id` as the immutable, internal owner key for every existing
  store, file path, session, conversation, memory, image, and mood record.
- [ ] Create new accounts with a server-generated opaque `user_id`; never let
  email/password registration claim an arbitrary existing user ID.
- [ ] Add one normalized email identity per user, enforced with a global unique
  constraint.
- [ ] Bind a legacy account in place only after both an email challenge and the
  legacy `user_id` plus access-key proof succeed.
- [ ] Leave legacy login available during a published migration window. Do not
  remove it in the same release that introduces email login.
- [ ] Keep Cloudflare Access and MFA for the administrator application or
  administrator-only routes. Regular-user email login must not grant admin
  privileges.
- [ ] Treat Turnstile as bot mitigation only. It never establishes identity and
  never replaces server-side email verification.

## Decisions required before implementation

- [ ] Choose an outbound email provider and a verified sender domain.
- [ ] Choose the public migration period and legacy-login retirement criteria.
- [ ] Choose the user-facing support path for a lost legacy access key.
- [ ] Confirm the production storage backend. SQLite is the preferred
  authoritative registry for email uniqueness and migration transactions.
- [ ] Choose a password policy and Argon2id cost that remains safe under the
  production memory and concurrent-login budget.
- [ ] Choose a separate Cloudflare Access hostname or exact protected admin
  paths, plus the administrator allowlist and MFA policy.
- [ ] Define retention periods for verification challenges, rate-limit records,
  and audit events, and add them to the privacy notice and export/delete policy.

## Data model

Implement these tables in the centralized authentication registry. Email values
are normalized with a documented, conservative rule before storage and lookup.

- [ ] `email_identities`: `user_id` foreign key, `email_normalized` unique,
  `email_verified_at`, `created_at`, `updated_at`, `legacy_migrated_at`, and
  an authentication-state marker.
- [ ] `password_credentials`: `user_id` primary key, versioned Argon2id hash,
  hash parameters, `changed_at`, and a credential/session version.
- [ ] `email_challenges`: opaque challenge ID, email reference, purpose
  (`registration` or `legacy_migration`), code/token hash, expiry, attempts,
  send count, consumed timestamp, and creation metadata.
- [ ] `migration_audit_events`: user ID where known, event type, outcome,
  request correlation ID, and timestamp. Do not store passwords, access keys,
  raw email verification codes, raw Turnstile tokens, or full IP addresses.
- [ ] Add a transactional unique-email binding operation. Concurrent attempts
  must permit one successful bind and leave all others unchanged.
- [ ] Add schema migration and downgrade/compatibility tests. Reject a database
  newer than the running application rather than silently changing its version.

## Authentication API contract

All endpoints are POST-only, origin-checked, rate-limited, and return generic
public errors that do not reveal whether an email, user ID, or legacy credential
exists.

- [ ] `POST /api/v1/auth/email/start`: accepts email, purpose, and Turnstile
  token; validates Turnstile server-side; queues a code if permitted; returns a
  generic success response either way.
- [ ] `POST /api/v1/auth/email/verify`: accepts challenge ID and code; enforces
  one use, purpose, expiry, and attempt limits; returns a short-lived,
  purpose-bound verified-intent token.
- [ ] `POST /api/v1/auth/register`: accepts a registration intent token and a
  password; creates an opaque user ID, identity, credential, and signed session
  in one transaction.
- [ ] `POST /api/v1/auth/migrate-legacy`: accepts a migration intent token,
  legacy user ID, legacy access key, and new password; verifies all proofs,
  binds the email to that existing user ID, changes credentials, invalidates old
  sessions, and writes an audit event atomically.
- [ ] `POST /api/v1/auth/password/login`: accepts email and password; applies
  abuse controls and issues the existing signed session cookies for the mapped
  user ID.
- [ ] Keep `POST /api/v1/auth/login` as the explicitly labeled legacy path
  until retirement criteria are met. It must retain its existing origin and
  throttling protections.
- [ ] Version password hashes. Verify old PBKDF2 credentials during the
  compatibility window and rehash successfully authenticated credentials with
  Argon2id when safe to do so.

## Challenge, rate-limit, and recovery controls

- [ ] Make verification codes short-lived (five to ten minutes), single-use,
  cryptographically random, and stored only as a keyed hash.
- [ ] Rate-limit challenge sends by normalized email and privacy-preserving
  client/network key; return `Retry-After` without confirming account state.
- [ ] Rate-limit code guesses and legacy-migration failures independently from
  ordinary password login failures.
- [ ] Limit active challenges per email and invalidate older challenges when a
  new challenge is issued.
- [ ] Require a fresh Turnstile token for each protected submission and verify
  its action and hostname through Cloudflare Siteverify before proceeding.
- [ ] Add password-login throttling by both account/email and client/network
  key. Do not clear all relevant failure counters until successful completion.
- [ ] Provide a safe account-recovery procedure after migration. It must not
  turn support staff or a recovery endpoint into a bypass for email ownership.
- [ ] Log success and failure categories for operational review without logging
  secrets or enough personal data to reconstruct an account.

## Docker and deployment configuration

- [ ] Add file-backed Docker secrets for the email provider credential and the
  private Turnstile secret; never place either in `VITE_*`, committed `.env`,
  browser bundles, logs, or error responses.
- [ ] Add only the public Turnstile sitekey to the browser configuration.
- [ ] Add startup validation that rejects public email authentication when its
  secret files, configured public hostname, secure cookies, or sender settings
  are absent.
- [ ] Configure distinct development, staging, and production Turnstile
  widgets and restrict each production widget to expected hostnames.
- [ ] Keep administrator routes behind Cloudflare Access and enforce MFA in the
  Access policy. Application authorization must still check the mapped internal
  user ID after the edge check.
- [ ] Add readiness checks for the local authentication store only. Do not make
  transient email-provider availability a reason to report an unhealthy chat
  service.

## User experience and migration communication

- [ ] Replace the general login form with email/password sign-in, registration,
  and a clearly separate "Migrate legacy account" path.
- [ ] Explain that migration preserves the user's existing conversations,
  memories, images, and mood history because it keeps the same internal account.
- [ ] Do not display raw user IDs after registration except where needed for
  recovery/support.
- [ ] Use neutral responses such as "If this address can receive a code, we
  sent one" for challenge requests and "We could not complete verification" for
  failed migration proof.
- [ ] Provide accessible loading, retry, expiry, and Turnstile-refresh states
  on desktop and mobile.
- [ ] Announce the legacy-login retirement date in-product and provide a
  migration reminder only after a successful legacy login.
- [ ] Do not silently link a second legacy account to an email that is already
  bound. Explain the recovery/support option instead.

## Test plan

### Unit and storage tests

- [ ] Registering creates an opaque user ID and a unique email identity.
- [ ] Duplicate normalized email registration or binding fails without changing
  either account.
- [ ] A successful legacy migration preserves the original user ID and every
  referenced data path; no copy is created.
- [ ] Failed legacy migration rolls back the identity, password, session, and
  audit mutation as one unit.
- [ ] Email challenges reject expiry, replay, wrong purpose, excessive guesses,
  and concurrent consumption.
- [ ] Audit records never contain a raw password, access key, code, mail token,
  or Turnstile token.
- [ ] Legacy PBKDF2 compatibility and Argon2id rehash behavior are tested.

### API and integration tests

- [ ] All public challenge and login failure responses avoid account
  enumeration, including response status, body, and timing tolerance.
- [ ] Send, verify, registration, migration, password login, logout, CSRF, and
  session invalidation run against real FastAPI and an isolated SQLite database.
- [ ] The JSON-content compatibility path, if still supported in production,
  has the same unique-email and locking guarantees.
- [ ] Turnstile success, duplicate token, expiry, hostname/action mismatch, and
  verification-service failure are covered with a controlled Siteverify stub.
- [ ] Email-provider failure does not create an account or disclose whether an
  email exists.
- [ ] Parallel migration attempts for the same legacy account and for the same
  email have deterministic, safe outcomes.

### Browser and deployment acceptance

- [ ] Desktop and mobile registration, legacy migration, expired code retry,
  login, logout, and legacy-login compatibility have Playwright coverage.
- [ ] CI runs the isolated FastAPI/SQLite authentication integration suite, not
  only mocked browser APIs.
- [ ] Staging uses a test email sink and Turnstile test keys; no production mail
  or secrets are used by CI.
- [ ] Perform a reversible staging migration using a copy of representative
  data, including image references and mood records.
- [ ] Before retiring legacy login, review migration completion rate, recovery
  events, failed attempts, and support volume without exposing personal data in
  dashboards.

## Rollout gates

- [ ] Release A: ship schema, secrets plumbing, challenge delivery, and tests;
  preserve the existing login as the default path.
- [ ] Release B: enable email registration and migration behind a feature flag
  in staging, then production.
- [ ] Release C: make email/password the default user interface while retaining
  legacy login as a visible compatibility option.
- [ ] Release D: after the announced window and support review, disable legacy
  login behind a reversible server-side flag. Do not delete legacy credentials
  until retention and recovery requirements are satisfied.
- [ ] Release E: remove legacy code only after a separate backup/restore drill
  and a reviewed migration completion report.

## Definition of done

- [ ] A new user can register and sign in with a verified email and password.
- [ ] An old user can migrate using an email code plus existing credentials and
  immediately sees the same historical account and data.
- [ ] An attacker cannot use Turnstile alone, a replayed code, an email guess,
  or an already-bound email to claim an account.
- [ ] Administrators remain protected by Cloudflare Access plus MFA.
- [ ] Secrets remain file-backed Docker secrets, and all critical tests pass in
  CI before the production rollout begins.
