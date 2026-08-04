import { useCallback, useEffect, useState } from 'react'
import { Check, Download, Pencil, Plus, RefreshCw, Search, ShieldAlert, Sparkles, Trash2, Undo2, Upload, X } from 'lucide-react'
import { apiFetch, csrfHeaders, readJson } from './api'

const json = (value) => JSON.stringify(value, null, 2)

function PageHeader({ title, description, action }) {
  return <header className="feature-header"><div><h1>{title}</h1><p>{description}</p></div>{action}</header>
}

function ErrorText({ error }) { return error ? <p className="inline-error" role="alert">{error}</p> : null }

function Loading({ loading, t }) { return loading ? <span className="loading-label"><RefreshCw size={14} className="spin" />{t('loading')}</span> : null }

export function MemoryPage({ t }) {
  const [snapshot, setSnapshot] = useState(null)
  const [quality, setQuality] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [editingIndex, setEditingIndex] = useState(null)
  const [memoryDraft, setMemoryDraft] = useState('')
  const [memorySaveMode, setMemorySaveMode] = useState('confirm')
  const refresh = async () => {
    setLoading(true); setError('')
    try {
      const [memory, report, preference] = await Promise.all([readJson('/api/v1/memory'), readJson('/api/v1/memory/quality'), readJson('/api/v1/memory/preference')])
      setSnapshot(memory); setQuality(report.report); setMemorySaveMode(preference.mode)
    } catch (requestError) { setError(requestError.message) } finally { setLoading(false) }
  }
  useEffect(() => { refresh() }, [])
  const saveLongTerm = async (items) => {
    setError('')
    try {
      const updated = await readJson('/api/v1/memory/long-term', { method: 'PUT', headers: csrfHeaders(), body: JSON.stringify({ items }) })
      setSnapshot(updated)
      setEditingIndex(null)
    } catch (requestError) { setError(requestError.message) }
  }
  const startEdit = (index) => { setEditingIndex(index); setMemoryDraft(snapshot.long_memory[index]?.text || '') }
  const saveEdit = async (index) => {
    const text = memoryDraft.trim()
    if (!text) { setError(t('memoryRequired')); return }
    await saveLongTerm(snapshot.long_memory.map((item, itemIndex) => itemIndex === index ? { ...item, text } : item))
  }
  const addMemory = () => {
    const next = [...snapshot.long_memory, { text: '', kind: 'manual' }]
    setSnapshot((current) => ({ ...current, long_memory: next }))
    setEditingIndex(next.length - 1)
    setMemoryDraft('')
  }
  const removeMemory = async (index) => { await saveLongTerm(snapshot.long_memory.filter((_, itemIndex) => itemIndex !== index)) }
  const updateMemorySaveMode = async (mode) => {
    const previousMode = memorySaveMode
    setMemorySaveMode(mode); setError('')
    try {
      const saved = await readJson('/api/v1/memory/preference', { method: 'PUT', headers: csrfHeaders(), body: JSON.stringify({ mode }) })
      setMemorySaveMode(saved.mode)
    } catch (requestError) { setMemorySaveMode(previousMode); setError(requestError.message) }
  }
  const undoAudit = async (eventId) => {
    setError('')
    try {
      const updated = await readJson(`/api/v1/memory/audit/${encodeURIComponent(eventId)}/undo`, { method: 'POST', headers: csrfHeaders() })
      setSnapshot(updated)
    } catch (requestError) { setError(requestError.message) }
  }
  const resolvePending = async (pendingId, action) => {
    setError('')
    const suffix = action === 'confirm' ? '/confirm' : ''
    const method = action === 'confirm' ? 'POST' : 'DELETE'
    try {
      const updated = await readJson(`/api/v1/memory/pending/${encodeURIComponent(pendingId)}${suffix}`, { method, headers: csrfHeaders() })
      setSnapshot(updated)
    } catch (requestError) { setError(requestError.message) }
  }
  const pendingMemory = snapshot?.pending_memory || []
  const summary = snapshot ? [
    [t('history'), snapshot.history.length], [t('emotion'), snapshot.emotion_memory.length], [t('longTerm'), snapshot.long_memory.length], [t('pendingMemories'), pendingMemory.length], [t('stableProfile'), snapshot.stable_profile.length], [t('interests'), snapshot.interest_memory.length],
  ] : []
  return <section className="feature-page">
    <PageHeader title={t('memoryTitle')} description={t('memoryDescription')} action={<button className="secondary-button" onClick={refresh}><RefreshCw size={16} />{t('refresh')}</button>} />
    <ErrorText error={error} /><Loading loading={loading} t={t} />
    {snapshot && <>
      <div className="stat-grid">{summary.map(([label, value]) => <div className="stat" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
      <div className="two-column"><section className="data-panel"><h2>{t('memoryQuality')}</h2><p className="report-text">{quality || t('noQualityResult')}</p></section><section className="data-panel"><h2>{t('memorySaveMode')}</h2><label className="memory-mode-field">{t('memorySaveModeLabel')}<select value={memorySaveMode} onChange={(event) => updateMemorySaveMode(event.target.value)}><option value="auto">{t('memorySaveAuto')}</option><option value="confirm">{t('memorySaveConfirm')}</option><option value="off">{t('memorySaveOff')}</option></select></label><p className="data-panel-copy">{t(`memorySaveMode_${memorySaveMode}`)}</p></section></div>
      <section className="data-panel"><h2>{t('pendingMemories')}</h2><PendingMemoryList items={pendingMemory} onResolve={resolvePending} t={t} /></section>
      <section className="data-panel"><div className="panel-title-row"><h2>{t('longTermMemory')}</h2><button className="icon-button" title={t('addMemory')} onClick={addMemory}><Plus size={17} /></button></div><p className="data-panel-copy">{t('longTermDescription')}</p>{snapshot.long_memory.length === 0 ? <p className="empty-state">{t('noLongTerm')}</p> : <div className="memory-list">{snapshot.long_memory.map((item, index) => <article className="memory-row" key={`${item.time || 'manual'}-${index}`}>{editingIndex === index ? <><textarea value={memoryDraft} rows="3" aria-label={t('memoryText')} onChange={(event) => setMemoryDraft(event.target.value)} /><div className="memory-row-actions"><button className="primary-button" onClick={() => saveEdit(index)}>{t('saveMemory')}</button><button className="secondary-button" onClick={() => { setEditingIndex(null); if (!item.text) setSnapshot((current) => ({ ...current, long_memory: current.long_memory.filter((_, itemIndex) => itemIndex !== index) })) }}><X size={15} />{t('cancel')}</button></div></> : <><div><p>{item.text}</p><span>{item.kind || 'memory'} {item.time ? `· ${String(item.time).slice(0, 10)}` : ''}</span></div><div className="memory-row-actions"><button className="icon-button" title={t('editMemory')} onClick={() => startEdit(index)}><Pencil size={15} /></button><button className="danger-icon" title={t('deleteMemory')} onClick={() => removeMemory(index)}><Trash2 size={16} /></button></div></>}</article>)}</div>}</section>
      <section className="data-panel"><h2>{t('interests')}</h2><JsonPreview value={snapshot.interest_memory} empty={t('noInterests')} /></section>
      <section className="data-panel"><h2>{t('emotionHistory')}</h2><EmotionMemoryList items={snapshot.emotion_memory} t={t} /></section>
      <section className="data-panel"><h2>{t('recentEvents')}</h2><MemoryAuditList events={snapshot.memory_events} onUndo={undoAudit} t={t} /></section>
    </>}
  </section>
}

function PendingMemoryList({ items, onResolve, t }) {
  if (items.length === 0) return <p className="empty-state">{t('noPendingMemories')}</p>
  return <div className="memory-list">{items.map((item) => {
    const candidate = item.candidate || {}
    return <article className="memory-row" key={item.id}><div><p>{candidate.text || t('memoryTitle')}</p><span>{item.created_at} · {item.reason}</span>{item.source_text && <span>{t('memorySource')}: {item.source_text}</span>}</div><div className="memory-row-actions pending-memory-actions"><button className="primary-button" title={t('confirmMemory')} onClick={() => onResolve(item.id, 'confirm')}><Check size={15} />{t('confirmMemory')}</button><button className="secondary-button" title={t('discardMemory')} onClick={() => onResolve(item.id, 'discard')}><X size={15} />{t('discardMemory')}</button></div></article>
  })}</div>
}

function MemoryAuditList({ events, onUndo, t }) {
  const sections = { stable: t('stableProfile'), interest: t('interests'), long: t('longTerm'), emotion: t('emotion'), none: t('memoryTitle') }
  const actions = { added: t('memoryEventAdded'), updated: t('memoryEventUpdated'), merged: t('memoryEventMerged'), unchanged: t('memoryEventUnchanged'), skipped: t('memoryEventSkipped'), pending: t('memoryEventPending'), confirmed: t('memoryEventConfirmed'), rejected: t('memoryEventRejected'), reverted: t('memoryEventReverted') }
  const visibleEvents = [...events].reverse().slice(0, 20)
  if (visibleEvents.length === 0) return <p className="empty-state">{t('noEvents')}</p>
  return <div className="memory-list">{visibleEvents.map((event) => <article className="memory-row memory-audit-row" key={event.id}><div><p><strong>{sections[event.section] || event.section} · {actions[event.action] || event.action}</strong></p><p>{event.text || t('memoryTitle')}</p><span>{event.time} · {event.reason}</span>{event.source_text && <span>{t('memorySource')}: {event.source_text}</span>}{event.undone_at && <span>{t('memoryEventReverted')}</span>}</div>{event.undoable && <button className="secondary-button" title={t('undoMemory')} onClick={() => onUndo(event.id)}><Undo2 size={15} />{t('undoMemory')}</button>}</article>)}</div>
}

export function MoodPage({ t }) {
  return <section className="feature-page"><PageHeader title={t('moodTitle')} description={t('moodDescription')} /><MoodCheckinContent t={t} /></section>
}

function MoodCheckinContent({ t }) {
  const [records, setRecords] = useState([])
  const [weekly, setWeekly] = useState(null)
  const emptyForm = { date: null, mood: '', intensity: 3, note: '' }
  const [form, setForm] = useState(emptyForm)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const isEditing = Boolean(form.date)
  const refresh = async () => {
    setLoading(true); setError('')
    try { const [all, trend] = await Promise.all([readJson('/api/v1/mood/checkins'), readJson('/api/v1/mood/weekly')]); setRecords(all.records); setWeekly(trend) } catch (requestError) { setError(requestError.message) } finally { setLoading(false) }
  }
  useEffect(() => { refresh() }, [])
  const submit = async (event) => {
    event.preventDefault(); setError('')
    try {
      // 后端按日期 upsert：带原日期提交即更新该天的记录，不带则记录今天。
      const payload = { mood: form.mood, intensity: form.intensity, note: form.note, ...(form.date ? { checkin_date: form.date } : {}) }
      await readJson('/api/v1/mood/checkins', { method: 'POST', headers: csrfHeaders(), body: JSON.stringify(payload) })
      setForm(emptyForm)
      await refresh()
    } catch (requestError) { setError(requestError.message) }
  }
  const startEdit = (record) => { setError(''); setForm({ date: record.date, mood: record.mood, intensity: record.intensity, note: record.note || '' }) }
  const cancelEdit = () => { setForm(emptyForm) }
  const remove = async (date) => { if (!window.confirm(`Delete the Mood Check-in for ${date}?`)) return; try { await apiFetch(`/api/v1/mood/checkins/${date}`, { method: 'DELETE', headers: csrfHeaders() }); if (form.date === date) setForm(emptyForm); await refresh() } catch (requestError) { setError(requestError.message) } }
  return <><div className="mood-page-actions"><button className="secondary-button" onClick={refresh}><RefreshCw size={16} />{t('refresh')}</button></div><ErrorText error={error} />
    <div className="two-column mood-top"><form className="data-panel checkin-form" onSubmit={submit}><h2>{isEditing ? `${t('editingCheckinFor')} ${form.date}` : t('todayCheckin')}</h2><label>{t('moodLabel')}<input required value={form.mood} placeholder={t('moodPlaceholder')} onChange={(event) => setForm((current) => ({ ...current, mood: event.target.value }))} /></label><label>{t('intensity')} <b>{form.intensity}/5</b><input type="range" min="1" max="5" value={form.intensity} onChange={(event) => setForm((current) => ({ ...current, intensity: Number(event.target.value) }))} /></label><label>{t('note')}<textarea rows="3" value={form.note} placeholder={t('notePlaceholder')} onChange={(event) => setForm((current) => ({ ...current, note: event.target.value }))} /></label><div className="checkin-form-actions"><button className="primary-button">{isEditing ? t('updateCheckin') : t('save')}</button>{isEditing && <button type="button" className="secondary-button" onClick={cancelEdit}><X size={16} />{t('cancelEdit')}</button>}</div></form><section className="data-panel trend-panel"><h2>{t('weeklyTrend')}</h2><MoodChart points={weekly?.points || []} /><p className="report-text">{weekly?.summary || t('noWeeklySummary')}</p><p className="report-text muted-report">{weekly?.analysis}</p></section></div>
    <section className="data-panel"><h2>{t('recentCheckins')}</h2><Loading loading={loading} t={t} />{records.length === 0 && !loading ? <p className="empty-state">{t('noCheckins')}</p> : <div className="record-list">{[...records].reverse().map((record) => <article className={`record-row ${form.date === record.date ? 'editing' : ''}`} key={record.date}><div><strong>{record.mood}</strong><span>{record.date} · {record.intensity}/5</span>{record.note && <p className="record-note">{record.note}</p>}</div><div className="record-actions"><button className="icon-button" title={`${t('editCheckin')} ${record.date}`} onClick={() => startEdit(record)}><Pencil size={16} /></button><button className="danger-icon" title={`${t('delete')} ${record.date}`} onClick={() => remove(record.date)}><Trash2 size={16} /></button></div></article>)}</div>}</section>
  </>
}

export function KnowledgePage({ t, canManageKnowledge = false }) {
  const [status, setStatus] = useState(null)
  const [quality, setQuality] = useState(null)
  const [latest, setLatest] = useState(null)
  const [feedback, setFeedback] = useState(null)
  const [jobs, setJobs] = useState([])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  // Comparing both modes runs the whole evaluation set twice, so keep it opt-in.
  const [compareModes, setCompareModes] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(true)
  const refresh = useCallback(async (showLoading = true) => { if (showLoading) setLoading(true); try { const requests = [readJson('/api/v1/rag/status'), readJson('/api/v1/rag/quality')]; if (canManageKnowledge) requests.push(readJson('/api/v1/rag/evaluations/latest'), readJson('/api/v1/jobs?limit=20'), readJson('/api/v1/rag/feedback/summary')); const [ragStatus, ragQuality, evaluation, jobData, feedbackSummary] = await Promise.all(requests); setStatus(ragStatus); setQuality(ragQuality); setLatest(evaluation?.report || null); setJobs(jobData?.jobs || []); setFeedback(feedbackSummary || null) } catch (requestError) { setError(requestError.message) } finally { if (showLoading) setLoading(false) } }, [canManageKnowledge])
  useEffect(() => { refresh(); const timer = window.setInterval(() => refresh(false), 2500); return () => window.clearInterval(timer) }, [refresh])
  // retrieval_mode is deliberately omitted so the panel always reflects the server's
  // configured mode; pinning it here would make diagnostics disagree with live chat.
  const search = async (event) => { event.preventDefault(); if (!query.trim()) return; setError(''); try { setResults(await readJson('/api/v1/rag/search', { method: 'POST', headers: csrfHeaders(), body: JSON.stringify({ query, top_k: 4, threshold: 0.35, candidate_multiplier: 4 }) })) } catch (requestError) { setError(requestError.message) } }
  const submitJob = async (path, options) => { setError(''); setMessage(''); try { const response = await readJson(path, options); setJobs((current) => [response.job, ...current.filter((job) => job.id !== response.job.id)]); setMessage(t('jobQueued')) } catch (requestError) { setError(requestError.message) } }
  const runEvaluation = async () => { await submitJob('/api/v1/rag/evaluations/run', { method: 'POST', headers: csrfHeaders(), body: JSON.stringify({ top_k: 4, threshold: 0.35, candidate_multiplier: 4, compare_modes: compareModes }) }) }
  const upload = async (event) => { event.preventDefault(); const file = event.currentTarget.file.files[0]; if (!file) return; const form = new FormData(); form.append('file', file); await submitJob('/api/v1/rag/documents', { method: 'POST', headers: csrfHeaders(), body: form }); event.currentTarget.reset() }
  const rebuild = async () => { await submitJob('/api/v1/rag/rebuild', { method: 'POST', headers: csrfHeaders() }) }
  const removeDocument = async (name) => { if (!window.confirm(`Delete ${name} and rebuild the knowledge index?`)) return; await submitJob(`/api/v1/rag/documents/${encodeURIComponent(name)}`, { method: 'DELETE', headers: csrfHeaders() }) }
  const hasActiveJob = jobs.some((job) => job.status === 'queued' || job.status === 'running')
  return <section className="feature-page"><PageHeader title={t('knowledgeTitle')} description={t('knowledgeDescription')} action={<button className="secondary-button" onClick={() => refresh()}><RefreshCw size={16} />{t('refresh')}</button>} /><ErrorText error={error} />{message && <p className="success-text">{message}</p>}<Loading loading={loading} t={t} />
    <div className="stat-grid">{[[t('status'), status?.status || 'Unknown'], ...(canManageKnowledge ? [[t('documents'), status?.documents?.length || 0]] : []), [t('quality'), quality?.level || 'Unknown'], [t('chunks'), quality?.chunks ?? 0], [t('release'), status?.release?.enabled ? status.release.state : 'Off']].map(([label, value]) => <div className="stat" key={label}><span>{label}</span><strong className="small-stat">{value}</strong></div>)}</div>
    {status?.release?.enabled && <section className="data-panel"><h2>{t('releaseGate')}</h2><p className="report-text">{status.release.reason || status.release.state}</p><JsonPreview value={{ thresholds: status.release.thresholds, report: status.release.report }} empty={t('noEvaluation')} /></section>}
    {canManageKnowledge && <section className="data-panel"><div className="panel-title-row"><h2>{t('jobs')}</h2><span className="job-count">{jobs.length}</span></div>{jobs.length ? <div className="job-list">{jobs.map((job) => <article className={`job-row ${job.status}`} key={job.id}><div><strong>{job.kind.replaceAll('_', ' ')}</strong><span>{t(job.status)} · {job.progress}%</span><p>{job.error || job.message}</p></div><i aria-label={`${job.progress}%`}><b style={{ width: `${job.progress}%` }} /></i></article>)}</div> : <p className="empty-state">{t('noJobs')}</p>}</section>}
    <div className="two-column">{canManageKnowledge && <section className="data-panel"><div className="panel-title-row"><h2>{t('indexedDocuments')}</h2><button className="secondary-button" disabled={hasActiveJob} onClick={rebuild}><RefreshCw size={16} />{t('rebuild')}</button></div><form className="upload-form" onSubmit={upload}><input name="file" type="file" accept=".txt,.md,.markdown,.csv,.json,.pdf,.docx" /><button className="primary-button" disabled={hasActiveJob}><Upload size={16} />{t('upload')}</button></form>{status?.documents?.length ? <div className="document-list">{status.documents.map((document) => <div className="document-row" key={document.name}><div><strong>{document.name}</strong><span>{document.chunks ?? 0} {t('chunks')} · {formatBytes(document.size_bytes)}</span></div><button className="danger-icon" title={`${t('delete')} ${document.name}`} disabled={hasActiveJob} onClick={() => removeDocument(document.name)}><Trash2 size={16} /></button></div>)}</div> : <p className="empty-state">{t('noDocuments')}</p>}</section>}<section className="data-panel"><h2>{t('qualityReport')}</h2><JsonPreview value={quality} empty={t('noQualityResult')} />{canManageKnowledge && <><h2>{t('ragFeedback')}</h2><JsonPreview value={feedback} empty={t('noRagFeedback')} /></>}</section></div>
    <div className="two-column"><section className="data-panel"><h2>{t('retrievalCheck')}</h2><form className="inline-form" onSubmit={search}><input value={query} placeholder={t('retrievalPlaceholder')} onChange={(event) => setQuery(event.target.value)} /><button className="primary-button"><Search size={16} />{t('search')}</button></form>{results && <JsonPreview value={results.results || results} empty={t('noMatchingChunks')} />}</section>{canManageKnowledge && <section className="data-panel"><div className="panel-title-row"><h2>{t('ragEvaluation')}</h2><div className="panel-title-actions"><label className="inline-checkbox"><input type="checkbox" checked={compareModes} onChange={(event) => setCompareModes(event.target.checked)} />{t('compareRetrievalModes')}</label><button className="secondary-button" disabled={hasActiveJob} onClick={runEvaluation}><Sparkles size={16} />{t('runEvaluation')}</button></div></div><JsonPreview value={latest} empty={t('noEvaluation')} /></section>}</div>
  </section>
}

export function OperationsPage({ t }) {
  const [days, setDays] = useState(7)
  const [dashboard, setDashboard] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const refresh = useCallback(async (showLoading = true) => {
    if (showLoading) setLoading(true)
    setError('')
    try {
      setDashboard(await readJson(`/api/v1/operations/dashboard?days=${days}`))
    } catch (requestError) {
      setError(requestError.status === 403 ? t('accessDenied') : requestError.message)
    } finally {
      if (showLoading) setLoading(false)
    }
  }, [days, t])
  useEffect(() => {
    refresh()
    const timer = window.setInterval(() => refresh(false), 30_000)
    return () => window.clearInterval(timer)
  }, [refresh])
  const activeAlerts = dashboard?.alerts?.filter((alert) => alert.status === 'active') || []
  const stats = dashboard ? [
    [t('selectedWindow'), `${dashboard.window_days}d`],
    [t('requests'), dashboard.http.requests || 0],
    [t('failureRate'), `${Number(dashboard.http.failure_rate || 0).toFixed(1)}%`],
    [t('averageLatency'), `${Math.round(dashboard.http.average_duration_ms || 0)} ms`],
    [t('activeAlerts'), activeAlerts.length],
  ] : []
  return <section className="feature-page"><PageHeader title={t('operationsTitle')} description={t('operationsDescription')} action={<div className="operations-actions"><select aria-label={t('selectedWindow')} value={days} onChange={(event) => setDays(Number(event.target.value))}><option value="1">1d</option><option value="7">7d</option><option value="30">30d</option><option value="90">90d</option></select><button className="secondary-button" onClick={() => refresh()}><RefreshCw size={16} />{t('refresh')}</button></div>} /><ErrorText error={error} /><Loading loading={loading} t={t} />
    {dashboard && <><div className="stat-grid">{stats.map(([label, value]) => <div className="stat" key={label}><span>{label}</span><strong className="small-stat">{value}</strong></div>)}</div><section className="data-panel"><h2>{t('alerts')}</h2>{dashboard.alerts.length ? <div className="alert-list">{dashboard.alerts.map((alert) => <article className={`alert-row ${alert.severity} ${alert.status}`} key={alert.fingerprint}><div><strong>{alert.message}</strong><span>{alert.severity} · {alert.status} · {alert.last_seen_at || dashboard.generated_at}</span></div></article>)}</div> : <p className="empty-state">{t('noAlerts')}</p>}</section><div className="two-column"><section className="data-panel"><h2>{t('providerFailures')}</h2><JsonPreview value={dashboard.provider_failures} empty={t('noProviderFailures')} /></section><section className="data-panel"><h2>{t('jobFailures')}</h2>{dashboard.jobs.recent_failures?.length ? <JsonPreview value={dashboard.jobs.recent_failures} empty={t('noJobFailures')} /> : <p className="empty-state">{t('noJobFailures')}</p>}<JsonPreview value={dashboard.jobs.counts} empty={t('noJobFailures')} /></section></div><section className="data-panel"><h2>{t('requests')}</h2><JsonPreview value={{ top_paths: dashboard.http.top_paths, statuses: dashboard.http.statuses }} empty={t('noQualityResult')} /></section></>}
  </section>
}

export function PrivacyPage({ onDeleted, t }) {
  const [summary, setSummary] = useState(null)
  const [confirmation, setConfirmation] = useState('')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const refresh = async () => { setError(''); try { setSummary(await readJson('/api/v1/privacy')) } catch (requestError) { setError(requestError.message) } }
  useEffect(() => { refresh() }, [])
  const exportData = async () => { setError(''); try { const response = await apiFetch('/api/v1/export'); const payload = await response.json(); const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })); const link = document.createElement('a'); link.href = url; link.download = `serenova-export-${payload.user_id}.json`; link.click(); URL.revokeObjectURL(url); setMessage('Your export has been downloaded.') } catch (requestError) { setError(requestError.message) } }
  const deleteData = async () => { if (confirmation !== 'DELETE') { setError('Type DELETE to confirm.'); return } if (!window.confirm('This permanently removes all of your user data. Continue?')) return; try { await readJson('/api/v1/privacy/data', { method: 'DELETE', headers: csrfHeaders(), body: JSON.stringify({ confirmation }) }); onDeleted() } catch (requestError) { setError(requestError.message) } }
  return <section className="feature-page"><PageHeader title={t('privacyTitle')} description={t('privacyDescription')} action={<button className="secondary-button" onClick={refresh}><RefreshCw size={16} />{t('refresh')}</button>} /><ErrorText error={error} />{message && <p className="success-text">{message}</p>}
    <div className="stat-grid">{summary && Object.entries(summary).filter(([key]) => key.endsWith('_count')).map(([key, value]) => <div className="stat" key={key}><span>{key.replace('_count', '').replaceAll('_', ' ')}</span><strong>{value}</strong></div>)}</div>
    <div className="two-column"><section className="data-panel"><h2>{t('exportTitle')}</h2><p>{t('exportDescription')}</p><button className="primary-button" onClick={exportData}><Download size={16} />{t('downloadExport')}</button></section><section className="data-panel danger-panel"><h2><ShieldAlert size={18} />{t('deleteTitle')}</h2><p>{t('deleteDescription')}</p><input value={confirmation} placeholder={t('deletePlaceholder')} onChange={(event) => setConfirmation(event.target.value)} /><button className="danger-button" onClick={deleteData}><Trash2 size={16} />{t('deleteMyData')}</button></section></div>
  </section>
}

function MoodChart({ points }) {
  const valid = points.map((point, index) => ({ ...point, index })).filter((point) => point.intensity !== null && point.intensity !== undefined)
  const x = (index) => 32 + index * (256 / Math.max(points.length - 1, 1))
  const y = (value) => 18 + (5 - Number(value)) * 36
  const path = valid.map((point, index) => `${index ? 'L' : 'M'} ${x(point.index)} ${y(point.intensity)}`).join(' ')
  return <div className="mood-chart"><svg viewBox="0 0 320 210" role="img" aria-label="Weekly Mood trend"><g className="chart-grid">{[1, 2, 3, 4, 5].map((value) => <g key={value}><line x1="32" x2="292" y1={y(value)} y2={y(value)} /><text x="12" y={y(value) + 4}>{value}</text></g>)}</g>{path && <path className="trend-line" d={path} />}{valid.map((point) => <g key={point.date}><circle className="trend-dot" cx={x(point.index)} cy={y(point.intensity)} r="5" /><title>{`${point.date}: ${point.mood || ''} ${point.intensity}/5`}</title></g>)}{points.map((point, index) => <text className="chart-label" key={point.date || index} x={x(index)} y="198" textAnchor="middle">{point.label || point.date?.slice(5)}</text>)}</svg></div>
}

function EmotionMemoryList({ items, t }) {
  if (!items.length) return <p className="empty-state">{t('noEmotionHistory')}</p>
  return <div className="record-list">{[...items].reverse().map((item, index) => <article className="record-row" key={`${item.time || 'emotion'}-${index}`}><div><strong>{item.label || t('emotion')}</strong><span>{item.time ? String(item.time).slice(0, 19).replace('T', ' ') : ''}{item.score !== undefined ? ` · ${Number(item.score).toFixed(2)}` : ''}</span></div></article>)}</div>
}

function JsonPreview({ value, empty }) { return value && (Array.isArray(value) ? value.length : Object.keys(value).length) ? <pre className="json-preview">{json(value)}</pre> : <p className="empty-state">{empty}</p> }
function formatBytes(value) { if (!value) return '0 B'; if (value < 1024) return `${value} B`; return `${(value / 1024).toFixed(1)} KB` }
