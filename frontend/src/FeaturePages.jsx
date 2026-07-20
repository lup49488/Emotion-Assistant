import { useEffect, useState } from 'react'
import { Download, Pencil, Plus, RefreshCw, Search, ShieldAlert, Sparkles, Trash2, Upload, X } from 'lucide-react'
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
  const refresh = async () => {
    setLoading(true); setError('')
    try {
      const [memory, report] = await Promise.all([readJson('/api/v1/memory'), readJson('/api/v1/memory/quality')])
      setSnapshot(memory); setQuality(report.report)
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
  const summary = snapshot ? [
    [t('history'), snapshot.history.length], [t('emotion'), snapshot.emotion_memory.length], [t('longTerm'), snapshot.long_memory.length], [t('stableProfile'), snapshot.stable_profile.length], [t('interests'), snapshot.interest_memory.length],
  ] : []
  return <section className="feature-page"><PageHeader title={t('memoryTitle')} description={t('memoryDescription')} action={<button className="secondary-button" onClick={refresh}><RefreshCw size={16} />{t('refresh')}</button>} /><ErrorText error={error} /><Loading loading={loading} t={t} />
    {snapshot && <><div className="stat-grid">{summary.map(([label, value]) => <div className="stat" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div><div className="two-column"><section className="data-panel"><h2>{t('memoryQuality')}</h2><p className="report-text">{quality || t('noQualityResult')}</p></section><section className="data-panel"><h2>{t('stableProfile')}</h2><JsonPreview value={snapshot.stable_profile} empty={t('noStable')} /></section></div><section className="data-panel"><div className="panel-title-row"><h2>{t('longTermMemory')}</h2><button className="icon-button" title={t('addMemory')} onClick={addMemory}><Plus size={17} /></button></div><p className="data-panel-copy">{t('longTermDescription')}</p>{snapshot.long_memory.length === 0 ? <p className="empty-state">{t('noLongTerm')}</p> : <div className="memory-list">{snapshot.long_memory.map((item, index) => <article className="memory-row" key={`${item.time || 'manual'}-${index}`}>{editingIndex === index ? <><textarea value={memoryDraft} rows="3" aria-label={t('memoryText')} onChange={(event) => setMemoryDraft(event.target.value)} /><div className="memory-row-actions"><button className="primary-button" onClick={() => saveEdit(index)}>{t('saveMemory')}</button><button className="secondary-button" onClick={() => { setEditingIndex(null); if (!item.text) setSnapshot((current) => ({ ...current, long_memory: current.long_memory.filter((_, itemIndex) => itemIndex !== index) })) }}><X size={15} />{t('cancel')}</button></div></> : <><div><p>{item.text}</p><span>{item.kind || 'memory'} {item.time ? `· ${String(item.time).slice(0, 10)}` : ''}</span></div><div className="memory-row-actions"><button className="icon-button" title={t('editMemory')} onClick={() => startEdit(index)}><Pencil size={15} /></button><button className="danger-icon" title={t('deleteMemory')} onClick={() => removeMemory(index)}><Trash2 size={16} /></button></div></>}</article>)}</div>}</section><section className="data-panel"><h2>{t('recentEvents')}</h2><JsonPreview value={snapshot.memory_events.slice(-12)} empty={t('noEvents')} /></section></>}
  </section>
}

export function MoodPage({ t }) {
  const [records, setRecords] = useState([])
  const [weekly, setWeekly] = useState(null)
  const [form, setForm] = useState({ mood: '', intensity: 3, note: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const refresh = async () => {
    setLoading(true); setError('')
    try { const [all, trend] = await Promise.all([readJson('/api/v1/mood/checkins'), readJson('/api/v1/mood/weekly')]); setRecords(all.records); setWeekly(trend) } catch (requestError) { setError(requestError.message) } finally { setLoading(false) }
  }
  useEffect(() => { refresh() }, [])
  const submit = async (event) => { event.preventDefault(); setError(''); try { await readJson('/api/v1/mood/checkins', { method: 'POST', headers: csrfHeaders(), body: JSON.stringify(form) }); setForm({ mood: '', intensity: 3, note: '' }); await refresh() } catch (requestError) { setError(requestError.message) } }
  const remove = async (date) => { if (!window.confirm(`Delete the Mood Check-in for ${date}?`)) return; try { await apiFetch(`/api/v1/mood/checkins/${date}`, { method: 'DELETE', headers: csrfHeaders() }); await refresh() } catch (requestError) { setError(requestError.message) } }
  return <section className="feature-page"><PageHeader title={t('moodTitle')} description={t('moodDescription')} action={<button className="secondary-button" onClick={refresh}><RefreshCw size={16} />{t('refresh')}</button>} /><ErrorText error={error} />
    <div className="two-column mood-top"><form className="data-panel checkin-form" onSubmit={submit}><h2>{t('todayCheckin')}</h2><label>{t('moodLabel')}<input required value={form.mood} placeholder={t('moodPlaceholder')} onChange={(event) => setForm((current) => ({ ...current, mood: event.target.value }))} /></label><label>{t('intensity')} <b>{form.intensity}/5</b><input type="range" min="1" max="5" value={form.intensity} onChange={(event) => setForm((current) => ({ ...current, intensity: Number(event.target.value) }))} /></label><label>{t('note')}<textarea rows="3" value={form.note} placeholder={t('notePlaceholder')} onChange={(event) => setForm((current) => ({ ...current, note: event.target.value }))} /></label><button className="primary-button">{t('save')}</button></form><section className="data-panel trend-panel"><h2>{t('weeklyTrend')}</h2><MoodChart points={weekly?.points || []} /><p className="report-text">{weekly?.summary || t('noWeeklySummary')}</p><p className="report-text muted-report">{weekly?.analysis}</p></section></div>
    <section className="data-panel"><h2>{t('recentCheckins')}</h2><Loading loading={loading} t={t} />{records.length === 0 && !loading ? <p className="empty-state">{t('noCheckins')}</p> : <div className="record-list">{[...records].reverse().map((record) => <article className="record-row" key={record.date}><div><strong>{record.mood}</strong><span>{record.date} · {record.intensity}/5</span>{record.note && <p>{record.note}</p>}</div><button className="danger-icon" title={`${t('delete')} ${record.date}`} onClick={() => remove(record.date)}><Trash2 size={16} /></button></article>)}</div>}</section>
  </section>
}

export function KnowledgePage({ t }) {
  const [status, setStatus] = useState(null)
  const [quality, setQuality] = useState(null)
  const [latest, setLatest] = useState(null)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const refresh = async () => { setLoading(true); setError(''); try { const [ragStatus, ragQuality, evaluation] = await Promise.all([readJson('/api/v1/rag/status'), readJson('/api/v1/rag/quality'), readJson('/api/v1/rag/evaluations/latest')]); setStatus(ragStatus); setQuality(ragQuality); setLatest(evaluation.report) } catch (requestError) { setError(requestError.message) } finally { setLoading(false) } }
  useEffect(() => { refresh() }, [])
  const search = async (event) => { event.preventDefault(); if (!query.trim()) return; setError(''); try { setResults(await readJson('/api/v1/rag/search', { method: 'POST', headers: csrfHeaders(), body: JSON.stringify({ query, top_k: 4, threshold: 0.35, candidate_multiplier: 4 }) })) } catch (requestError) { setError(requestError.message) } }
  const runEvaluation = async () => { setError(''); try { const response = await readJson('/api/v1/rag/evaluations/run', { method: 'POST', headers: csrfHeaders(), body: JSON.stringify({ top_k: 4, threshold: 0.35, candidate_multiplier: 4 }) }); setLatest(response.report) } catch (requestError) { setError(requestError.message) } }
  const upload = async (event) => { event.preventDefault(); const file = event.currentTarget.file.files[0]; if (!file) return; setBusy(true); setError(''); try { const form = new FormData(); form.append('file', file); await readJson('/api/v1/rag/documents', { method: 'POST', headers: csrfHeaders(), body: form }); event.currentTarget.reset(); await refresh() } catch (requestError) { setError(requestError.message) } finally { setBusy(false) } }
  const rebuild = async () => { setBusy(true); setError(''); try { await readJson('/api/v1/rag/rebuild', { method: 'POST', headers: csrfHeaders() }); await refresh() } catch (requestError) { setError(requestError.message) } finally { setBusy(false) } }
  const removeDocument = async (name) => { if (!window.confirm(`Delete ${name} and rebuild the knowledge index?`)) return; setBusy(true); setError(''); try { await readJson(`/api/v1/rag/documents/${encodeURIComponent(name)}`, { method: 'DELETE', headers: csrfHeaders() }); await refresh() } catch (requestError) { setError(requestError.message) } finally { setBusy(false) } }
  return <section className="feature-page"><PageHeader title={t('knowledgeTitle')} description={t('knowledgeDescription')} action={<button className="secondary-button" onClick={refresh}><RefreshCw size={16} />{t('refresh')}</button>} /><ErrorText error={error} /><Loading loading={loading} t={t} />
    <div className="stat-grid">{[[t('status'), status?.status || 'Unknown'], [t('documents'), status?.documents?.length || 0], [t('quality'), quality?.level || 'Unknown'], [t('chunks'), quality?.chunks ?? 0]].map(([label, value]) => <div className="stat" key={label}><span>{label}</span><strong className="small-stat">{value}</strong></div>)}</div>
    <div className="two-column"><section className="data-panel"><div className="panel-title-row"><h2>{t('indexedDocuments')}</h2><button className="secondary-button" disabled={busy} onClick={rebuild}><RefreshCw size={16} />{t('rebuild')}</button></div><form className="upload-form" onSubmit={upload}><input name="file" type="file" accept=".txt,.md,.markdown,.csv,.json,.pdf,.docx" /><button className="primary-button" disabled={busy}><Upload size={16} />{t('upload')}</button></form>{status?.documents?.length ? <div className="document-list">{status.documents.map((document) => <div className="document-row" key={document.name}><div><strong>{document.name}</strong><span>{document.chunks ?? 0} {t('chunks')} · {formatBytes(document.size_bytes)}</span></div><button className="danger-icon" title={`${t('delete')} ${document.name}`} disabled={busy} onClick={() => removeDocument(document.name)}><Trash2 size={16} /></button></div>)}</div> : <p className="empty-state">{t('noDocuments')}</p>}</section><section className="data-panel"><h2>{t('qualityReport')}</h2><JsonPreview value={quality} empty={t('noQualityResult')} /></section></div>
    <div className="two-column"><section className="data-panel"><h2>{t('retrievalCheck')}</h2><form className="inline-form" onSubmit={search}><input value={query} placeholder={t('retrievalPlaceholder')} onChange={(event) => setQuery(event.target.value)} /><button className="primary-button"><Search size={16} />{t('search')}</button></form>{results && <JsonPreview value={results.results || results} empty={t('noMatchingChunks')} />}</section><section className="data-panel"><div className="panel-title-row"><h2>{t('ragEvaluation')}</h2><button className="secondary-button" onClick={runEvaluation}><Sparkles size={16} />{t('runEvaluation')}</button></div><JsonPreview value={latest} empty={t('noEvaluation')} /></section></div>
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

function JsonPreview({ value, empty }) { return value && (Array.isArray(value) ? value.length : Object.keys(value).length) ? <pre className="json-preview">{json(value)}</pre> : <p className="empty-state">{empty}</p> }
function formatBytes(value) { if (!value) return '0 B'; if (value < 1024) return `${value} B`; return `${(value / 1024).toFixed(1)} KB` }
