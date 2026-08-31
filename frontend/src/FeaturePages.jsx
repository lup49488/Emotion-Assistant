import { useCallback, useEffect, useRef, useState } from 'react'
import { Check, Download, Pencil, Plus, RefreshCw, Search, ShieldAlert, Sparkles, Trash2, Undo2, Upload, X } from 'lucide-react'
import { API_BASE_URL, apiFetch, csrfHeaders, readJson } from './api'

const json = (value) => JSON.stringify(value, null, 2)

function PageHeader({ title, description, action }) {
  return <header className="feature-header"><div><h1>{title}</h1><p>{description}</p></div>{action}</header>
}

function ErrorText({ error }) { return error ? <p className="inline-error" role="alert">{error}</p> : null }

function Loading({ loading, t }) { return loading ? <span className="loading-label"><RefreshCw size={14} className="spin" />{t('loading')}</span> : null }

function localizedMemoryReason(reason, locale) {
  if (locale !== 'en' || typeof reason !== 'string') return reason
  const match = reason.match(/^情绪模型置信度\s+([\d.]+)\s+达到情绪记录阈值\s+([\d.]+)$/)
  return match ? `Emotion model confidence ${match[1]} met the emotion-memory threshold ${match[2]}.` : reason
}

export function MemoryPage({ t, locale }) {
  const [snapshot, setSnapshot] = useState(null)
  const [quality, setQuality] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [editingIndex, setEditingIndex] = useState(null)
  const [memoryDraft, setMemoryDraft] = useState('')
  const [editingInterestIndex, setEditingInterestIndex] = useState(null)
  const [interestDraft, setInterestDraft] = useState('')
  const [memorySaveMode, setMemorySaveMode] = useState('confirm')
  const refresh = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const [memory, report, preference] = await Promise.all([readJson('/api/v1/memory'), readJson(`/api/v1/memory/quality?locale=${encodeURIComponent(locale)}`), readJson('/api/v1/memory/preference')])
      setSnapshot(memory); setQuality(report.report); setMemorySaveMode(preference.mode)
    } catch (requestError) { setError(requestError.message) } finally { setLoading(false) }
  }, [locale])
  useEffect(() => { refresh() }, [refresh])
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
  const saveInterests = async (items) => {
    setError('')
    try {
      const updated = await readJson('/api/v1/memory/interests', { method: 'PUT', headers: csrfHeaders(), body: JSON.stringify({ items }) })
      setSnapshot(updated)
      setEditingInterestIndex(null)
    } catch (requestError) { setError(requestError.message) }
  }
  const startInterestEdit = (index) => { setEditingInterestIndex(index); setInterestDraft(snapshot.interest_memory[index]?.text || '') }
  const saveInterestEdit = async (index) => {
    const text = interestDraft.trim()
    if (!text) { setError(t('memoryRequired')); return }
    await saveInterests(snapshot.interest_memory.map((item, itemIndex) => itemIndex === index ? { ...item, text } : item))
  }
  const addInterest = () => {
    const next = [...snapshot.interest_memory, { text: '', kind: 'manual' }]
    setSnapshot((current) => ({ ...current, interest_memory: next }))
    setEditingInterestIndex(next.length - 1)
    setInterestDraft('')
  }
  const removeInterest = async (index) => { await saveInterests(snapshot.interest_memory.filter((_, itemIndex) => itemIndex !== index)) }
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
  const editPending = async (pendingId, text) => {
    setError('')
    try {
      const updated = await readJson(`/api/v1/memory/pending/${encodeURIComponent(pendingId)}`, { method: 'PUT', headers: csrfHeaders(), body: JSON.stringify({ text }) })
      setSnapshot(updated)
    } catch (requestError) { setError(requestError.message); throw requestError }
  }
  const pendingMemory = snapshot?.pending_memory || []
  const summary = snapshot ? [
    [t('pendingMemories'), pendingMemory.length], [t('longTerm'), snapshot.long_memory.length], [t('interests'), snapshot.interest_memory.length], [t('stableProfile'), snapshot.stable_profile.length],
  ] : []
  return <section className="feature-page memory-page v2-feature-page">
    <PageHeader title={t('memoryWorkspaceTitle')} description={t('memoryWorkspaceDescription')} action={<button className="secondary-button" onClick={refresh}><RefreshCw size={16} />{t('refresh')}</button>} />
    <ErrorText error={error} /><Loading loading={loading} t={t} />
    {snapshot && <>
      <div className="stat-grid">{summary.map(([label, value]) => <div className="stat" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
      <div className="memory-v2-grid"><div className="memory-v2-primary"><section className="data-panel"><div className="panel-title-row"><h2>{t('pendingMemories')}</h2></div><PendingMemoryList items={pendingMemory} onEdit={editPending} onResolve={resolvePending} t={t} /></section><section className="data-panel"><div className="panel-title-row"><h2>{t('longTermMemory')}</h2><button className="icon-button" title={t('addMemory')} onClick={addMemory}><Plus size={17} /></button></div><p className="data-panel-copy">{t('longTermDescription')}</p>{snapshot.long_memory.length === 0 ? <p className="empty-state">{t('noLongTerm')}</p> : <div className="memory-list">{snapshot.long_memory.map((item, index) => <article className="memory-row" key={`${item.time || 'manual'}-${index}`}>{editingIndex === index ? <><textarea value={memoryDraft} rows="3" aria-label={t('memoryText')} onChange={(event) => setMemoryDraft(event.target.value)} /><div className="memory-row-actions"><button className="primary-button" onClick={() => saveEdit(index)}>{t('saveMemory')}</button><button className="secondary-button" onClick={() => { setEditingIndex(null); if (!item.text) setSnapshot((current) => ({ ...current, long_memory: current.long_memory.filter((_, itemIndex) => itemIndex !== index) })) }}><X size={15} />{t('cancel')}</button></div></> : <><div><p>{item.text}</p><span>{item.kind || 'memory'} {item.time ? `· ${String(item.time).slice(0, 10)}` : ''}</span></div><div className="memory-row-actions"><button className="icon-button" title={t('editMemory')} onClick={() => startEdit(index)}><Pencil size={15} /></button><button className="danger-icon" title={t('deleteMemory')} onClick={() => removeMemory(index)}><Trash2 size={16} /></button></div></>}</article>)}</div>}</section><section className="data-panel"><div className="panel-title-row"><h2>{t('interests')}</h2><button className="icon-button" title={t('addMemory')} onClick={addInterest}><Plus size={17} /></button></div><p className="data-panel-copy">{t('interestDescription')}</p>{snapshot.interest_memory.length === 0 ? <p className="empty-state">{t('noInterests')}</p> : <div className="memory-list">{snapshot.interest_memory.map((item, index) => <article className="memory-row" key={`${item.time || 'interest'}-${index}`}>{editingInterestIndex === index ? <><textarea value={interestDraft} rows="3" aria-label={t('memoryText')} onChange={(event) => setMemoryDraft(event.target.value)} /><div className="memory-row-actions"><button className="primary-button" onClick={() => saveInterestEdit(index)}>{t('saveMemory')}</button><button className="secondary-button" onClick={() => { setEditingInterestIndex(null); if (!item.text) setSnapshot((current) => ({ ...current, interest_memory: current.interest_memory.filter((_, itemIndex) => itemIndex !== index) })) }}><X size={15} />{t('cancel')}</button></div></> : <><div><p>{item.text || String(item)}</p><span>{item.kind || 'interest'} {item.time ? `· ${String(item.time).slice(0, 10)}` : ''}</span></div><div className="memory-row-actions"><button className="icon-button" title={t('editMemory')} onClick={() => startInterestEdit(index)}><Pencil size={15} /></button><button className="danger-icon" title={t('deleteMemory')} onClick={() => removeInterest(index)}><Trash2 size={16} /></button></div></>}</article>)}</div>}</section></div><aside className="memory-v2-sidebar"><section className="data-panel"><h2>{t('memorySaveMode')}</h2><div className="memory-policy-options">{[['auto', t('memorySaveAuto')], ['confirm', t('memorySaveConfirm')], ['off', t('memorySaveOff')]].map(([mode, label]) => <label className={memorySaveMode === mode ? 'selected' : ''} key={mode}><input type="radio" name="memory-save-mode" value={mode} checked={memorySaveMode === mode} onChange={() => updateMemorySaveMode(mode)} /><span><b>{label}</b><small>{t(`memorySaveMode_${mode}`)}</small></span></label>)}</div><p className="report-text memory-quality-note">{quality || t('noQualityResult')}</p></section><section className="data-panel"><h2>{t('recentEvents')}</h2><MemoryAuditList events={snapshot.memory_events} limit={3} onUndo={undoAudit} t={t} locale={locale} />{snapshot.memory_events.length > 3 && <details className="memory-event-details"><summary>{t('viewAll')}</summary><div className="memory-event-scroll"><MemoryAuditList events={snapshot.memory_events.slice(0, -3)} onUndo={undoAudit} t={t} locale={locale} /></div></details>}</section></aside></div>
    </>}
  </section>
}

function PendingMemoryList({ items, onEdit, onResolve, t }) {
  const [editingId, setEditingId] = useState(null)
  const [draft, setDraft] = useState('')
  const save = async (id) => { if (!draft.trim()) return; await onEdit(id, draft); setEditingId(null); setDraft('') }
  if (items.length === 0) return <p className="empty-state">{t('noPendingMemories')}</p>
  return <div className="memory-list">{items.map((item) => {
    const candidate = item.candidate || {}
    const editing = editingId === item.id
    return <article className="memory-row pending-memory-row" key={item.id}>{editing ? <><textarea rows="3" aria-label={t('memoryText')} value={draft} onChange={(event) => setDraft(event.target.value)} /><div className="memory-row-actions pending-memory-actions"><button className="primary-button" onClick={() => save(item.id)}><Check size={15} />{t('saveMemory')}</button><button className="secondary-button" onClick={() => { setEditingId(null); setDraft('') }}><X size={15} />{t('cancel')}</button></div></> : <><div><p>{candidate.text || t('memoryTitle')}</p><span>{item.created_at} · {item.reason}</span>{item.source_text && <span>{t('memorySource')}: {item.source_text}</span>}</div><div className="memory-row-actions pending-memory-actions"><button className="primary-button" title={t('confirmMemory')} onClick={() => onResolve(item.id, 'confirm')}><Check size={15} />{t('confirmMemory')}</button><button className="secondary-button" title={t('editMemory')} onClick={() => { setEditingId(item.id); setDraft(candidate.text || '') }}><Pencil size={15} />{t('editMemory')}</button><button className="secondary-button" title={t('discardMemory')} onClick={() => onResolve(item.id, 'discard')}><X size={15} />{t('discardMemory')}</button></div></>}</article>
  })}</div>
}

function MemoryAuditList({ events, limit, onUndo, t, locale }) {
  const sections = { stable: t('stableProfile'), interest: t('interests'), long: t('longTerm'), emotion: t('emotion'), none: t('memoryTitle') }
  const actions = { added: t('memoryEventAdded'), updated: t('memoryEventUpdated'), merged: t('memoryEventMerged'), unchanged: t('memoryEventUnchanged'), skipped: t('memoryEventSkipped'), pending: t('memoryEventPending'), confirmed: t('memoryEventConfirmed'), rejected: t('memoryEventRejected'), reverted: t('memoryEventReverted') }
  const visibleEvents = [...events].reverse().slice(0, limit || 20)
  if (visibleEvents.length === 0) return <p className="empty-state">{t('noEvents')}</p>
  return <div className="memory-list">{visibleEvents.map((event) => <article className="memory-row memory-audit-row" key={event.id}><div><p><strong>{sections[event.section] || event.section} · {actions[event.action] || event.action}</strong></p><p>{event.text || t('memoryTitle')}</p><span>{event.time} · {localizedMemoryReason(event.reason, locale)}</span>{event.source_text && <span>{t('memorySource')}: {event.source_text}</span>}{event.undone_at && <span>{t('memoryEventReverted')}</span>}</div>{event.undoable && <button className="secondary-button" title={t('undoMemory')} onClick={() => onUndo(event.id)}><Undo2 size={15} />{t('undoMemory')}</button>}</article>)}</div>
}

export function MoodPage({ t, onReflect, locale }) {
  return <section className="feature-page mood-page v2-feature-page"><header className="feature-header mood-feature-header"><div><span className="mood-kicker">{t('moodTitle')}</span><h1>{t('moodWorkspaceTitle')}</h1><p>{t('moodWorkspaceDescription')}</p></div></header><MoodCheckinContent t={t} onReflect={onReflect} locale={locale} /></section>
}

function MoodCheckinContent({ t, onReflect, locale }) {
  const [records, setRecords] = useState([])
  const [weekly, setWeekly] = useState(null)
  const emptyForm = { date: null, mood: '', intensity: 3, note: '' }
  const [form, setForm] = useState(emptyForm)
  const [imageFiles, setImageFiles] = useState([])
  const [imagePreviews, setImagePreviews] = useState([])
  const imagePreviewsRef = useRef([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [savedRecord, setSavedRecord] = useState(null)
  const [trendDays, setTrendDays] = useState(30)
  const isEditing = Boolean(form.date)
  const refresh = useCallback(async () => {
    setLoading(true); setError('')
    try { const [all, trend] = await Promise.all([readJson('/api/v1/mood/checkins'), readJson(`/api/v1/mood/weekly?locale=${encodeURIComponent(locale)}`)]); setRecords(all.records); setWeekly(trend) } catch (requestError) { setError(requestError.message) } finally { setLoading(false) }
  }, [locale])
  useEffect(() => { refresh() }, [refresh])
  // Editing a check-in that already has photos: the cap covers stored ones too.
  const existingImageCount = (records.find((record) => record.date === form.date)?.images || []).length
  const chooseImages = (event) => {
    const selected = Array.from(event.target.files || [])
    if (selected.some((file) => !['image/png', 'image/jpeg', 'image/webp', 'image/gif'].includes(file.type) || file.size > 5 * 1024 * 1024) || existingImageCount + imageFiles.length + selected.length > 3) {
      setError(t('moodImageLimit')); event.target.value = ''; return
    }
    setImageFiles((current) => [...current, ...selected])
    const added = selected.map((file) => ({ file, url: URL.createObjectURL(file) }))
    imagePreviewsRef.current = [...imagePreviewsRef.current, ...added]
    setImagePreviews((current) => [...current, ...added])
    event.target.value = ''
  }
  const removeSelectedImage = (file) => {
    const removed = imagePreviewsRef.current.find((item) => item.file === file)
    if (removed) URL.revokeObjectURL(removed.url)
    imagePreviewsRef.current = imagePreviewsRef.current.filter((item) => item.file !== file)
    setImageFiles((current) => current.filter((item) => item !== file))
    setImagePreviews((current) => current.filter((item) => item.file !== file))
  }
  const clearSelectedImages = () => {
    imagePreviewsRef.current.forEach((item) => URL.revokeObjectURL(item.url))
    imagePreviewsRef.current = []
    setImageFiles([])
    setImagePreviews([])
  }
  useEffect(() => { imagePreviewsRef.current = imagePreviews }, [imagePreviews])
  useEffect(() => () => { imagePreviewsRef.current.forEach((item) => URL.revokeObjectURL(item.url)) }, [])
  const uploadImages = async (record) => {
    if (imageFiles.length === 0) return record
    const results = await Promise.allSettled(imageFiles.map(async (file) => {
      const payload = new FormData(); payload.append('file', file)
      return readJson(`/api/v1/mood/checkins/${encodeURIComponent(record.date)}/images`, { method: 'POST', headers: csrfHeaders(), body: payload })
    }))
    const images = results.filter((item) => item.status === 'fulfilled').map((item) => item.value)
    const failed = imageFiles.filter((_, index) => results[index].status === 'rejected')
    // Keep only what still needs sending, so a retry cannot duplicate an upload.
    imageFiles.filter((file) => !failed.includes(file)).forEach(removeSelectedImage)
    if (failed.length) setError(results.find((item) => item.status === 'rejected').reason?.message || t('moodImageLimit'))
    return { ...record, images: [...(record.images || []), ...images] }
  }
  const submit = async (event) => {
    event.preventDefault(); setError('')
    try {
      const payload = { mood: form.mood, intensity: form.intensity, note: form.note, ...(form.date ? { checkin_date: form.date } : {}) }
      const result = await readJson('/api/v1/mood/checkins', { method: 'POST', headers: csrfHeaders(), body: JSON.stringify(payload) })
      const record = await uploadImages(result.record)
      setSavedRecord(record); setForm(emptyForm); await refresh()
    } catch (requestError) { setError(requestError.message) }
  }
  const startEdit = (record) => { setError(''); setSavedRecord(null); clearSelectedImages(); setForm({ date: record.date, mood: record.mood, intensity: record.intensity, note: record.note || '' }) }
  const cancelEdit = () => { setSavedRecord(null); clearSelectedImages(); setForm(emptyForm) }
  const remove = async (date) => { if (!window.confirm(`Delete the Mood Check-in for ${date}?`)) return; try { await apiFetch(`/api/v1/mood/checkins/${date}`, { method: 'DELETE', headers: csrfHeaders() }); if (form.date === date) cancelEdit(); await refresh() } catch (requestError) { setError(requestError.message) } }
  const removeImage = async (record, image) => { try { await apiFetch(`/api/v1/mood/checkins/${encodeURIComponent(record.date)}/images/${encodeURIComponent(image.id)}`, { method: 'DELETE', headers: csrfHeaders() }); await refresh() } catch (requestError) { setError(requestError.message) } }
  const orderedRecords = [...records].sort((left, right) => left.date.localeCompare(right.date))
  const chartPoints = trendDays === 'all' ? orderedRecords : orderedRecords.slice(-trendDays)
  const averageIntensity = chartPoints.length ? (chartPoints.reduce((total, record) => total + Number(record.intensity || 0), 0) / chartPoints.length).toFixed(1) : '—'
  const moodCounts = chartPoints.reduce((counts, record) => ({ ...counts, [record.mood]: (counts[record.mood] || 0) + 1 }), {})
  const commonMood = Object.entries(moodCounts).sort((left, right) => right[1] - left[1])[0]?.[0] || '—'
  const chips = [t('moodChipCalm'), t('moodChipTired'), t('moodChipHopeful'), t('moodChipNervous')]
  return <><div className="mood-page-actions"><button className="secondary-button" onClick={refresh}><RefreshCw size={16} />{t('refresh')}</button></div><ErrorText error={error} />
    <div className="two-column mood-top"><form className="data-panel checkin-form" onSubmit={submit}><h2>{isEditing ? `${t('editingCheckinFor')} ${form.date}` : t('todayCheckin')}</h2><label>{t('moodLabel')}<input required value={form.mood} placeholder={t('moodPlaceholder')} onChange={(event) => setForm((current) => ({ ...current, mood: event.target.value }))} /></label><div className="mood-chip-row">{chips.map((chip) => <button className={form.mood === chip ? 'selected' : ''} type="button" key={chip} onClick={() => setForm((current) => ({ ...current, mood: chip }))}>{chip}</button>)}</div><label>{t('intensity')} <b>{form.intensity}/5</b><div className="mood-intensity-row">{[1, 2, 3, 4, 5].map((level) => <button className={form.intensity === level ? 'selected' : ''} type="button" key={level} onClick={() => setForm((current) => ({ ...current, intensity: level }))} aria-label={`${t('intensity')} ${level}/5`}>{level}</button>)}</div></label><label>{t('note')}<textarea rows="3" value={form.note} placeholder={t('notePlaceholder')} onChange={(event) => setForm((current) => ({ ...current, note: event.target.value }))} /></label><div className="mood-photo-field"><span>{t('moodImages')}</span><div className="mood-image-picker">{imagePreviews.map(({ file, url }) => <figure key={url}><img src={url} alt={file.name} /><button type="button" title={t('removeMoodImage')} onClick={() => removeSelectedImage(file)}><X size={14} /></button></figure>)}<label className="mood-photo-add"><Plus size={26} /><input type="file" accept="image/png,image/jpeg,image/webp,image/gif" aria-label={t('moodImages')} multiple onChange={chooseImages} /></label></div><small>{t('moodImageHelp')}</small></div><div className="checkin-form-actions"><button className="primary-button">{isEditing ? t('updateCheckin') : t('save')}</button><button type="button" className="secondary-button" onClick={() => onReflect({ ...form, date: form.date || new Date().toISOString().slice(0, 10) })}><Sparkles size={16} />{t('talkAboutCheckin')}</button>{isEditing && <button type="button" className="secondary-button" onClick={cancelEdit}><X size={16} />{t('cancelEdit')}</button>}</div>{savedRecord && <div className="mood-reflection-callout" role="status"><span>{t('moodReflectionReady')}</span><button type="button" className="secondary-button" onClick={() => onReflect(savedRecord)}><Sparkles size={16} />{t('talkAboutCheckin')}</button></div>}</form><section className="data-panel trend-panel"><div className="mood-trend-title"><h2>{t('recentTrend')}</h2><div className="mood-period-tabs">{[[30, t('thirtyDays')], [7, t('sevenDays')], ['all', t('all')]].map(([value, label]) => <button type="button" className={trendDays === value ? 'selected' : ''} key={value} onClick={() => setTrendDays(value)}>{label}</button>)}</div></div><MoodChart points={chartPoints} showPoints={trendDays !== 30} /><div className="mood-summary-strip"><div><span>{t('averageIntensity')}</span><strong>{averageIntensity}</strong></div><div><span>{t('recordCount')}</span><strong>{chartPoints.length}</strong></div><div><span>{t('mostFrequentMood')}</span><strong>{commonMood}</strong></div></div><section className="mood-observation"><h3>{t('moodObservation')}</h3><p>{weekly?.summary || t('noWeeklySummary')}</p><button type="button" className="secondary-button" onClick={() => onReflect(chartPoints.at(-1) || { date: new Date().toISOString().slice(0, 10), mood: form.mood, intensity: form.intensity, note: form.note })}><Sparkles size={16} />{t('talkAboutTrend')}</button></section></section></div>
    <section className="data-panel"><h2>{t('recentCheckins')}</h2><Loading loading={loading} t={t} />{records.length === 0 && !loading ? <p className="empty-state">{t('noCheckins')}</p> : <div className="record-list">{[...records].reverse().map((record) => <article className={`record-row ${form.date === record.date ? 'editing' : ''}`} key={record.date}><div><strong>{record.mood}</strong><span>{record.date} · {record.intensity}/5</span>{record.note && <p className="record-note">{record.note}</p>}<MoodImageGrid record={record} onRemove={removeImage} t={t} /></div><div className="record-actions"><button className="icon-button" title={t('talkAboutCheckin')} onClick={() => onReflect(record)}><Sparkles size={16} /></button><button className="icon-button" title={`${t('editCheckin')} ${record.date}`} onClick={() => startEdit(record)}><Pencil size={16} /></button><button className="danger-icon" title={`${t('delete')} ${record.date}`} onClick={() => remove(record.date)}><Trash2 size={16} /></button></div></article>)}</div>}</section>
  </>
}

function MoodImageGrid({ record, onRemove, t }) {
  const images = record.images || []
  if (images.length === 0) return null
  return <div className="mood-image-grid">{images.map((image) => <figure key={image.id}><img src={`${API_BASE_URL}/api/v1/mood/checkins/${encodeURIComponent(record.date)}/images/${encodeURIComponent(image.id)}`} alt={t('moodImageAlt')} loading="lazy" /><button type="button" className="danger-icon" title={t('removeMoodImage')} onClick={() => onRemove(record, image)}><Trash2 size={14} /></button></figure>)}</div>
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
  const summary = [[t('indexStatus'), quality?.level || t('unknown')], [t('documents'), quality?.documents ?? 0], [t('chunks'), quality?.chunks ?? 0], [t('releaseGate'), status?.release?.enabled ? status.release.state : t('notEnabled')]]
  const matches = results?.results || []
  return <section className="feature-page v2-feature-page knowledge-page"><header className="feature-header"><div><span className="knowledge-kicker">KNOWLEDGE &amp; RAG</span><h1>{t('knowledgeTitle')}</h1><p>{t('knowledgeWorkspaceDescription')}</p></div><button className="secondary-button" onClick={() => refresh()}><RefreshCw size={16} />{t('refresh')}</button></header><ErrorText error={error} />{message && <p className="success-text">{message}</p>}<Loading loading={loading} t={t} />
    <div className="stat-grid knowledge-stat-grid">{summary.map(([label, value]) => <div className="stat" key={label}><span>{label}</span><strong className="small-stat">{value}</strong></div>)}</div>
    <section className="knowledge-search-section"><h2>{t('retrievalCheck')}</h2><form className="knowledge-search-form" onSubmit={search}><input value={query} placeholder={t('retrievalPlaceholder')} onChange={(event) => setQuery(event.target.value)} /><button className="secondary-button"><Search size={16} />{t('retrievalCheck')}</button></form>{results && <div className="knowledge-results">{matches.length ? matches.map((result, index) => <article className="knowledge-result-card" key={`${result.source || 'result'}-${result.chunk_index ?? index}`}><div><strong>{result.source || t('matchingSource')} {Number.isInteger(result.chunk_index) ? `· #${result.chunk_index + 1}` : ''}</strong><p>{result.excerpt || result.text || t('noMatchingChunks')}</p></div>{typeof result.score === 'number' && <b>{result.score.toFixed(2)}</b>}</article>) : <p className="empty-state">{results.scope_reason || results.reason || t('noMatchingChunks')}</p>}</div>}</section>
    <section className="knowledge-documents"><div className="panel-title-row"><h2>{t('indexedDocuments')}</h2>{canManageKnowledge && <div className="panel-title-actions"><form className="knowledge-upload-form" onSubmit={upload}><label className="secondary-button"><Upload size={16} />{t('upload')}<input name="file" type="file" accept=".txt,.md,.markdown,.csv,.json,.pdf,.docx" onChange={(event) => { if (event.currentTarget.files?.length) event.currentTarget.form?.requestSubmit() }} /></label></form><button className="secondary-button" disabled={hasActiveJob} onClick={rebuild}><RefreshCw size={16} />{t('rebuild')}</button></div>}</div>{canManageKnowledge ? <div className="knowledge-document-table"><div className="knowledge-document-head"><span>{t('documentName')}</span><span>{t('chunks')}</span><span>{t('updatedAt')}</span><span /></div>{status?.documents?.length ? status.documents.map((document) => <div className="knowledge-document-row" key={document.name}><strong>{document.name}</strong><span>{document.chunks ?? 0}</span><span>{document.modified_at ? String(document.modified_at).slice(0, 10) : formatBytes(document.size_bytes)}</span><button className="danger-icon" title={`${t('delete')} ${document.name}`} disabled={hasActiveJob} onClick={() => removeDocument(document.name)}><Trash2 size={16} /></button></div>) : <p className="empty-state">{t('noDocuments')}</p>}</div> : <p className="knowledge-access-note">{t('knowledgeDocumentAccessNote')}</p>}</section>
    {canManageKnowledge && <details className="knowledge-details"><summary>{t('knowledgeDetails')}</summary><div className="knowledge-details-content">{status?.release?.enabled && <section><h3>{t('releaseGate')}</h3><p className="report-text">{status.release.reason || status.release.state}</p></section>}<section><h3>{t('qualityReport')}</h3><JsonPreview value={quality} empty={t('noQualityResult')} /></section><section><h3>{t('ragFeedback')}</h3><JsonPreview value={feedback} empty={t('noRagFeedback')} /></section><section><div className="panel-title-row"><h3>{t('ragEvaluation')}</h3><div className="panel-title-actions"><label className="inline-checkbox"><input type="checkbox" checked={compareModes} onChange={(event) => setCompareModes(event.target.checked)} />{t('compareRetrievalModes')}</label><button className="secondary-button" disabled={hasActiveJob} onClick={runEvaluation}><Sparkles size={16} />{t('runEvaluation')}</button></div></div><JsonPreview value={latest} empty={t('noEvaluation')} /></section><section><h3>{t('jobs')}</h3>{jobs.length ? <div className="job-list">{jobs.map((job) => <article className={`job-row ${job.status}`} key={job.id}><div><strong>{job.kind.replaceAll('_', ' ')}</strong><span>{t(job.status)} · {job.progress}%</span><p>{job.error || job.message}</p></div><i aria-label={`${job.progress}%`}><b style={{ width: `${job.progress}%` }} /></i></article>)}</div> : <p className="empty-state">{t('noJobs')}</p>}</section></div></details>}
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
  const [importFile, setImportFile] = useState(null)
  const [importMode, setImportMode] = useState('merge')
  const [externalFile, setExternalFile] = useState(null)
  const [externalPreview, setExternalPreview] = useState(null)
  const [profileFields, setProfileFields] = useState([])
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const refresh = async () => { setError(''); try { setSummary(await readJson('/api/v1/privacy')) } catch (requestError) { setError(requestError.message) } }
  useEffect(() => { refresh() }, [])
  const exportData = async () => { setError(''); try { const response = await apiFetch('/api/v1/export'); const payload = await response.json(); const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })); const link = document.createElement('a'); link.href = url; link.download = `serenova-export-${payload.user_id}.json`; link.click(); URL.revokeObjectURL(url); setMessage('Your export has been downloaded.') } catch (requestError) { setError(requestError.message) } }
  const importData = async (event) => { event.preventDefault(); if (!importFile) { setError(t('importFileRequired')); return } if (importMode === 'replace' && !window.confirm(t('importReplaceConfirm'))) return; setError(''); try { const form = new FormData(); form.append('file', importFile); const result = await readJson(`/api/v1/import?mode=${importMode}`, { method: 'POST', headers: csrfHeaders(), body: form }); setMessage(t('importComplete').replace('{conversations}', result.conversations).replace('{memories}', result.memories).replace('{mood_checkins}', result.mood_checkins)); setImportFile(null); await refresh() } catch (requestError) { setError(requestError.message) } }
  const previewExternal = async (file) => { if (!file) return; setError(''); setExternalPreview(null); try { const form = new FormData(); form.append('file', file); const preview = await readJson('/api/v1/import/external/preview', { method: 'POST', headers: csrfHeaders(), body: form }); setExternalPreview(preview); setProfileFields([]) } catch (requestError) { setError(requestError.message) } }
  const importExternal = async () => { if (!externalFile || !externalPreview) return; if (importMode === 'replace' && !window.confirm(t('importReplaceConfirm'))) return; setError(''); try { const form = new FormData(); form.append('file', externalFile); form.append('profile_fields', JSON.stringify(profileFields)); const result = await readJson(`/api/v1/import/external?mode=${importMode}`, { method: 'POST', headers: csrfHeaders(), body: form }); setMessage(t('externalImportComplete').replace('{source}', result.source).replace('{conversations}', result.conversations).replace('{memories}', result.memories)); setExternalFile(null); setExternalPreview(null); setProfileFields([]); await refresh() } catch (requestError) { setError(requestError.message) } }
  const deleteData = async () => { if (confirmation !== 'DELETE') { setError('Type DELETE to confirm.'); return } if (!window.confirm('This permanently removes all of your user data. Continue?')) return; try { await readJson('/api/v1/privacy/data', { method: 'DELETE', headers: csrfHeaders(), body: JSON.stringify({ confirmation }) }); onDeleted() } catch (requestError) { setError(requestError.message) } }
  const stats = summary ? [[t('conversations'), summary.conversation_count || 0], [t('memories'), summary.memory_count || 0], [t('mood'), summary.mood_count || 0]] : []
  return <section className="feature-page privacy-page v2-feature-page"><header className="feature-header"><div><span className="privacy-kicker">PRIVACY &amp; EXPORT</span><h1>{t('privacyWorkspaceTitle')}</h1><p>{t('privacyWorkspaceDescription')}</p></div><button className="secondary-button" onClick={refresh}><RefreshCw size={16} />{t('refresh')}</button></header><ErrorText error={error} />{message && <p className="success-text">{message}</p>}
    <div className="stat-grid privacy-stat-grid">{stats.map(([label, value]) => <div className="stat" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
    <section className="privacy-section"><h2>{t('exportTitle')}</h2><p>{t('exportDescription')}</p><button className="secondary-button" onClick={exportData}><Download size={16} />{t('downloadExport')}</button></section>
    <section className="privacy-section"><h2>{t('importTitle')}</h2><p>{t('importDescription')}</p><form className="privacy-import-form" onSubmit={importData}><label className="secondary-button privacy-file-button">{t('chooseFile')}<input type="file" accept="application/json,.json" aria-label={t('importFile')} onChange={(event) => setImportFile(event.target.files?.[0] || null)} /></label><span>{importFile?.name || t('noFileChosen')}</span><select aria-label={t('importMode')} value={importMode} onChange={(event) => setImportMode(event.target.value)}><option value="merge">{t('importMerge')}</option><option value="replace">{t('importReplace')}</option></select><button className="secondary-button" disabled={!importFile}><Upload size={16} />{t('importData')}</button></form></section>
    <section className="privacy-section external-import-section"><h2>{t('externalImportTitle')}</h2><p>{t('externalImportDescription')}</p><label className="secondary-button privacy-file-button">{t('chooseExternalFile')}<input type="file" accept="application/json,.json,application/zip,.zip" aria-label={t('externalImportFile')} onChange={(event) => { const file = event.target.files?.[0] || null; setExternalFile(file); previewExternal(file) }} /></label>{externalFile && <span className="privacy-file-name">{externalFile.name}</span>}{externalPreview && <div className="external-import-preview"><p><b>{t('externalImportDetected')}</b>{externalPreview.source} · {externalPreview.conversations} {t('conversations')} · {externalPreview.messages} {t('messages')}</p>{externalPreview.sample_titles?.length > 0 && <ul>{externalPreview.sample_titles.map((title) => <li key={title}>{title}</li>)}</ul>}{externalPreview.profile_fields?.length > 0 && <div className="external-profile-fields"><b>{t('externalProfilePrompt')}</b>{externalPreview.profile_fields.map((field) => <label key={field.key}><input type="checkbox" checked={profileFields.includes(field.key)} onChange={(event) => setProfileFields((current) => event.target.checked ? [...current, field.key] : current.filter((key) => key !== field.key))} />{field.label}: {field.value}</label>)}</div>}<button className="secondary-button" onClick={importExternal}><Upload size={16} />{t('externalImportAction')}</button></div>}</section>
    <section className="privacy-section danger-panel"><h2><ShieldAlert size={18} />{t('deleteTitle')}</h2><p>{t('deleteDescription')}</p><div className="privacy-delete-row"><input value={confirmation} placeholder={t('deletePlaceholder')} onChange={(event) => setConfirmation(event.target.value)} /><button className="danger-button" onClick={deleteData}><Trash2 size={16} />{t('deleteMyData')}</button></div></section>
  </section>
}

function MoodChart({ points, showPoints = true }) {
  const maximumVisiblePoints = 10
  const step = Math.max(1, Math.ceil((points.length - 1) / Math.max(maximumVisiblePoints - 1, 1)))
  const visiblePoints = points.filter((_, index) => index === 0 || index === points.length - 1 || index % step === 0)
  const valid = visiblePoints.map((point, index) => ({ ...point, index })).filter((point) => point.intensity !== null && point.intensity !== undefined)
  const x = (index) => 32 + index * (256 / Math.max(visiblePoints.length - 1, 1))
  const y = (value) => 18 + (5 - Number(value)) * 36
  const path = valid.map((point, index) => `${index ? 'L' : 'M'} ${x(point.index)} ${y(point.intensity)}`).join(' ')
  const labelIndexes = new Set([0, Math.floor((visiblePoints.length - 1) / 2), visiblePoints.length - 1])
  return <div className="mood-chart"><svg viewBox="0 0 320 210" role="img" aria-label="Weekly Mood trend"><g className="chart-grid">{[1, 2, 3, 4, 5].map((value) => <g key={value}><line x1="32" x2="292" y1={y(value)} y2={y(value)} /><text x="12" y={y(value) + 4}>{value}</text></g>)}</g>{path && <path className="trend-line" d={path} />}{showPoints && valid.map((point) => <g key={point.date}><circle className="trend-dot" cx={x(point.index)} cy={y(point.intensity)} r="3.5" /><title>{`${point.date}: ${point.mood || ''} ${point.intensity}/5`}</title></g>)}{visiblePoints.map((point, index) => labelIndexes.has(index) && <text className="chart-label" key={point.date || index} x={x(index)} y="198" textAnchor={index === 0 ? 'start' : index === visiblePoints.length - 1 ? 'end' : 'middle'}>{point.label || point.date?.slice(5)?.replace('-', '/')}</text>)}</svg></div>
}

function JsonPreview({ value, empty }) { return value && (Array.isArray(value) ? value.length : Object.keys(value).length) ? <pre className="json-preview">{json(value)}</pre> : <p className="empty-state">{empty}</p> }
function formatBytes(value) { if (!value) return '0 B'; if (value < 1024) return `${value} B`; return `${(value / 1024).toFixed(1)} KB` }
