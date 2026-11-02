/**
 * 类型化的 fetch 封装（方案 §6.5 的 src/api/client.ts）。
 * 用 JSDoc 类型代替 TS，保持零构建复杂度。
 */

/** @typedef {{id:number,filepath:string,audio_url:string,description:string,category:string,intensity:number|null,gender:string|null,duration_sec:number,gain_db:number|null,score:number,verified:number,commercial_ok:number,target_use:string,clip_kind:string,license:string,transcript:string|null,language:string|null,speaker_id:string|null,notes:string|null,studio_quality:number,has_background:number,loudness_lufs:number|null}} Clip */

const BASE = ''

async function request(path, options = {}) {
  const res = await fetch(BASE + path, {
    headers: options.body ? { 'Content-Type': 'application/json' } : {},
    ...options,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = await res.json()
      detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
    } catch (_) {
      /* 非 JSON 错误体 */
    }
    throw new Error(`HTTP ${res.status}: ${detail}`)
  }
  if (res.status === 204) return null
  return res.json()
}

const qs = (params) => {
  const u = new URLSearchParams()
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null && v !== '') u.append(k, v)
  }
  const s = u.toString()
  return s ? `?${s}` : ''
}

export const api = {
  health: () => request('/api/health'),
  meta: () => request('/api/meta'),
  stats: () => request('/api/stats'),
  config: () => request('/api/config'),

  /** @param {object} body */
  search: (body) => request('/api/search', { method: 'POST', body: JSON.stringify(body) }),
  categories: (use) => request(`/api/categories${qs({ use })}`),

  clip: (id) => request(`/api/clips/${id}`),
  waveform: (id, points = 400) => request(`/api/clips/${id}/waveform${qs({ points })}`),
  /** @param {Partial<Clip>} patch */
  patchClip: (id, patch) =>
    request(`/api/clips/${id}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  reembed: (id, description) =>
    request(`/api/clips/${id}/reebed`, {
      method: 'POST',
      body: JSON.stringify({ description: description ?? null }),
    }),
  deleteClip: (id, deleteFile = false) =>
    request(`/api/clips/${id}${qs({ delete_file: deleteFile })}`, { method: 'DELETE' }),

  previewImport: (path, kind) => request(`/api/import/preview${qs({ path, kind })}`),
  startImport: (body) => request('/api/import', { method: 'POST', body: JSON.stringify(body) }),
  jobs: () => request('/api/jobs'),
  job: (id) => request(`/api/jobs/${id}`),
  cancelJob: (id) => request(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  reembedAll: (body) =>
    request('/api/reebed', { method: 'POST', body: JSON.stringify(body) }),

  exportClips: (body) => request('/api/export', { method: 'POST', body: JSON.stringify(body) }),
  exports: () => request('/api/exports'),

  voices: () => request('/api/voices'),
}

/** 任务进度 WebSocket（§6.3）；onFrame 收到每一帧，返回 close 函数。 */
export function watchJob(jobId, onFrame, onError) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const url = `${proto}://${location.host}/api/jobs/${jobId}/ws`
  let ws
  let closed = false
  try {
    ws = new WebSocket(url)
  } catch (err) {
    onError && onError(err)
    return () => {}
  }
  ws.onmessage = (ev) => {
    try {
      onFrame(JSON.parse(ev.data))
    } catch (_) {
      /* 忽略坏帧 */
    }
  }
  ws.onerror = (err) => onError && onError(err)
  ws.onclose = () => {
    closed = true
  }
  return () => {
    if (!closed && ws.readyState <= 1) ws.close()
  }
}

export const audioUrl = (clip) => clip.audio_url || `/api/clips/${clip.id}/audio`
