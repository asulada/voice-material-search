<script setup>
/**
 * 导入向导 + 进度（方案 §6.3 / §6.5）。
 * 流程：填目录 → 预览（自动识别数据源类型 / 样本 / 条数）→ 选用途与许可 → 后台任务 + WS 进度。
 */
import { computed, onBeforeUnmount, reactive, ref } from 'vue'
import { api, watchJob } from '../api/client'
import ProgressBar from '../components/ProgressBar.vue'

const props = defineProps({
  meta: { type: Object, default: null },
})
const emit = defineEmits(['changed'])

const form = reactive({
  kind: 'csemotions',
  path: '',
  target_use: 'indextts',
  license: 'apache-2.0',
  language: '',
  limit: null,
  clip_kind: null,
  overwrite: true,
  dry_run: false,
  duration: null,
  gender_map_text: '',
})

const preview = ref(null)
const previewError = ref('')
const previewing = ref(false)
const job = ref(null)
const starting = ref(false)
const history = ref([])
let closeWs = null

const durationRule = computed(() => {
  const u = (props.meta?.uses || []).find((x) => x.value === form.target_use)
  return u ? u.duration : form.target_use === 'seedvc' ? [0.3, 5] : [3, 15]
})

const licenseOptions = [
  { value: 'apache-2.0', label: 'Apache-2.0（CSEMOTIONS / EmoGator，可商用）' },
  { value: 'cc-by-nc-sa-4.0', label: 'CC-BY-NC-SA-4.0（speech-utterances，不可商用）' },
  { value: 'cc-by-4.0', label: 'CC-BY-4.0（署名，可商用）' },
  { value: 'unknown', label: '未知（保守处理为不可商用）' },
]

const kindOptions = [
  { value: 'csemotions', label: 'CSEMOTIONS 目录（<说话人>/<情绪>/*.wav）' },
  { value: 'emogator', label: 'EmoGator 目录（<类别>/*.mp3）' },
  { value: 'flat', label: '自定义目录（需 manifest.json）' },
]

const commercialOk = computed(() => {
  const s = form.license.toLowerCase()
  return !(s.includes('nc') || s.includes('noncommercial') || s.includes('non-commercial'))
})

const canStart = computed(() => form.path.trim() && !starting.value)

function parseGenderMap() {
  const out = {}
  for (const line of form.gender_map_text.split('\n')) {
    const t = line.trim()
    if (!t) continue
    const [k, v] = t.split(/[:=,\s]+/)
    if (k && v) out[k.trim()] = v.trim().toLowerCase()
  }
  return Object.keys(out).length ? out : null
}

async function runPreview() {
  previewError.value = ''
  preview.value = null
  previewing.value = true
  try {
    // 不传 kind：让后端按目录形状自动识别（否则会用表单里的旧值覆盖猜测）
    preview.value = await api.previewImport(form.path.trim())
    // 自动识别结果回填到表单，用户仍可改
    if (preview.value.detected_kind) form.kind = preview.value.detected_kind
    if (preview.value.suggested_target_use) form.target_use = preview.value.suggested_target_use
  } catch (err) {
    previewError.value = err.message
  } finally {
    previewing.value = false
  }
}

async function start() {
  starting.value = true
  try {
    const body = {
      kind: form.kind,
      path: form.path.trim(),
      target_use: form.target_use,
      license: form.license,
      language: form.language || '',
      limit: form.limit || null,
      clip_kind: form.clip_kind || null,
      overwrite: form.overwrite,
      dry_run: form.dry_run,
      gender_map: parseGenderMap(),
      duration: form.duration,
    }
    const { job_id } = await api.startImport(body)
    watchJobId(job_id)
  } catch (err) {
    previewError.value = err.message
  } finally {
    starting.value = false
  }
}

function watchJobId(jobId) {
  closeWs?.()
  job.value = { id: jobId, status: 'running', done: 0, total: 0, message: '排队中…' }
  closeWs = watchJob(
    jobId,
    (frame) => {
      if (frame.error) {
        job.value = { ...job.value, status: 'failed', message: frame.error }
        return
      }
      job.value = frame
      if (['done', 'failed', 'cancelled'].includes(frame.status)) {
        emit('changed')
        loadHistory()
      }
    },
    () => {
      // WS 不可用时退化为轮询
      pollJob(jobId)
    },
  )
}

async function pollJob(jobId) {
  for (let i = 0; i < 3000; i++) {
    try {
      const snap = await api.job(jobId)
      job.value = snap
      if (['done', 'failed', 'cancelled'].includes(snap.status)) {
        emit('changed')
        loadHistory()
        return
      }
    } catch (_) {
      return
    }
    await new Promise((r) => setTimeout(r, 500))
  }
}

async function cancel() {
  if (!job.value?.id) return
  try {
    await api.cancelJob(job.value.id)
  } catch (err) {
    previewError.value = err.message
  }
}

async function loadHistory() {
  try {
    history.value = (await api.jobs()).jobs.slice(0, 8)
  } catch (_) {
    history.value = []
  }
}

const summary = computed(() => {
  const r = job.value?.result
  if (!r) return null
  const rows = [
    ['入库', r.imported],
    ['太短', r.skipped_short],
    ['太长', r.skipped_long],
    ['静音', r.skipped_silent],
    ['已存在', r.duplicated],
  ]
  if (r.error_count) rows.push(['错误', r.error_count])
  return rows
})

loadHistory()
onBeforeUnmount(() => closeWs?.())
</script>

<template>
  <div class="import">
    <section class="card">
      <h2>导入素材</h2>
      <p class="note">
        先把数据集下载到本地，再把目录填在这里。导入会：裁掉首尾数字静音 →
        写进 <code>voice_library/audio/</code> → 生成中文描述 → 建向量 → 入库。
      </p>

      <div class="grid">
        <label class="field wide">
          <span>数据源目录</span>
          <input v-model="form.kind" type="hidden" />
          <input
            v-model="form.path"
            placeholder="例如 D:\datasets\CSEMOTIONS"
            @keydown.enter="runPreview"
          />
        </label>

        <label class="field">
          <span>目录类型</span>
          <select v-model="form.kind">
            <option v-for="k in kindOptions" :key="k.value" :value="k.value">{{ k.label }}</option>
          </select>
        </label>

        <label class="field">
          <span>用于线路</span>
          <select v-model="form.target_use">
            <option value="indextts">IndexTTS（说话片段）</option>
            <option value="seedvc">SeedVC（副语言）</option>
          </select>
        </label>

        <label class="field">
          <span>许可</span>
          <select v-model="form.license">
            <option v-for="l in licenseOptions" :key="l.value" :value="l.value">{{ l.label }}</option>
          </select>
        </label>

        <label class="field">
          <span>语言</span>
          <input v-model="form.language" placeholder="zh / en / 留空" />
        </label>

        <label class="field">
          <span>只导入前 N 条（试跑）</span>
          <input v-model.number="form.limit" type="number" min="1" placeholder="留空 = 全部" />
        </label>
      </div>

      <div class="row">
        <button class="ghost" :disabled="!form.path.trim() || previewing" @click="runPreview">
          {{ previewing ? '识别中…' : '预览目录' }}
        </button>
        <span class="rule">时长筛选：{{ durationRule[0] }}–{{ durationRule[1] }} 秒（来自线路规则）</span>
        <span class="pill" :class="commercialOk ? 'ok' : 'warn'">
          {{ commercialOk ? '可商用' : '不可商用' }}
        </span>
      </div>

      <details class="advanced">
        <summary>高级选项</summary>
        <div class="grid">
          <label class="field wide">
            <span>说话人性别映射（每行 S03: male）</span>
            <textarea v-model="form.gender_map_text" rows="3" placeholder="S03: male&#10;S07: female" />
          </label>
          <label class="field">
            <span>强制素材类型</span>
            <select v-model="form.clip_kind">
              <option :value="null">自动（按线路）</option>
              <option value="speech">speech</option>
              <option value="vocal_burst">vocal_burst</option>
            </select>
          </label>
          <label class="field check">
            <input v-model="form.dry_run" type="checkbox" />
            <span>只扫描不写盘（统计用）</span>
          </label>
          <label class="field check">
            <input v-model="form.overwrite" type="checkbox" />
            <span>覆盖已存在的同路径素材</span>
          </label>
        </div>
      </details>

      <p v-if="previewError" class="banner error">{{ previewError }}</p>

      <div v-if="preview" class="preview">
        <p>
          识别为 <b>{{ preview.detected_kind }}</b>，共 <b>{{ preview.count }}</b> 条素材。
          建议用于 <b>{{ preview.suggested_target_use === 'seedvc' ? 'SeedVC' : 'IndexTTS' }}</b>。
        </p>
        <table>
          <thead>
            <tr><th>文件</th><th>类别</th><th>说话人</th><th>性别</th></tr>
          </thead>
          <tbody>
            <tr v-for="(s, i) in preview.samples.slice(0, 8)" :key="i">
              <td class="mono">{{ s.relpath }}</td>
              <td>{{ s.category }}</td>
              <td>{{ s.speaker_id || '—' }}</td>
              <td>{{ s.gender || '—' }}</td>
            </tr>
          </tbody>
        </table>
        <p v-if="preview.count > 8" class="muted">…仅显示前 8 条</p>
      </div>

      <div class="row actions">
        <button class="primary" :disabled="!canStart" @click="start">
          {{ starting ? '启动中…' : form.dry_run ? '开始统计' : '开始导入' }}
        </button>
        <button v-if="job?.status === 'running'" class="ghost" @click="cancel">取消</button>
      </div>
    </section>

    <section v-if="job" class="card">
      <h2>任务进度</h2>
      <ProgressBar :job="job" />
      <div v-if="summary" class="summary">
        <div v-for="[k, v] in summary" :key="k" class="stat">
          <span class="k">{{ k }}</span><span class="v">{{ v }}</span>
        </div>
      </div>
    </section>

    <section v-if="history.length" class="card">
      <h2>最近任务</h2>
      <ul class="history">
        <li v-for="h in history" :key="h.id">
          <span class="st" :class="h.status">{{ h.status }}</span>
          <span class="mono id">{{ h.id.slice(0, 8) }}</span>
          <span class="muted">{{ h.done }}/{{ h.total }}</span>
          <span class="muted msg">{{ h.message }}</span>
        </li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.import {
  padding: 18px;
  max-width: 900px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.card {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--panel);
  padding: 16px 18px;
}
h2 {
  margin: 0 0 8px;
  font-size: 14px;
}
.note {
  margin: 0 0 14px;
  font-size: 12px;
  color: var(--muted);
  line-height: 1.6;
}
.note code {
  background: var(--chip);
  padding: 1px 4px;
  border-radius: 3px;
}
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 14px;
}
.field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
}
.field.wide {
  grid-column: 1 / -1;
}
.field > span {
  color: var(--muted);
}
.field.check {
  flex-direction: row;
  align-items: center;
  gap: 6px;
}
input,
select,
textarea {
  font: inherit;
  font-size: 12.5px;
  padding: 6px 8px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--fg);
}
textarea {
  resize: vertical;
}
.row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 14px;
  flex-wrap: wrap;
}
.rule {
  font-size: 11px;
  color: var(--muted);
}
.pill {
  font-size: 11px;
  border-radius: 10px;
  padding: 2px 8px;
  border: 1px solid;
}
.pill.ok {
  color: var(--ok);
  border-color: color-mix(in srgb, var(--ok) 40%, transparent);
}
.pill.warn {
  color: var(--warn);
  border-color: color-mix(in srgb, var(--warn) 40%, transparent);
}
button {
  font: inherit;
  font-size: 12.5px;
  padding: 7px 14px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--fg);
  cursor: pointer;
}
button.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
button.ghost {
  background: none;
  color: var(--muted);
}
button:disabled {
  opacity: 0.5;
  cursor: default;
}
.actions {
  margin-top: 16px;
}
.advanced {
  margin-top: 14px;
  font-size: 12px;
}
.advanced summary {
  cursor: pointer;
  color: var(--muted);
  margin-bottom: 10px;
}
.banner {
  margin: 12px 0 0;
  padding: 8px 12px;
  font-size: 12px;
  border-radius: 6px;
  background: color-mix(in srgb, var(--danger) 12%, transparent);
  color: var(--danger);
}
.preview {
  margin-top: 14px;
  padding: 12px;
  border-radius: 8px;
  background: var(--bg);
  border: 1px solid var(--border);
  font-size: 12px;
}
.preview p {
  margin: 0 0 8px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11.5px;
}
th,
td {
  text-align: left;
  padding: 3px 6px;
  border-bottom: 1px solid var(--border);
}
th {
  color: var(--muted);
  font-weight: 500;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.muted {
  color: var(--muted);
  font-size: 11.5px;
}
.summary {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  margin-top: 12px;
}
.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.stat .k {
  font-size: 11px;
  color: var(--muted);
}
.stat .v {
  font-size: 17px;
  font-variant-numeric: tabular-nums;
}
.history {
  list-style: none;
  margin: 0;
  padding: 0;
  font-size: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.history li {
  display: flex;
  gap: 10px;
  align-items: baseline;
}
.st {
  font-size: 11px;
  border-radius: 4px;
  padding: 1px 6px;
  background: var(--chip);
}
.st.done {
  color: var(--ok);
}
.st.failed {
  color: var(--danger);
}
.st.running {
  color: var(--accent);
}
.id {
  color: var(--muted);
}
.msg {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
