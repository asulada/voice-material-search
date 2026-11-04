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
  kind: 'auto',
  path: '',
  target_use: '',
  license: 'apache-2.0',
  language: '',
  limit: null,
  clip_kind: null,
  overwrite: true,
  // 音频落盘：overwrite = 重写库里的 wav；keep = 沿用已有文件，只刷新元数据
  audio_mode: 'overwrite',
  dry_run: false,
  duration: null,
  gender_map_text: '',
  // 「选哪一层做分类名」：目录名或绝对路径；空 = 不启用，按目录形状自动判
  category_level: '',
})

const preview = ref(null)
const previewError = ref('')
const previewing = ref(false)
const job = ref(null)
const starting = ref(false)
const history = ref([])
const advancedOpen = ref(false)
let closeWs = null

/** 许可是否被用户手动改过：改过就不再自动预填 */
const licenseTouched = ref(false)

/**
 * 按预览结果推荐许可。
 *
 * 依据数据集的已知许可（方案 §4.5）：
 *   CSEMOTIONS / EmoGator → Apache-2.0（可商用）
 *   自定义目录             → 未知（commercial_ok=0，保守处理）
 *
 * 混合目录取"最保守"的那个单元：只要有一个是自定义目录，就按未知处理。
 *
 * 注意：这只是**预填**，不能代替用户确认——所以预览卡上会一直显示许可，
 * 即使它被折叠进高级选项里。
 */
function licenseForUnits(units) {
  if (!units || !units.length) return 'unknown'
  const kinds = units.map((u) => u.kind)
  return kinds.every((k) => k === 'csemotions' || k === 'emogator')
    ? 'apache-2.0'
    : 'unknown'
}

/** 把预览结果套用到表单：只预填许可。
 *
 * 线路不在这里回填——空值代表"自动，由每个数据源自己决定"，
 * 提前填上具体值会让后端误以为用户强制指定了线路。
 */
function applyPreview(p) {
  if (!licenseTouched.value) form.license = licenseForUnits(p.units)
}

// 线路由后端按每个数据源的类型决定，前端不再需要推算时长窗口

/** 单独一个单元时，在摘要里显示它的类型 */
function unitTitle(unit) {
  if (!unit) return '未识别'
  return kindLabel(unit.kind)
}

/** 单元标题里的类型标签：目录名本身就是类型名时不再重复显示 */
function unitKindTag(unit) {
  const label = kindLabel(unit.kind)
  return unit.relpath === label ? '' : label
}

/** 每个单元最多展示几条样本（混合目录时避免页面过长） */
const showSamples = 3

const licenseOptions = [
  { value: 'apache-2.0', label: 'Apache-2.0（CSEMOTIONS / EmoGator，可商用）' },
  { value: 'cc-by-nc-sa-4.0', label: 'CC-BY-NC-SA-4.0（speech-utterances，不可商用）' },
  { value: 'cc-by-4.0', label: 'CC-BY-4.0（署名，可商用）' },
  { value: 'unknown', label: '未知（保守处理为不可商用）' },
]

const kindOptions = [
  { value: 'auto', label: '自动识别（推荐；混合目录会分别处理）' },
  { value: 'csemotions', label: '强制 CSEMOTIONS（<说话人>/<情绪>/*.wav）' },
  { value: 'emogator', label: '强制 EmoGator（<类别>/*.mp3）' },
  { value: 'flat', label: '强制自定义目录（需 manifest.json）' },
]

/** 是否可商用。必须与后端 tags.commercial_ok_of 保持一致：
 *  只有明确的可商用许可才算 1；空值 / NC / 未知一律不可商用。 */
const commercialOk = computed(() => {
  const s = (form.license || '').trim().toLowerCase()
  if (!s) return false
  if (s.includes('nc') || s.includes('noncommercial') || s.includes('non-commercial')) return false
  if (['unknown', 'unspecified', 'unclear', 'n/a', 'na', 'none', 'other'].includes(s)) return false
  return true
})

function kindLabel(kind) {
  return { csemotions: 'CSEMOTIONS', emogator: 'EmoGator', flat: '自定义目录' }[kind] || kind
}

function licenseLabel(value) {
  return {
    'apache-2.0': 'Apache-2.0',
    'cc-by-nc-sa-4.0': 'CC-BY-NC-SA-4.0',
    'cc-by-4.0': 'CC-BY-4.0',
    unknown: '未知',
  }[value] || value
}

/**
 * 许可提醒：折叠进高级选项后，必须让用户看到"当前是按什么许可入库"。
 * 只提示两类真正会出错的情况，不做无意义的唠叨。
 */
const licenseWarning = computed(() => {
  if (form.license === 'unknown') {
    return '许可为「未知」：按保守策略记为不可商用（commercial_ok = 0）。若确认可商用，请在「高级选项 → 许可」里改。'
  }
  if (!commercialOk.value) {
    return `许可 ${licenseLabel(form.license)} 含 NC 条款：这批素材会被标为不可商用，检索时「仅可商用」会排除它们。`
  }
  return ''
})

const canStart = computed(() => form.path.trim() && !starting.value)

/**
 * 层级按钮是否处于选中态。
 *
 * 比对用绝对路径而不是目录名：祖先链里可能有同名目录
 * （例如 a/x 与 b/x），只比名字会同时点亮两个。
 */
function isLevelOn(lv) {
  const cur = (form.category_level || '').trim()
  if (!cur) {
    // 未指定时，等价于「就用当前导入目录这一层」
    return !!lv.is_root
  }
  return cur === lv.path || cur === lv.name
}

/** 选中某一层做分类名；再点一次取消，回到按形状自动判 */
async function pickLevel(lv) {
  const cur = (form.category_level || '').trim()
  const same = cur === lv.path || cur === lv.name
  // 点「当前导入目录」这一层且当前未指定时，等价于默认行为，不必设值
  form.category_level = same || (!cur && lv.is_root) ? '' : lv.path
  await onLevelChange()
}

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
    // 不传 kind：让后端按目录结构自动拆分（混合目录会拆成多个单元）
    preview.value = await api.previewImport(
      form.path.trim(),
      undefined,
      form.category_level || undefined,
    )
    applyPreview(preview.value)
  } catch (err) {
    previewError.value = err.message
  } finally {
    previewing.value = false
  }
}

/**
 * 切换分类层级后立刻重新预览。
 *
 * 必须重算：选定层级会改变「扫描范围」和「类别名」，
 * 旧的预览结果不再代表将要导入的内容。
 */
async function onLevelChange() {
  if (!form.path.trim()) return
  await runPreview()
}

async function start() {
  starting.value = true
  try {
    // 没预览过：先识别一次，让 license 等有依据，也避免导入空目录
    if (!preview.value) {
      try {
        preview.value = await api.previewImport(
          form.path.trim(),
          undefined,
          form.category_level || undefined,
        )
        applyPreview(preview.value)
      } catch (err) {
        previewError.value = `无法识别该目录，导入已取消：${err.message}`
        return
      }
    }

    const body = {
      kind: form.kind,          // 'auto' 时由后端按目录结构拆分
      path: form.path.trim(),
      license: form.license,
      language: form.language || '',
      limit: form.limit || null,
      clip_kind: form.clip_kind || null,
      overwrite: form.overwrite,
      audio_mode: form.audio_mode,
      dry_run: form.dry_run,
      gender_map: parseGenderMap(),
      duration: form.duration,
      // 选定分类层级时，后端整层统一类别、不再按形状拆
      category_level: form.category_level || null,
    }
    // 自动模式下线路由每个单元自己决定；手动指定类型时才传
    if (form.kind !== 'auto' && form.target_use) body.target_use = form.target_use
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
    // 过短与超长都不再丢弃：入库并写 notes，这里单独提示，避免被当成「全部可用」
    ['其中过短', r.imported_short || 0],
    ['其中超长', r.imported_over_long || 0],
    ['静音', r.skipped_silent],
    ['已存在', r.duplicated],
  ]
  if (r.skipped_short) rows.push(['太短（丢弃）', r.skipped_short])
  if (r.skipped_long) rows.push(['太长', r.skipped_long])
  if (r.error_count) rows.push(['错误', r.error_count])
  return rows
})

/** 有过短素材入库时给一条说明 */
const shortHint = computed(() => {
  const n = job.value?.result?.imported_short || 0
  if (!n) return ''
  return `其中 ${n} 条低于线路时长下限，已入库并在 notes 标注「作参考偏短」；默认检索按时长过滤，看不到它们，放宽时长后可见。`
})

/** 有超长素材入库时给一条说明，告诉用户它们不会出现在默认检索里 */
const overLongHint = computed(() => {
  const n = job.value?.result?.imported_over_long || 0
  if (!n) return ''
  return `其中 ${n} 条超出线路时长范围，已入库并在 notes 标注「需人工截取」；默认检索按线路时长过滤，看不到它们，放宽时长后可见。`
})

loadHistory()
onBeforeUnmount(() => closeWs?.())
</script>

<template>
  <div class="import">
    <section class="card">
      <h2>导入素材</h2>
      <p class="note">
        把数据集的本地目录填进来即可。目录类型、所属线路、许可都会在预览时自动识别；
        导入会依次：裁掉首尾数字静音 → 写进 <code>voice_library/audio/</code> → 生成中文描述 → 建向量 → 入库。
      </p>

      <div class="path-row">
        <input
          v-model="form.path"
          class="path-input"
          placeholder="数据源目录，例如 D:\datasets\CSEMOTIONS"
          @keydown.enter="runPreview"
        />
        <button class="ghost nowrap" :disabled="!form.path.trim() || previewing" @click="runPreview">
          {{ previewing ? '识别中…' : '预览' }}
        </button>
      </div>

      <!-- 识别结果：即使相关字段被折叠，这里也把「将要怎么做」摆明 -->
      <div v-if="preview" class="preview">
        <!-- 分类层级：选哪一层的目录名当类别，并决定扫描范围 -->
        <div v-if="preview.level_choices && preview.level_choices.length > 1" class="level-row">
          <span class="level-label">分类层级</span>
          <div class="level-chips">
            <button
              v-for="lv in preview.level_choices"
              :key="lv.path"
              type="button"
              class="level-chip"
              :class="{ on: isLevelOn(lv) }"
              :disabled="previewing"
              :title="lv.path"
              @click="pickLevel(lv)"
            >
              <b>{{ lv.name }}</b>
              <span class="level-count">{{ lv.count_truncated ? `≥${lv.count}` : lv.count }} 条</span>
            </button>
          </div>
        </div>
        <p v-if="form.category_level" class="level-note">
          已选「{{ form.category_level }}」：该层以下所有音频统一用这个名称作类别。
          选上级目录会扩大扫描范围（含其所有子目录）。
        </p>

        <div class="sum-row">
          <span class="chip">
            <b>{{ preview.mixed ? `识别到 ${preview.unit_count} 个数据源` : unitTitle(preview.units[0]) }}</b>
          </span>
          <span class="chip">共 <b>{{ preview.count }}</b> 条</span>
          <span class="chip" :class="commercialOk ? 'ok' : 'warn'">
            许可 <b>{{ licenseLabel(form.license) }}</b> · {{ commercialOk ? '可商用' : '不可商用' }}
          </span>
        </div>

        <p v-if="preview.mixed" class="mixed-note">
          这个目录里混有多种排布，将分别按各自的类型与线路一次导入：
        </p>

        <!-- 每个单元一行：这就是「将要做的事」清单 -->
        <div v-for="(u, i) in preview.units" :key="i" class="unit">
          <div class="unit-head">
            <span class="unit-name mono">{{ u.relpath }}</span>
            <span v-if="unitKindTag(u)" class="tag">{{ unitKindTag(u) }}</span>
            <span class="tag">{{ u.target_use === 'seedvc' ? 'SeedVC' : 'IndexTTS' }}</span>
            <span class="tag">{{ u.clip_kind === 'speech' ? '说话片段' : '副语言' }}</span>
            <span class="unit-count">{{ u.count }} 条</span>
          </div>
          <table>
            <thead>
              <tr><th>文件</th><th>类别</th><th>说话人</th><th>性别</th></tr>
            </thead>
            <tbody>
              <tr v-for="(s, j) in u.samples.slice(0, showSamples)" :key="j">
                <td class="mono">{{ s.relpath }}</td>
                <td>{{ s.category }}</td>
                <td>{{ s.speaker_id || '—' }}</td>
                <td>{{ s.gender || '—' }}</td>
              </tr>
            </tbody>
          </table>
          <p v-if="u.count > showSamples" class="muted">…仅显示前 {{ showSamples }} 条</p>
        </div>

        <p v-if="licenseWarning" class="license-warn">{{ licenseWarning }}</p>
      </div>

      <p v-else-if="!previewing" class="hint-line">
        选好目录后先点「预览」，确认识别结果无误再导入。
      </p>

      <details class="advanced" :open="advancedOpen" @toggle="advancedOpen = $event.target.open">
        <summary>高级选项<span class="muted">（一般不用动）</span></summary>
        <div class="grid">
          <label class="field">
            <span>目录类型</span>
            <select v-model="form.kind">
              <option v-for="k in kindOptions" :key="k.value" :value="k.value">{{ k.label }}</option>
            </select>
          </label>

          <label class="field">
            <span>用于线路</span>
            <select v-model="form.target_use">
              <option value="">自动（按每个数据源的类型）</option>
              <option value="indextts">IndexTTS（说话片段）</option>
              <option value="seedvc">SeedVC（副语言）</option>
            </select>
          </label>

          <label class="field">
            <span>许可</span>
            <select v-model="form.license" @change="licenseTouched = true">
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

          <label class="field">
            <span>强制素材类型</span>
            <select v-model="form.clip_kind">
              <option :value="null">自动（按线路）</option>
              <option value="speech">speech</option>
              <option value="vocal_burst">vocal_burst</option>
            </select>
          </label>

          <label class="field wide">
            <span>说话人性别映射（每行 <code>S03: male</code>；不填则性别为空，性别筛选会漏）</span>
            <textarea v-model="form.gender_map_text" rows="2" placeholder="S03: male&#10;S07: female" />
          </label>

          <label class="field check">
            <input v-model="form.dry_run" type="checkbox" />
            <span>只扫描不写盘（统计用）</span>
          </label>
          <label class="field check">
            <input v-model="form.overwrite" type="checkbox" />
            <span>覆盖已存在的同路径素材</span>
          </label>

          <label class="field">
            <span>音频文件</span>
            <select v-model="form.audio_mode">
              <option value="overwrite">覆盖导入（重写库里的 wav）</option>
              <option value="keep">沿用库里已有的（只刷新元数据）</option>
            </select>
          </label>
          <p class="field-hint">
            文件名由音频内容 hash 决定：同名即同内容。选「沿用」时不会重写磁盘文件，
            但元数据（描述 / 类别 / 语种等程序标注）照常刷新；文件若已丢失会自动补写。
            与上面的「覆盖已存在的同路径素材」不是一回事：那个决定遇到重复记录时是否跳过。
          </p>
        </div>
      </details>

      <p v-if="previewError" class="banner error">{{ previewError }}</p>

      <div class="row actions">
        <button class="primary" :disabled="!canStart" @click="start">
          {{ starting ? '启动中…' : form.dry_run ? '开始统计' : '开始导入' }}
        </button>
        <button v-if="job?.status === 'running'" class="ghost" @click="cancel">取消</button>
        <span v-if="!preview && form.path.trim()" class="muted">
          未预览也可以直接导入（会使用自动识别的设置）
        </span>
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
      <p v-if="shortHint" class="license-warn">{{ shortHint }}</p>
      <p v-if="overLongHint" class="license-warn">{{ overLongHint }}</p>
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
/* 选项说明：整行铺开，小字灰色，避免和字段标签抢注意力 */
.field-hint {
  grid-column: 1 / -1;
  margin: -2px 0 2px;
  font-size: 11px;
  line-height: 1.6;
  color: var(--muted);
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
/* 主输入：目录一行搞定 */
.path-row {
  display: flex;
  gap: 8px;
  align-items: center;
}
.path-input {
  flex: 1 1 auto;
  min-width: 0;
  font: inherit;
  font-size: 13px;
  padding: 8px 10px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--fg);
}
.path-input:focus {
  outline: 2px solid color-mix(in srgb, var(--accent) 45%, transparent);
  outline-offset: -1px;
  border-color: var(--accent);
}
.nowrap {
  flex: 0 0 auto;
  white-space: nowrap;
}
.hint-line {
  margin: 12px 0 0;
  font-size: 12px;
  color: var(--muted);
}
/* 识别结果摘要 */
.sum-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}
.chip {
  font-size: 11.5px;
  padding: 3px 9px;
  border-radius: 12px;
  background: var(--chip);
  border: 1px solid var(--border);
  color: var(--muted);
}
.chip b {
  color: var(--fg);
  font-weight: 600;
}
.chip.ok {
  color: var(--ok);
  border-color: color-mix(in srgb, var(--ok) 35%, transparent);
}
.chip.ok b {
  color: var(--ok);
}
.chip.warn {
  color: var(--warn);
  border-color: color-mix(in srgb, var(--warn) 35%, transparent);
}
.chip.warn b {
  color: var(--warn);
}
/* 分类层级：选哪一层目录名当类别 */
.level-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 10px;
  padding-bottom: 10px;
  border-bottom: 1px dashed var(--border);
}
.level-label {
  flex: 0 0 auto;
  font-size: 11.5px;
  color: var(--muted);
  padding-top: 4px;
}
.level-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  min-width: 0;
}
.level-chip {
  display: inline-flex;
  align-items: baseline;
  gap: 6px;
  font-size: 11.5px;
  padding: 3px 9px;
  border-radius: 12px;
  background: var(--chip);
  border: 1px solid var(--border);
  color: var(--muted);
  cursor: pointer;
}
.level-chip b {
  color: var(--fg);
  font-weight: 600;
}
.level-chip .level-count {
  font-size: 10.5px;
  opacity: 0.75;
}
.level-chip:hover:not(:disabled) {
  border-color: color-mix(in srgb, var(--fg) 25%, transparent);
}
.level-chip.on {
  border-color: var(--ok);
  background: color-mix(in srgb, var(--ok) 12%, transparent);
}
.level-chip.on b,
.level-chip.on .level-count {
  color: var(--ok);
}
.level-chip:disabled {
  opacity: 0.55;
  cursor: default;
}
.level-note {
  margin: -4px 0 10px;
  font-size: 11.5px;
  color: var(--muted);
}
/* 混合目录：每个数据源一张小卡 */
.mixed-note {
  margin: 0 0 10px;
  font-size: 11.5px;
  color: var(--muted);
}
.unit {
  margin-bottom: 10px;
  padding: 8px 10px;
  border-radius: 6px;
  background: var(--panel);
  border: 1px solid var(--border);
}
.unit:last-child {
  margin-bottom: 0;
}
.unit-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}
.unit-name {
  font-size: 11.5px;
  font-weight: 600;
  margin-right: 2px;
}
.tag {
  font-size: 10.5px;
  padding: 1px 7px;
  border-radius: 10px;
  background: var(--chip);
  border: 1px solid var(--border);
  color: var(--muted);
}
.unit-count {
  margin-left: auto;
  font-size: 11px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.license-warn {
  margin: 0 0 10px;
  padding: 7px 10px;
  font-size: 11.5px;
  line-height: 1.6;
  border-radius: 6px;
  color: var(--warn);
  background: color-mix(in srgb, var(--warn) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--warn) 30%, transparent);
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
