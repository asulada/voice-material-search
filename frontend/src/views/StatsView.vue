<script setup>
/** 统计 + 维护：分布、向量状态、全库重建向量、导出历史。 */
import { computed, ref } from 'vue'
import { api, watchJob } from '../api/client'
import ProgressBar from '../components/ProgressBar.vue'

const props = defineProps({
  stats: { type: Object, default: null },
  meta: { type: Object, default: null },
})
const emit = defineEmits(['changed'])

const job = ref(null)
const error = ref('')
const exports = ref([])

async function reembed(missingOnly = false) {
  error.value = ''
  try {
    const { job_id } = await api.reembedAll({ missing_only: missingOnly })
    job.value = { id: job_id, status: 'running', done: 0, total: 0 }
    watchJob(job_id, (frame) => {
      if (frame.error) {
        job.value = { ...job.value, status: 'failed', message: frame.error }
        return
      }
      job.value = frame
      if (['done', 'failed', 'cancelled'].includes(frame.status)) emit('changed')
    })
  } catch (err) {
    error.value = err.message
  }
}

async function loadExports() {
  try {
    exports.value = (await api.exports()).exports
  } catch (_) {
    exports.value = []
  }
}
loadExports()

const missingEmbeddings = computed(() => {
  const s = props.stats
  if (!s) return 0
  return Math.max(0, s.total - s.embedded)
})

const distBars = (obj) => {
  const entries = Object.entries(obj || {})
  const max = entries.reduce((m, [, v]) => Math.max(m, v), 0) || 1
  return entries.map(([k, v]) => ({ k, v, pct: (v / max) * 100 }))
}
</script>

<template>
  <div class="stats">
    <section class="cards">
      <div class="kpi">
        <span class="k">素材总量</span>
        <span class="v">{{ stats?.total ?? '—' }}</span>
      </div>
      <div class="kpi">
        <span class="k">总时长</span>
        <span class="v">{{ stats ? (stats.total_duration_sec / 60).toFixed(1) : '—' }} <small>分钟</small></span>
      </div>
      <div class="kpi">
        <span class="k">已验证</span>
        <span class="v">{{ stats?.verified ?? '—' }}</span>
      </div>
      <div class="kpi">
        <span class="k">已建向量</span>
        <span class="v">
          {{ stats?.embedded ?? '—' }}
          <small v-if="missingEmbeddings">（缺 {{ missingEmbeddings }}）</small>
        </span>
      </div>
      <div class="kpi">
        <span class="k">可商用</span>
        <span class="v">{{ stats?.commercial_ok ?? '—' }}</span>
      </div>
    </section>

    <section class="card">
      <h2>线路分布</h2>
      <table v-if="stats?.by_use?.length">
        <thead><tr><th>用途</th><th>类型</th><th>数量</th></tr></thead>
        <tbody>
          <tr v-for="u in stats.by_use" :key="`${u.target_use}-${u.clip_kind}`">
            <td>{{ u.target_use }}</td>
            <td>{{ u.clip_kind }}</td>
            <td class="num">{{ u.count }}</td>
          </tr>
        </tbody>
      </table>
      <p v-else class="muted">库里还没有素材。</p>
    </section>

    <section v-if="stats?.categories" class="card">
      <h2>各线路可用类别</h2>
      <div class="cols">
        <div v-for="(list, use) in stats.categories" :key="use" class="col">
          <h3>{{ use === 'indextts' ? 'IndexTTS' : 'SeedVC' }}</h3>
          <ul>
            <li v-for="c in list" :key="c.category">
              <span>{{ c.category_cn || c.category }}</span><span class="num">{{ c.count }}</span>
            </li>
            <li v-if="!list.length" class="muted">暂无</li>
          </ul>
        </div>
      </div>
    </section>

    <div class="two">
      <section class="card">
        <h2>类别分布（全库）</h2>
        <div class="bars">
          <div v-for="b in distBars(stats?.by_category)" :key="b.k" class="bar-row">
            <span class="label">{{ b.k }}</span>
            <div class="track"><div class="fill" :style="{ width: `${b.pct}%` }" /></div>
            <span class="num">{{ b.v }}</span>
          </div>
          <p v-if="!stats?.total" class="muted">暂无数据。</p>
        </div>
      </section>

      <section class="card">
        <h2>许可与语言</h2>
        <div class="bars">
          <div v-for="b in distBars(stats?.by_license)" :key="b.k" class="bar-row">
            <span class="label">{{ b.k }}</span>
            <div class="track"><div class="fill" :style="{ width: `${b.pct}%` }" /></div>
            <span class="num">{{ b.v }}</span>
          </div>
        </div>
        <div class="bars" style="margin-top: 14px">
          <div v-for="b in distBars(stats?.by_gender)" :key="b.k" class="bar-row">
            <span class="label">{{ b.k === 'male' ? '男' : b.k === 'female' ? '女' : b.k }}</span>
            <div class="track"><div class="fill alt" :style="{ width: `${b.pct}%` }" /></div>
            <span class="num">{{ b.v }}</span>
          </div>
        </div>
      </section>
    </div>

    <section class="card">
      <h2>维护</h2>
      <p class="muted">
        描述改了必须重建向量（单条在检索页保存时自动做）。这里用于批量补建或全库重建。
        当前模型：<b>{{ meta?.encoder?.model }}</b>（{{ meta?.encoder?.backend }}，{{ meta?.encoder?.dim }} 维）
      </p>
      <div class="row">
        <button class="primary" :disabled="job?.status === 'running'" @click="reembed(true)">
          补建缺失向量（{{ missingEmbeddings }} 条）
        </button>
        <button class="ghost" :disabled="job?.status === 'running'" @click="reembed(false)">
          全库重建向量
        </button>
      </div>
      <ProgressBar v-if="job" :job="job" />
      <p v-if="error" class="err">{{ error }}</p>
    </section>

    <section class="card">
      <h2>导出历史</h2>
      <ul v-if="exports.length" class="exports">
        <li v-for="e in exports" :key="e.name">
          <span class="mono">{{ e.name }}</span>
          <span class="num">{{ e.count ?? '?' }} 条</span>
          <span class="muted mono">{{ e.path }}</span>
        </li>
      </ul>
      <p v-else class="muted">还没有导出记录。</p>
    </section>
  </div>
</template>

<style scoped>
.stats {
  padding: 18px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1100px;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}
.kpi {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--panel);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.kpi .k {
  font-size: 11px;
  color: var(--muted);
}
.kpi .v {
  font-size: 22px;
  font-variant-numeric: tabular-nums;
}
.kpi small {
  font-size: 11px;
  color: var(--muted);
}
.card {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--panel);
  padding: 16px 18px;
}
h2 {
  margin: 0 0 10px;
  font-size: 13px;
}
h3 {
  margin: 0 0 6px;
  font-size: 12px;
  color: var(--muted);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
th,
td {
  text-align: left;
  padding: 4px 6px;
  border-bottom: 1px solid var(--border);
}
th {
  color: var(--muted);
  font-weight: 500;
}
.num {
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.cats,
.cols {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}
ul {
  list-style: none;
  margin: 0;
  padding: 0;
  font-size: 12px;
}
li {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 3px 0;
}
.two {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.bars {
  display: flex;
  flex-direction: column;
  gap: 5px;
  font-size: 12px;
}
.bar-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.bar-row .label {
  flex: 0 0 84px;
  color: var(--fg);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.track {
  flex: 1 1 auto;
  height: 8px;
  border-radius: 4px;
  background: var(--chip);
  overflow: hidden;
}
.fill {
  height: 100%;
  background: var(--accent);
}
.fill.alt {
  background: var(--ok);
}
.row {
  display: flex;
  gap: 10px;
  margin: 12px 0;
  flex-wrap: wrap;
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
.muted {
  color: var(--muted);
  font-size: 12px;
}
.err {
  color: var(--danger);
  font-size: 12px;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
}
.exports li {
  align-items: baseline;
  gap: 12px;
}
.exports .muted {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
