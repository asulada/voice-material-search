<script setup>
/**
 * 检索 + 结果 + 试听 + 标注（方案 §6.4）。
 *
 * 交互要点：
 *   1. 结果项内联试听，不跳页；用 gain_db 统一音量
 *   2. 星标 / ✓ 行内更新，不重载列表
 *   3. 多选导出，写 selection.json
 *   4. 描述可编辑，保存后自动重建向量
 */
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api/client'
import ClipCard from '../components/ClipCard.vue'
import SearchFilter from '../components/SearchFilter.vue'

const props = defineProps({
  meta: { type: Object, default: null },
  stats: { type: Object, default: null },
})
const emit = defineEmits(['changed'])

const query = reactive({
  use: 'indextts',
  q: '',
  categories: [],
  gender: null,
  min_intensity: 0,
  duration: [3, 15],
  commercial_only: false,
  verified_only: false,
  top_k: 20,
  offset: 0,
  sort: 'relevance',
})

const uses = computed(() => props.meta?.uses || [])
const result = ref({ total: 0, items: [], returned: 0 })
const categories = ref([])
const loading = ref(false)
/** 任一请求在飞时为真：控制整体「更新中」提示与交互禁用 */
const pending = ref(false)
const exporting = ref(false)
const errorText = ref('')
const toast = ref('')
const selected = reactive(new Set())
const waveforms = reactive({}) // id -> number[]
const busyIds = reactive(new Set())
let toastTimer = null
/** 检索请求序号：只有最后一次发出的请求可以写结果，避免慢响应覆盖新结果 */
let searchSeq = 0
/** 已经为哪条线路拉过类别：切换线路时才重新拉，避免每次筛选都请求 */
let categoriesFor = null
/** 是否已经拿到过第一批结果：决定「共 N 条」还是「检索中…」 */
const hasLoaded = ref(false)

const backend = computed(() => props.meta?.encoder?.backend || '')

function flash(msg) {
  toast.value = msg
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => (toast.value = ''), 2600)
}

async function loadCategories({ force = false } = {}) {
  // 类别只跟线路有关；筛选条件变化时不必重拉（之前每次都拉，白白闪一下「该线路暂无素材」）
  if (!force && categoriesFor === query.use) return
  const use = query.use
  try {
    if (use === 'all') {
      // 全库：把两条线路的类别合并去重
      const [a, b] = await Promise.all([api.categories('indextts'), api.categories('seedvc')])
      const merged = new Map()
      for (const c of [...a.categories, ...b.categories]) {
        merged.set(c.category, (merged.get(c.category) || 0) + c.count)
      }
      categories.value = [...merged.entries()]
        .map(([category, count]) => ({ category, count }))
        .sort((x, y) => y.count - x.count)
    } else {
      categories.value = (await api.categories(use)).categories
    }
    if (use === query.use) categoriesFor = use
  } catch (err) {
    // 类别拉取失败不该阻断检索；下一次切线路会重试
    categoriesFor = null
    flash(`类别加载失败：${err.message}`)
  }
}

async function runSearch({ keepSelection = false } = {}) {
  const seq = ++searchSeq
  loading.value = true
  pending.value = true
  errorText.value = ''
  if (!keepSelection) selected.clear()
  const current = { ...query }
  try {
    const body = {
      use: current.use === 'all' ? 'all' : current.use,
      q: current.q?.trim() || null,
      categories: current.categories.length ? current.categories : null,
      gender: current.gender,
      min_intensity: current.min_intensity > 0 ? current.min_intensity : null,
      duration: current.duration?.length === 2 ? current.duration : null,
      commercial_only: current.commercial_only,
      verified_only: current.verified_only,
      top_k: current.top_k,
      offset: current.offset,
      sort: current.sort,
    }
    const data = await api.search(body)
    // 只有最后发出的那次请求可以落地：筛选连点时慢响应会被丢弃
    if (seq !== searchSeq) return
    result.value = data
    hasLoaded.value = true
    loading.value = false // 波形加载不该拖住「检索中…」
    if (seq === searchSeq) pending.value = false
    loadWaveforms(data.items)
  } catch (err) {
    if (seq !== searchSeq) return
    errorText.value = err.message
    result.value = { total: 0, items: [], returned: 0 }
  } finally {
    if (seq === searchSeq) {
      loading.value = false
      pending.value = false
    }
  }
}

async function loadWaveforms(items) {
  // 波形是装饰，失败就算了；并发拉取，但限制同时进行的请求数
  const pending = items.filter((it) => !waveforms[it.id])
  const concurrency = 6
  for (let i = 0; i < pending.length; i += concurrency) {
    const batch = pending.slice(i, i + concurrency)
    await Promise.all(
      batch.map(async (it) => {
        try {
          const w = await api.waveform(it.id, 220)
          waveforms[it.id] = w.points
        } catch (_) {
          waveforms[it.id] = []
        }
      }),
    )
  }
}

function onFilterUpdate(v) {
  Object.assign(query, v)
  query.offset = 0
  // 类别数量只随线路变化；筛选变化时不必等类别接口
  const needCats = categoriesFor !== query.use
  if (needCats) loadCategories().then(runSearch)
  else runSearch()
}

/** 重置：面板一次性给出整套默认值，这里覆盖后只检索一次 */
function onReset(v) {
  Object.assign(query, v)
  query.offset = 0
  selected.clear()
  loadCategories().then(runSearch)
}

async function patchClip(id, patch) {
  busyIds.add(id)
  try {
    const updated = await api.patchClip(id, patch)
    const idx = result.value.items.findIndex((c) => c.id === id)
    if (idx >= 0) Object.assign(result.value.items[idx], updated)
    emit('changed')
  } catch (err) {
    flash(`更新失败：${err.message}`)
  } finally {
    busyIds.delete(id)
  }
}

async function describeClip(id, text) {
  busyIds.add(id)
  try {
    const updated = await api.patchClip(id, { description: text })
    const idx = result.value.items.findIndex((c) => c.id === id)
    if (idx >= 0) Object.assign(result.value.items[idx], updated)
    flash(updated.reembedded ? '描述已保存，已重建向量' : '描述已保存')
    emit('changed')
  } catch (err) {
    flash(`保存失败：${err.message}`)
  } finally {
    busyIds.delete(id)
  }
}

function toggleSelect(id, on) {
  if (on) selected.add(id)
  else selected.delete(id)
}

const selectedCount = computed(() => selected.size)

async function exportSelected() {
  if (!selected.size) return
  exporting.value = true
  try {
    const res = await api.exportClips({
      clip_ids: [...selected],
      name: '',
      apply_gain: false,
    })
    flash(`已导出 ${res.exported} 条到 ${res.directory}`)
    if (res.missing?.length) flash(`已导出 ${res.exported} 条；${res.missing.length} 条缺失`)
    emit('changed')
  } catch (err) {
    flash(`导出失败：${err.message}`)
  } finally {
    exporting.value = false
  }
}

function selectAllVisible() {
  for (const c of result.value.items) selected.add(c.id)
}

const canPrev = computed(() => query.offset > 0)
const canNext = computed(() => query.offset + query.top_k < result.value.total)

function page(dir) {
  query.offset = Math.max(0, query.offset + dir * query.top_k)
  runSearch({ keepSelection: true })
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

watch(
  () => query.use,
  () => {
    selected.clear()
  },
)

onMounted(async () => {
  const rule = uses.value.find((u) => u.value === query.use)
  if (rule) {
    query.duration = [...rule.duration]
  }
  await loadCategories()
  await runSearch()
})
</script>

<template>
  <div class="layout">
    <SearchFilter
      :uses="uses"
      :categories="categories"
      :model-value="query"
      :busy="pending"
      @update:model-value="onFilterUpdate"
      @search="query.offset = 0; runSearch()"
      @reset="onReset"
    />

    <main class="results">
      <div class="toolbar">
        <div class="count" :class="{ refreshing: loading }">
          <span class="spinner" v-if="loading" aria-hidden="true" />
          <template v-if="hasLoaded">
            共 <b>{{ result.total }}</b> 条
            <span v-if="result.total > result.returned" class="sub">
              （显示 {{ query.offset + 1 }}–{{ query.offset + result.returned }}）
            </span>
          </template>
          <template v-else>检索中…</template>
        </div>
        <div class="spacer" />
        <button class="tool" :disabled="!result.items.length" @click="selectAllVisible">
          选中本页
        </button>
        <button class="tool primary" :disabled="!selectedCount || exporting" @click="exportSelected">
          {{ exporting ? '导出中…' : `导出选中 ${selectedCount} 条` }}
        </button>
      </div>

      <p v-if="backend && backend !== 'sentence-transformers'" class="banner">
        嵌入模型未加载（当前为 <b>{{ backend }}</b>），只会按字面匹配。
        安装并下载 BGE 后重启即可启用语义检索：
        <code>pip install sentence-transformers</code>
      </p>

      <p v-if="errorText" class="banner error">检索失败：{{ errorText }}</p>

      <div v-if="!result.items.length && !loading && hasLoaded" class="empty">
        <p>没有命中。</p>
        <p class="hint" v-if="!query.q && !query.categories.length">
          库里还没有这个线路的素材 —— 去「导入」页把数据集加进来。
        </p>
        <p class="hint" v-else>
          试着放宽筛选（时长区间 / 仅已验证 / 最低强度），或换一种描述说法。
        </p>
      </div>

      <div class="list" :class="{ refreshing: loading }">
        <ClipCard
          v-for="clip in result.items"
          :key="clip.id"
          :clip="clip"
          :selected="selected.has(clip.id)"
          :waveform="waveforms[clip.id] || []"
          :busy="busyIds.has(clip.id)"
          @select="toggleSelect"
          @patch="patchClip"
          @describe="describeClip"
        />
      </div>

      <div v-if="result.total > query.top_k" class="pager">
        <button :disabled="!canPrev" @click="page(-1)">上一页</button>
        <span>{{ Math.floor(query.offset / query.top_k) + 1 }} / {{ Math.ceil(result.total / query.top_k) }}</span>
        <button :disabled="!canNext" @click="page(1)">下一页</button>
      </div>
    </main>

    <transition name="toast">
      <div v-if="toast" class="toast">{{ toast }}</div>
    </transition>
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  flex: 1 1 auto;
  min-height: 0;
}
.results {
  flex: 1 1 auto;
  min-width: 0;
  padding: 14px 18px 40px;
  overflow-y: auto;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  position: sticky;
  top: 0;
  z-index: 2;
  background: var(--bg);
  padding: 6px 0;
}
.count {
  font-size: 13px;
  color: var(--fg);
  display: flex;
  align-items: center;
  gap: 6px;
  /* 数字位数变化时不要让右半边工具按钮跟着跳 */
  min-width: 190px;
}
.count b {
  font-variant-numeric: tabular-nums;
  /* 数字用等宽字形，1 与 8 占同样宽度，计数刷新时不抖动 */
  font-feature-settings: 'tnum';
}
/* 未拿到第一批结果前先占住位置，避免「检索中…」→「共 N 条」造成高度跳变 */
.count.refreshing {
  color: var(--muted);
}
.spinner {
  width: 10px;
  height: 10px;
  flex: 0 0 auto;
  border-radius: 50%;
  border: 1.5px solid var(--chip);
  border-top-color: var(--accent);
  animation: spin 0.7s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
/* 列表刷新时只做轻微的透明度变化，不替换节点、不重排 */
.list {
  transition: opacity 0.15s ease;
}
.list.refreshing {
  opacity: 0.55;
}
.sub {
  color: var(--muted);
  font-size: 11px;
}
.spacer {
  flex: 1 1 auto;
}
.tool {
  font: inherit;
  font-size: 12px;
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--fg);
  cursor: pointer;
}
.tool.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.tool:disabled {
  opacity: 0.5;
  cursor: default;
}
.banner {
  margin: 0 0 12px;
  padding: 8px 12px;
  font-size: 12px;
  border-radius: 6px;
  background: color-mix(in srgb, var(--warn) 12%, transparent);
  color: var(--warn);
  border: 1px solid color-mix(in srgb, var(--warn) 30%, transparent);
}
.banner code {
  font-size: 11px;
  background: var(--chip);
  padding: 1px 4px;
  border-radius: 3px;
}
.banner.error {
  background: color-mix(in srgb, var(--danger) 12%, transparent);
  color: var(--danger);
  border-color: color-mix(in srgb, var(--danger) 30%, transparent);
}
.empty {
  padding: 50px 20px;
  text-align: center;
  color: var(--muted);
}
.empty p {
  margin: 4px 0;
}
.hint {
  font-size: 12px;
  opacity: 0.8;
}
.list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.pager {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  margin-top: 18px;
  font-size: 12px;
  color: var(--muted);
}
.pager button {
  font: inherit;
  font-size: 12px;
  padding: 5px 12px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--fg);
  cursor: pointer;
}
.pager button:disabled {
  opacity: 0.4;
  cursor: default;
}
.toast {
  position: fixed;
  left: 50%;
  bottom: 26px;
  transform: translateX(-50%);
  background: var(--toast);
  color: var(--toast-fg);
  padding: 9px 16px;
  border-radius: 8px;
  font-size: 12.5px;
  box-shadow: 0 6px 22px rgba(0, 0, 0, 0.3);
  max-width: 80vw;
  z-index: 50;
}
.toast-enter-active,
.toast-leave-active {
  transition: opacity 0.2s, transform 0.2s;
}
.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(6px);
}
</style>
