<script setup>
/**
 * 左侧筛选面板（方案 §6.4）。
 * 两条线路互斥，切换用途时：清空类别与时长（默认时长由 meta 里的 usage 规则给出）。
 */
import { computed, ref, watch } from 'vue'

const props = defineProps({
  uses: { type: Array, default: () => [] },
  categories: { type: Array, default: () => [] },
  modelValue: { type: Object, required: true },
  busy: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'search', 'reset'])

const local = ref({ ...props.modelValue })
watch(
  () => props.modelValue,
  (v) => {
    local.value = { ...v }
  },
  { deep: true },
)

// 线路切换：必须由 local.use 的变化驱动（父级不会先改）。
// 之前监听 props.modelValue.use 是错的——点击按钮只改 local，
// 父级的 use 没变，watch 不触发，于是检索仍在旧线路上跑。
watch(
  () => local.value.use,
  (use, prev) => {
    if (!prev || use === prev) return
    const rule = props.uses.find((u) => u.value === use)
    local.value.categories = []
    if (rule) {
      local.value.duration = [...rule.duration]
    }
    push()
  },
)

const currentRule = computed(() => props.uses.find((u) => u.value === local.value.use))

function push() {
  emit('update:modelValue', { ...local.value })
}

function toggleCategory(cat) {
  const list = local.value.categories || []
  local.value.categories = list.includes(cat)
    ? list.filter((c) => c !== cat)
    : [...list, cat]
  push()
}

function submit() {
  emit('search')
}

function reset() {
  local.value.categories = []
  local.value.gender = null
  local.value.min_intensity = 0
  local.value.commercial_only = false
  local.value.verified_only = false
  local.value.q = ''
  local.value.duration = currentRule.value ? [...currentRule.value.duration] : null
  // 只发一次事件：父级据此覆盖 query 并检索一次。
  // 之前这里还 push() 了一次，于是重置会连发两次检索，列表闪两下。
  emit('reset', { ...local.value })
}

const totalCategoryCount = computed(() =>
  (props.categories || []).reduce((s, c) => s + c.count, 0),
)
</script>

<template>
  <aside class="filters">
    <div class="block">
      <label class="lbl">线路</label>
      <div class="seg">
        <button
          v-for="u in uses"
          :key="u.value"
          :class="{ on: local.use === u.value }"
          :title="`${u.kinds.join('/')} · ${u.duration[0]}–${u.duration[1]}s`"
          @click="local.use = u.value"
        >
          {{ u.value === 'indextts' ? 'IndexTTS' : u.value === 'seedvc' ? 'SeedVC' : '全库' }}
        </button>
      </div>
      <p v-if="currentRule" class="rule">
        仅 {{ currentRule.kinds.join(' / ') }} · {{ currentRule.duration[0] }}–{{ currentRule.duration[1] }} 秒
      </p>
    </div>

    <div class="block">
      <label class="lbl">
        描述检索
        <span class="sub">语义，不是音频</span>
      </label>
      <input
        v-model="local.q"
        class="q"
        type="search"
        placeholder="例如：压抑但克制的愤怒"
        @keydown.enter="submit"
      />
      <button class="primary wide fixed" :disabled="busy" @click="submit">
        <span class="btn-inner">{{ busy ? '检索中…' : '检索' }}</span>
      </button>
    </div>

    <div class="block">
      <label class="lbl">
        类别
        <span class="sub">{{ (local.categories || []).length || '全部' }}</span>
      </label>
      <div class="cats">
        <button
          v-for="c in categories"
          :key="c.category"
          class="cat"
          :class="{ on: (local.categories || []).includes(c.category) }"
          @click="toggleCategory(c.category)"
        >
          {{ c.category }}<span class="n">{{ c.count }}</span>
        </button>
        <p v-if="!categories.length" class="empty">该线路暂无素材</p>
        <p v-else-if="!totalCategoryCount" class="empty">暂无</p>
      </div>
    </div>

    <div class="block">
      <label class="lbl">性别</label>
      <div class="seg small">
        <button :class="{ on: !local.gender }" @click="local.gender = null; push()">不限</button>
        <button :class="{ on: local.gender === 'male' }" @click="local.gender = 'male'; push()">男</button>
        <button :class="{ on: local.gender === 'female' }" @click="local.gender = 'female'; push()">女</button>
      </div>
    </div>

    <div class="block">
      <label class="lbl">
        最低强度
        <span class="sub">{{ local.min_intensity ? `≥ ${local.min_intensity}★` : '不限' }}</span>
      </label>
      <input
        class="strength"
        type="range"
        min="0"
        max="5"
        step="1"
        :value="local.min_intensity"
        @input="local.min_intensity = Number($event.target.value)"
        @change="push"
      />
      <div class="scale" aria-hidden="true">
        <span :class="{ on: local.min_intensity <= 0 }">不限</span>
        <span v-for="n in 5" :key="n" :class="{ on: local.min_intensity === n }">{{ n }}★</span>
      </div>
      <p class="help">
        素材的 <b>听感强度</b>（1–5 星，人工打分，就是结果条上的星标）。拖动后只显示<b>达到或超过</b>该星级的素材；
        未打过分的素材（无星级）会被筛掉。不知道选几就留「不限」。
      </p>
    </div>

    <div class="block">
      <label class="lbl">时长（秒）</label>
      <div class="range">
        <input
          v-model.number="local.duration[0]"
          type="number"
          step="0.1"
          min="0"
          @change="push"
        />
        <span>–</span>
        <input
          v-model.number="local.duration[1]"
          type="number"
          step="0.1"
          min="0"
          @change="push"
        />
      </div>
    </div>

    <div class="block">
      <label class="check">
        <input v-model="local.commercial_only" type="checkbox" @change="push" />
        仅可商用
      </label>
      <label class="check">
        <input v-model="local.verified_only" type="checkbox" @change="push" />
        仅已验证
      </label>
      <label class="check">
        <input v-model="local.sort" type="radio" value="quality" @change="push" />
        优先质量排序
      </label>
    </div>

    <button class="wide ghost" @click="reset">重置筛选</button>
  </aside>
</template>

<style scoped>
.filters {
  width: 250px;
  flex: 0 0 auto;
  padding: 14px;
  overflow-y: auto;
  border-right: 1px solid var(--border);
  background: var(--sidebar);
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.block {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.lbl {
  font-size: 12px;
  font-weight: 600;
  color: var(--fg);
  display: flex;
  align-items: baseline;
  justify-content: space-between;
}
.sub {
  font-weight: 400;
  color: var(--muted);
  font-size: 11px;
}
.rule {
  margin: 2px 0 0;
  font-size: 11px;
  color: var(--muted);
}
.seg {
  display: flex;
  gap: 4px;
}
.seg button {
  flex: 1 1 0;
  font-size: 12px;
  padding: 5px 4px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--muted);
  cursor: pointer;
}
.seg button.on {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.seg.small button {
  font-size: 12px;
  padding: 4px 2px;
}
.q {
  font: inherit;
  font-size: 13px;
  padding: 7px 9px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--fg);
}
.q:focus {
  outline: 2px solid color-mix(in srgb, var(--accent) 45%, transparent);
  outline-offset: -1px;
  border-color: var(--accent);
}
button.primary {
  font: inherit;
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid var(--accent);
  background: var(--accent);
  color: #fff;
  cursor: pointer;
}
button.primary:disabled {
  opacity: 0.6;
  cursor: progress;
}
/* 「检索」↔「检索中…」切换时文字长度会变，但按钮宽度固定，
   且用不可见的占位撑住高度 —— 否则整块面板会跟着抖一下 */
button.primary.fixed {
  position: relative;
}
button.primary.fixed .btn-inner {
  display: inline-block;
  min-width: 5.5em;
  text-align: center;
}
button.ghost {
  font: inherit;
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: none;
  color: var(--muted);
  cursor: pointer;
}
.wide {
  width: 100%;
}
.cats {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.cat {
  font: inherit;
  font-size: 12px;
  padding: 3px 7px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--fg);
  cursor: pointer;
}
.cat .n {
  color: var(--muted);
  font-size: 10px;
  margin-left: 4px;
  font-variant-numeric: tabular-nums;
}
.cat.on {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.cat.on .n {
  color: rgba(255, 255, 255, 0.8);
}
.empty {
  font-size: 11px;
  color: var(--muted);
  margin: 0;
}
.range {
  display: flex;
  align-items: center;
  gap: 6px;
}
.range input {
  width: 100%;
  min-width: 0;
  font: inherit;
  font-size: 12px;
  padding: 4px 6px;
  border-radius: 5px;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--fg);
}
.check {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  cursor: pointer;
  color: var(--fg);
}
input[type='range'] {
  accent-color: var(--accent);
}
/* 「最低强度」的刻度与说明：让这个参数自解释，不再需要猜 */
.scale {
  display: flex;
  justify-content: space-between;
  font-size: 10.5px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  padding: 0 1px;
}
.scale .on {
  color: var(--accent);
  font-weight: 600;
}
.help {
  margin: 2px 0 0;
  font-size: 10.5px;
  line-height: 1.6;
  color: var(--muted);
  background: var(--chip);
  border-radius: 5px;
  padding: 6px 7px;
}
.help b {
  color: var(--fg);
  font-weight: 600;
}
</style>
