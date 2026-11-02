<script setup>
/**
 * 封装 <audio> + GainNode（方案 §6.5）。
 *
 * 关键点：
 *   - gain_db 只在这里临时应用，不改文件（§4.4）
 *   - 浏览器要求用户手势后才能 resume()，首次播放必须在点击事件里
 *   - MediaElementSource 每个元素只能创建一次，缓存在 WeakMap 里
 */
import { computed, onBeforeUnmount, ref, watch } from 'vue'

const props = defineProps({
  src: { type: String, required: true },
  gainDb: { type: Number, default: 0 },
  waveform: { type: Array, default: () => [] },
  durationSec: { type: Number, default: 0 },
  compact: { type: Boolean, default: false },
})

const emit = defineEmits(['ended', 'play'])

const audioEl = ref(null)
const playing = ref(false)
const loading = ref(false)
const currentTime = ref(0)
const error = ref('')

let ctx = null
let gainNode = null
let sourceNode = null
const sources = new WeakMap()

const gainValue = computed(() => Math.pow(10, (props.gainDb || 0) / 20))
const progress = computed(() =>
  props.durationSec > 0 ? Math.min(1, currentTime.value / props.durationSec) : 0,
)
const gainLabel = computed(() => {
  const g = props.gainDb || 0
  return `${g > 0 ? '+' : ''}${g.toFixed(1)} dB`
})

function ensureGraph() {
  if (ctx) return
  const AC = window.AudioContext || window.webkitAudioContext
  if (!AC || !audioEl.value) return
  ctx = new AC()
  gainNode = ctx.createGain()
  gainNode.gain.value = gainValue.value
  // 如果浏览器阻止了对 <audio> 建源（CORS / 未解码），退化为直接播放原音量
  try {
    let src = sources.get(audioEl.value)
    if (!src) {
      src = ctx.createMediaElementSource(audioEl.value)
      sources.set(audioEl.value, src)
    }
    sourceNode = src
    sourceNode.connect(gainNode).connect(ctx.destination)
  } catch (err) {
    console.warn('无法建立 WebAudio 增益链，退化为原始音量', err)
    ctx = null
    gainNode = null
  }
}

async function toggle() {
  const el = audioEl.value
  if (!el) return
  if (playing.value) {
    el.pause()
    return
  }
  error.value = ''
  loading.value = true
  try {
    ensureGraph()
    // 必须在点击事件（用户手势）里 resume
    if (ctx?.state === 'suspended') await ctx.resume()
    if (gainNode) gainNode.gain.value = gainValue.value
    el.currentTime = el.currentTime || 0
    await el.play()
    playing.value = true
    emit('play')
  } catch (err) {
    error.value = `播放失败：${err.message || err}`
  } finally {
    loading.value = false
  }
}

function onEnded() {
  playing.value = false
  currentTime.value = 0
  emit('ended')
}
function onTimeUpdate() {
  currentTime.value = audioEl.value?.currentTime || 0
}
function onPause() {
  playing.value = false
}
function seekTo(ev) {
  const el = audioEl.value
  if (!el || !props.durationSec) return
  const rect = ev.currentTarget.getBoundingClientRect()
  const ratio = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width))
  el.currentTime = ratio * props.durationSec
  currentTime.value = el.currentTime
}

watch(
  () => props.gainDb,
  (v) => {
    if (gainNode) gainNode.gain.value = Math.pow(10, (v || 0) / 20)
  },
)
watch(
  () => props.src,
  () => {
    playing.value = false
    currentTime.value = 0
    error.value = ''
  },
)

onBeforeUnmount(() => {
  audioEl.value?.pause()
  try {
    sourceNode?.disconnect()
    gainNode?.disconnect()
    ctx?.close()
  } catch (_) {
    /* 忽略关闭错误 */
  }
})

defineExpose({ toggle, stop: () => audioEl.value?.pause() })
</script>

<template>
  <div class="player" :class="{ compact }">
    <button class="play-btn" :class="{ playing }" :disabled="loading" @click.stop="toggle">
      <span v-if="loading">…</span>
      <span v-else-if="playing">❚❚</span>
      <span v-else>▶</span>
    </button>

    <div class="wave" title="点击定位" @click.stop="seekTo">
      <div class="wave-bars">
        <span
          v-for="(p, i) in waveform"
          :key="i"
          class="bar"
          :class="{ past: i / (waveform.length || 1) <= progress }"
          :style="{ height: `${Math.max(6, p * 100)}%` }"
        />
        <span v-if="!waveform.length" class="wave-placeholder">波形加载中…</span>
      </div>
    </div>

    <span class="meta">{{ (durationSec || 0).toFixed(1) }}s</span>
    <span class="gain" :title="'试听时应用的增益（不改文件）'">{{ gainLabel }}</span>
    <span v-if="error" class="err" :title="error">⚠</span>

    <audio
      ref="audioEl"
      :src="src"
      preload="none"
      @ended="onEnded"
      @timeupdate="onTimeUpdate"
      @pause="onPause"
      @error="error = '音频加载失败'"
    />
  </div>
</template>

<style scoped>
.player {
  display: flex;
  align-items: center;
  gap: 8px;
}
.play-btn {
  flex: 0 0 auto;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--fg);
  cursor: pointer;
  font-size: 11px;
  line-height: 1;
  display: grid;
  place-items: center;
}
.play-btn.playing {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.play-btn:disabled {
  opacity: 0.6;
  cursor: progress;
}
.wave {
  flex: 1 1 auto;
  min-width: 40px;
  height: 24px;
  cursor: pointer;
  position: relative;
}
.wave-bars {
  display: flex;
  align-items: center;
  gap: 1px;
  height: 100%;
  overflow: hidden;
}
.bar {
  flex: 1 1 auto;
  min-width: 1px;
  background: var(--bar);
  border-radius: 1px;
}
.bar.past {
  background: var(--accent);
}
.wave-placeholder {
  font-size: 11px;
  color: var(--muted);
}
.meta,
.gain {
  flex: 0 0 auto;
  font-size: 11px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.gain {
  color: var(--accent);
  opacity: 0.75;
}
.err {
  color: var(--danger);
  cursor: help;
}
.compact .meta {
  display: none;
}
</style>
