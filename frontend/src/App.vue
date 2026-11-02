<script setup>
/**
 * 应用外壳（方案 §6.4 布局）：
 *   顶栏：线路切换在左侧筛选面板里，这里放统计 / 导入 / 重建向量入口
 */
import { computed, onMounted, ref } from 'vue'
import { api } from './api/client'
import ImportView from './views/ImportView.vue'
import SearchView from './views/SearchView.vue'
import StatsView from './views/StatsView.vue'

const meta = ref(null)
const stats = ref(null)
const tab = ref('search') // search | import | stats
const startupError = ref('')
const booting = ref(true)

async function refreshStats() {
  try {
    stats.value = await api.stats()
  } catch (err) {
    startupError.value = err.message
  }
}

onMounted(async () => {
  try {
    meta.value = await api.meta()
  } catch (err) {
    startupError.value = `无法连接后端：${err.message}`
  } finally {
    booting.value = false
  }
  await refreshStats()
})

const backendLabel = computed(() => {
  if (!meta.value) return ''
  const e = meta.value.encoder
  const name = e.backend === 'sentence-transformers' ? e.model.split('/').pop() : '哈希兜底（无语义）'
  return `${name} · ${e.dim} 维`
})

const totalLabel = computed(() => {
  if (!stats.value) return '—'
  return `${stats.value.total} 条 / ${(stats.value.total_duration_sec / 60).toFixed(0)} 分钟`
})
</script>

<template>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <span class="logo">◉</span>
        <span class="title">语音素材库</span>
        <span class="tagline">IndexTTS · SeedVC</span>
      </div>

      <nav class="tabs">
        <button :class="{ on: tab === 'search' }" @click="tab = 'search'">检索</button>
        <button :class="{ on: tab === 'import' }" @click="tab = 'import'">导入</button>
        <button :class="{ on: tab === 'stats' }" @click="tab = 'stats'">统计</button>
      </nav>

      <div class="spacer" />

      <div class="info" :title="meta ? `根目录：${meta.root}` : ''">
        <span class="pill">{{ totalLabel }}</span>
        <span class="pill dim">{{ backendLabel }}</span>
      </div>
    </header>

    <div v-if="booting" class="boot">正在连接后端…</div>
    <div v-else-if="startupError" class="boot error">
      {{ startupError }}
      <p class="hint">
        后端未启动？在 <code>backend/</code> 目录运行
        <code>uvicorn server:app --host 127.0.0.1 --port 8000</code>。
      </p>
    </div>

    <SearchView
      v-else-if="tab === 'search'"
      :meta="meta"
      :stats="stats"
      @changed="refreshStats"
    />
    <ImportView v-else-if="tab === 'import'" :meta="meta" @changed="refreshStats" />
    <StatsView v-else :stats="stats" :meta="meta" @changed="refreshStats" />
  </div>
</template>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  min-height: 0;
}
.topbar {
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 0 16px;
  height: 48px;
  flex: 0 0 auto;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
}
.brand {
  display: flex;
  align-items: baseline;
  gap: 7px;
}
.logo {
  color: var(--accent);
  font-size: 15px;
}
.title {
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.3px;
}
.tagline {
  font-size: 11px;
  color: var(--muted);
}
.tabs {
  display: flex;
  gap: 2px;
}
.tabs button {
  font: inherit;
  font-size: 12.5px;
  padding: 5px 13px;
  border-radius: 6px;
  border: 1px solid transparent;
  background: none;
  color: var(--muted);
  cursor: pointer;
}
.tabs button.on {
  background: var(--chip);
  color: var(--fg);
  border-color: var(--border);
}
.spacer {
  flex: 1 1 auto;
}
.info {
  display: flex;
  gap: 8px;
}
.pill {
  font-size: 11px;
  padding: 3px 9px;
  border-radius: 10px;
  background: var(--chip);
  color: var(--fg);
  border: 1px solid var(--border);
}
.pill.dim {
  color: var(--muted);
}
.boot {
  flex: 1 1 auto;
  display: grid;
  place-content: center;
  gap: 10px;
  color: var(--muted);
  font-size: 13px;
  text-align: center;
}
.boot.error {
  color: var(--danger);
}
.boot .hint {
  color: var(--muted);
  font-size: 12px;
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--chip);
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 11.5px;
}
</style>
