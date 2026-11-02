<script setup>
/** 进度条（WebSocket 驱动，§6.5）。 */
import { computed } from 'vue'

const props = defineProps({
  job: { type: Object, default: null },
  compact: { type: Boolean, default: false },
})

const pct = computed(() => {
  if (!props.job) return 0
  if (!props.job.total) return props.job.status === 'done' ? 100 : 0
  return Math.min(100, Math.round((props.job.done / props.job.total) * 100))
})

const statusText = computed(() => {
  const j = props.job
  if (!j) return '未开始'
  const map = { running: '进行中', done: '完成', failed: '失败', cancelled: '已取消' }
  return map[j.status] || j.status
})

const cls = computed(() => (props.job ? props.job.status : 'idle'))
</script>

<template>
  <div class="progress" :class="[cls, { compact }]">
    <div class="row">
      <span class="status">{{ statusText }}</span>
      <span v-if="job && job.total" class="count">
        {{ job.done }} / {{ job.total }}
      </span>
      <span class="pct">{{ pct }}%</span>
    </div>
    <div class="track">
      <div class="fill" :style="{ width: `${pct}%` }" />
    </div>
    <p v-if="job && job.message" class="msg">{{ job.message }}</p>
    <p v-if="job && job.errors && job.errors.length" class="err">
      错误 {{ job.error_count || job.errors.length }} 条：{{ job.errors[0] }}
    </p>
  </div>
</template>

<style scoped>
.progress {
  font-size: 12px;
}
.row {
  display: flex;
  gap: 8px;
  align-items: baseline;
  color: var(--muted);
}
.status {
  font-weight: 600;
  color: var(--fg);
}
.count {
  font-variant-numeric: tabular-nums;
}
.pct {
  margin-left: auto;
  font-variant-numeric: tabular-nums;
}
.track {
  margin-top: 5px;
  height: 7px;
  border-radius: 4px;
  background: var(--chip);
  overflow: hidden;
}
.fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.25s ease;
}
.running .fill {
  background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 55%, #fff));
}
.done .fill {
  background: var(--ok);
}
.failed .fill {
  background: var(--danger);
}
.msg {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 11px;
}
.err {
  margin: 5px 0 0;
  color: var(--danger);
  font-size: 11px;
  word-break: break-all;
}
.compact .msg {
  display: none;
}
</style>
