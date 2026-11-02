<script setup>
/**
 * 单条结果：播放 / 星标改强度 / ✓ 改已验证 / 描述编辑 / 勾选导出。
 * 行内更新，不重载列表（§6.4 交互要点 1–4）。
 */
import { computed, ref } from 'vue'
import AudioPlayer from './AudioPlayer.vue'

const props = defineProps({
  clip: { type: Object, required: true },
  selected: { type: Boolean, default: false },
  waveform: { type: Array, default: () => [] },
  busy: { type: Boolean, default: false },
})

const emit = defineEmits(['select', 'patch', 'play', 'describe', 'delete'])

const editing = ref(false)
const draft = ref('')
const saving = ref(false)

const genderMark = computed(() => {
  if (props.clip.gender === 'male') return '♂'
  if (props.clip.gender === 'female') return '♀'
  return '·'
})

const stars = computed(() => {
  const n = props.clip.intensity || 0
  return Array.from({ length: 5 }, (_, i) => i < n)
})

const isSpeech = computed(() => props.clip.clip_kind === 'speech')

function toggleSelect() {
  emit('select', props.clip.id, !props.selected)
}

function setIntensity(n) {
  // 再点同一颗星 = 清零，方便撤销
  const next = props.clip.intensity === n ? 0 : n
  emit('patch', props.clip.id, { intensity: next })
}

function toggleVerified() {
  emit('patch', props.clip.id, { verified: props.clip.verified ? 0 : 1 })
}

function startEdit() {
  draft.value = props.clip.description
  editing.value = true
}

async function saveEdit() {
  const text = draft.value.trim()
  editing.value = false
  if (!text || text === props.clip.description) return
  saving.value = true
  try {
    emit('describe', props.clip.id, text)
  } finally {
    saving.value = false
  }
}

function cancelEdit() {
  editing.value = false
  draft.value = ''
}
</script>

<template>
  <div class="card" :class="{ selected, busy }">
    <label class="pick" :title="'选中用于导出'">
      <input type="checkbox" :checked="selected" @change="toggleSelect" />
    </label>

    <div class="body">
      <div class="head">
        <span class="id">#{{ clip.id }}</span>
        <span class="cat">{{ clip.category }}</span>
        <span class="g">{{ genderMark }}</span>
        <span class="dur">{{ (clip.duration_sec || 0).toFixed(1) }}s</span>
        <span class="stars" :title="'听感强度（点击设置 1-5，再点清零）'">
          <button
            v-for="(on, i) in stars"
            :key="i"
            class="star"
            :class="{ on }"
            @click.stop="setIntensity(i + 1)"
          >★</button>
        </span>
        <button
          class="tag verified"
          :class="{ on: clip.verified }"
          :title="clip.verified ? '已验证（点击取消）' : '标记为已验证'"
          @click.stop="toggleVerified"
        >✓</button>
        <span v-if="clip.commercial_ok" class="tag ok" title="可商用">商用</span>
        <span v-else class="tag nc" :title="`许可：${clip.license}`">非商用</span>
        <span v-if="clip.score" class="score" :title="'语义相似度'">
          {{ clip.score.toFixed(3) }}
        </span>
      </div>

      <div v-if="!editing" class="desc" :title="'点击编辑描述（保存后自动重建向量）'" @click="startEdit">
        {{ clip.description }}
        <span v-if="saving" class="saving">重建向量中…</span>
      </div>
      <div v-else class="desc-edit" @click.stop>
        <textarea v-model="draft" rows="2" @keydown.ctrl.enter="saveEdit" @keydown.esc="cancelEdit" />
        <div class="edit-actions">
          <button class="primary" @click="saveEdit">保存</button>
          <button @click="cancelEdit">取消</button>
          <span class="hint">Ctrl+Enter 保存 · Esc 取消</span>
        </div>
      </div>

      <div v-if="clip.transcript" class="transcript" :title="'台词'">“{{ clip.transcript }}”</div>

      <div class="foot">
        <AudioPlayer
          :src="clip.audio_url"
          :gain-db="clip.gain_db || 0"
          :waveform="waveform"
          :duration-sec="clip.duration_sec"
          compact
          @play="emit('play', clip.id)"
        />
        <span class="path" :title="clip.filepath">{{ clip.filepath }}</span>
      </div>

      <div v-if="clip.notes" class="notes" :title="'导入时的自动标记'">⚠ {{ clip.notes }}</div>
      <div v-if="!isSpeech" class="kind-hint">副语言 · SeedVC</div>
    </div>
  </div>
</template>

<style scoped>
.card {
  display: flex;
  gap: 10px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
  transition: border-color 0.15s;
}
.card.selected {
  border-color: var(--accent);
}
.card.busy {
  opacity: 0.65;
}
.pick {
  padding-top: 2px;
  cursor: pointer;
}
.pick input {
  cursor: pointer;
}
.body {
  flex: 1 1 auto;
  min-width: 0;
}
.head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12px;
}
.id {
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.cat {
  font-weight: 600;
  color: var(--fg);
  background: var(--chip);
  border-radius: 4px;
  padding: 1px 6px;
}
.g {
  color: var(--muted);
  font-size: 14px;
}
.dur {
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.stars {
  display: inline-flex;
  gap: 1px;
}
.star {
  background: none;
  border: none;
  padding: 0 1px;
  font-size: 13px;
  color: var(--bar);
  cursor: pointer;
  line-height: 1;
}
.star.on {
  color: #f5b301;
}
.tag {
  font-size: 11px;
  border-radius: 4px;
  padding: 1px 5px;
  border: 1px solid var(--border);
  background: none;
  color: var(--muted);
  cursor: default;
}
.tag.verified {
  cursor: pointer;
}
.tag.verified.on {
  color: #fff;
  background: var(--ok);
  border-color: var(--ok);
}
.tag.ok {
  color: var(--ok);
  border-color: color-mix(in srgb, var(--ok) 40%, transparent);
}
.tag.nc {
  color: var(--warn);
  border-color: color-mix(in srgb, var(--warn) 40%, transparent);
}
.score {
  margin-left: auto;
  font-size: 11px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.desc {
  margin-top: 6px;
  font-size: 13px;
  line-height: 1.5;
  cursor: text;
  border-radius: 4px;
  padding: 2px 4px;
  margin-left: -4px;
}
.desc:hover {
  background: var(--chip);
}
.saving {
  font-size: 11px;
  color: var(--accent);
  margin-left: 6px;
}
.desc-edit textarea {
  width: 100%;
  margin-top: 6px;
  font: inherit;
  font-size: 13px;
  line-height: 1.5;
  padding: 6px 8px;
  border-radius: 6px;
  border: 1px solid var(--accent);
  background: var(--bg);
  color: var(--fg);
  resize: vertical;
}
.edit-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
}
.edit-actions button {
  font-size: 12px;
  padding: 3px 10px;
  border-radius: 5px;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--fg);
  cursor: pointer;
}
.edit-actions button.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.hint {
  font-size: 11px;
  color: var(--muted);
}
.transcript {
  margin-top: 4px;
  font-size: 12px;
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.foot {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.path {
  font-size: 10px;
  color: var(--muted);
  opacity: 0.6;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 40%;
}
.notes {
  margin-top: 6px;
  font-size: 11px;
  color: var(--warn);
}
.kind-hint {
  margin-top: 4px;
  font-size: 10px;
  color: var(--muted);
  opacity: 0.7;
}
</style>
