<template>
  <div style="padding-right: 16px; padding-left: 16px">
    <div
      style="display: flex; justify-content: space-between; align-items: center"
    >
      <h2 v-if="isRaw">Âm thanh chưa qua xử lý</h2>
      <h2 v-else>Âm thanh đã xử lý</h2>
      <el-button
        round
        @click="play"
        type="primary"
        large
        style="width: 100px"
        :disabled="readyForReceive"
        size="large"
      >
        Phát
      </el-button>
    </div>

    <div ref="waveformContainer" class="waveform"></div>
    <div ref="timeline"></div>
    <div ref="waveSpectrogram" class="waveSpectrogram"></div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, onUnmounted } from "vue";
import WaveSurfer from "wavesurfer.js";
import TimelinePlugin from "wavesurfer.js/dist/plugin/wavesurfer.timeline.js";
import Spectrogram from "wavesurfer.js/dist/plugin/wavesurfer.spectrogram.js";
import colormap from "colormap";
const props = defineProps({
  readyForReceive: {
    type: Boolean,
    default: false,
  },
  isRaw: {
    type: Boolean,
    default: false,
  },
});
const SAMPLE_RATE = 8000; // Tần số mẫu âm thanh
const sampleRate = SAMPLE_RATE; // Tần số mẫu âm thanh
const CHUNK_SIZE = 1024;
const FIXED_MAX_AMPLITUDE = 0.8;
const wavesurfer = ref<WaveSurfer | null>(null);
const timeline = ref(null);
const waveSpectrogram = ref<HTMLElement | null>(null);
const waveformContainer = ref<HTMLElement | null>(null);
const audioContext = ref<AudioContext | null>(null);
const audioChunks = ref<Float32Array>(new Float32Array());
const isInitialized = ref(false);

const reset = () => {
  if (wavesurfer.value) {
    audioChunks.value = new Float32Array();
    wavesurfer.value?.empty();
    isInitialized.value = false;

    if (wavesurfer.value.getActivePlugins().spectrogram) {
      wavesurfer.value?.destroyPlugin("spectrogram");
    }
  }
};

const showSpectrogram = async () => {
  if (!wavesurfer.value) return;
  if (wavesurfer.value.getActivePlugins().spectrogram) {
    wavesurfer.value?.destroyPlugin("spectrogram");
  }

  wavesurfer.value.addPlugin(
    Spectrogram.create({
      wavesurfer: wavesurfer.value,
      container: waveSpectrogram.value as HTMLElement,
      labels: true,
      frequencyMax: 4000,
      frequencyMin: 0,
      fftSamples: 256,
      deferInit: true,
      colorMap: colormap({
        colormap: "summer",
        nshades: 256,
        format: "float",
      }),
    })
  );
  wavesurfer.value?.initPlugin("spectrogram");
};

const play = async () => {
  if (!wavesurfer.value) return;
  wavesurfer.value.stop();
  wavesurfer.value.seekTo(0);
  await wavesurfer.value.play();
};

const loadBuffer = async () => {
  if (!audioContext.value || audioChunks.value.length === 0) return null;

  const buffer = audioContext.value.createBuffer(
    1,
    audioChunks.value.length,
    sampleRate
  );
  buffer.getChannelData(0).set(audioChunks.value);

  wavesurfer.value?.empty();
  await wavesurfer.value?.loadDecodedBuffer(buffer);

  waveformContainer.value?.parentElement?.scrollTo({
    left: waveformContainer.value.scrollWidth,
    behavior: "smooth",
  });
};

onMounted(() => {
  audioContext.value = new AudioContext();

  wavesurfer.value = WaveSurfer.create({
    container: waveformContainer.value as HTMLElement,
    waveColor: "#4F46E5",
    progressColor: "#3730A3",
    interact: false,
    scrollParent: true, // Cho phép scroll ngang
    minPxPerSec: 100, // Độ rộng tối thiểu mỗi giây
    fillParent: false, // Không tự động co dãn
    height: 128,
    barWidth: 2,
    partialRender: true,
    normalize: false,
    autoCenter: true,
    plugins: [
      TimelinePlugin.create({
        container: timeline.value,
        primaryColor: "#4F46E5",
      }),
    ],
  });

  // Cố định zoom ban đầu
  wavesurfer.value.on("ready", () => {
    isInitialized.value = true;
  });

  if (props.isRaw) {
    // @ts-ignore
    eel.expose(receiveBytes, "receive_bytes_raw");
  } else {
    // @ts-ignore
    eel.expose(receiveBytes, "receive_bytes");
  }
  function receiveBytes(data: number[]) {
    if (!props.readyForReceive) return;
    if (wavesurfer.value?.getActivePlugins().spectrogram) {
      wavesurfer.value?.destroyPlugin("spectrogram");
    }

    const newData = new Float32Array(data);
    const merged = new Float32Array(audioChunks.value.length + newData.length);
    merged.set(audioChunks.value);
    merged.set(newData, audioChunks.value.length);
    audioChunks.value = merged;

    loadBuffer();
  }
});

onUnmounted(() => {
  wavesurfer.value?.destroy();
  if (audioContext.value?.state !== "closed") {
    audioContext.value?.close();
  }
});

defineExpose({
  reset,
  showSpectrogram,
});
</script>

<style>
.waveform {
  background: #f3f4f6;
  border-radius: 0.5rem;
}
</style>
