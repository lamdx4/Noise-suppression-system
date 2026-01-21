<template>
  <div style="padding: 16px">
    <div
      style="display: flex; justify-content: space-between; align-items: center"
    >
      <h2>Âm thanh test</h2>
      <el-button
        round
        @click="reset"
        type="primary"
        large
        style="width: 100px"
        size="large"
      >
        Reset
      </el-button>
    </div>

    <div ref="waveformContainer" class="waveform"></div>
    <div ref="waveformContainer2" class="waveform"></div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, onUnmounted } from "vue";
import WaveSurfer from "wavesurfer.js";

const waveformContainer = ref<HTMLElement | null>(null);
const wavesurfer = ref<WaveSurfer | null>(null);
const waveformContainer2 = ref<HTMLElement | null>(null);
const wavesurfer2 = ref<WaveSurfer | null>(null);
const data = ref<Float32Array>(new Float32Array(0));

const reset = async () => {
  if (wavesurfer.value) {
    await wavesurfer.value.playPause();
  }
};
const simulateRealtimeData = async () => {
  // 1. Tải file audio
  const buffer = await mp3ToAudioBuffer("/noise_audio.mp3");
  const audioData = buffer!.getChannelData(0); // Lấy dữ liệu kênh âm thanh

  // 2. Khởi tạo WaveSurfer nếu chưa có
  if (!wavesurfer.value) {
    wavesurfer.value = WaveSurfer.create({
      container: waveformContainer.value as HTMLElement,
      waveColor: "#4F46E5",
      progressColor: "#3730A3",
      interact: false,
      barWidth: 2,
      scrollParent: true, // Cho phép scroll ngang
      minPxPerSec: 100, // Độ rộng tối thiểu mỗi giây
      fillParent: false, // Không tự động co dãn
      height: 200,
      partialRender: true,
      normalize: false,
      autoCenter: true,
    });
  }

  const emptyBuffer = audioContext.createBuffer(
    1,
    audioData.length,
    buffer!.sampleRate
  );
  wavesurfer.value.loadDecodedBuffer(emptyBuffer);

  // 3. Cấu hình streaming
  const chunkSize = 1024; // Số sample mỗi lần thêm vào
  let position = 0;
  const updateInterval = 20; // ms (~20fps)

  const streamData = async () => {
    if (position >= audioData.length) {
      console.log("Đã thêm toàn bộ dữ liệu");
      return;
    }

    // 4. Lấy chunk dữ liệu từ audio
    const endPos = Math.min(position + chunkSize, audioData.length);
    const chunk = audioData.slice(position, endPos);

    // 5. Tạo mảng dữ liệu mới bằng cách nối thêm chunk
    const newData = new Float32Array(data.value.length + chunk.length);
    newData.set(data.value); // Copy dữ liệu cũ
    newData.set(chunk, data.value.length); // Thêm dữ liệu mới

    // 6. Cập nhật biến data
    data.value = newData;

    // 7. Tạo AudioBuffer từ data hiện tại
    const tempBuffer = audioContext.createBuffer(
      1,
      data.value.length,
      buffer!.sampleRate
    );
    tempBuffer.getChannelData(0).set(data.value);
    // 8. Cập nhật WaveSurfer
    await wavesurfer.value?.loadDecodedBuffer(tempBuffer);

    // 9. Di chuyển vị trí
    position = endPos;

    // 10. Lên lịch cho lần cập nhật tiếp theo
    setTimeout(streamData, updateInterval);
  };

  // Bắt đầu quá trình
  streamData();
};

onMounted(async () => {
  wavesurfer2.value = WaveSurfer.create({
    container: waveformContainer2.value as HTMLElement,
    waveColor: "#4F46E5",
    progressColor: "#3730A3",
    interact: false,
    barWidth: 2,
    scrollParent: true, // Cho phép scroll ngang
    minPxPerSec: 100, // Độ rộng tối thiểu mỗi giây
    fillParent: false, // Không tự động co dãn
    height: 200,
    partialRender: true,
    normalize: false,
    autoCenter: true,
  });
  const buffer = await mp3ToAudioBuffer("/noise_audio.mp3");
  wavesurfer2.value?.loadDecodedBuffer(buffer!);

  simulateRealtimeData();
});

const audioContext = new AudioContext();

async function mp3ToAudioBuffer(mp3URL) {
  try {
    const response = await fetch(mp3URL);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBuffer;
  } catch (error) {
    console.error("Error decoding or fetching the MP3:", error);
    return null;
  }
}
</script>

<style>
.waveform {
  margin-bottom: 2rem;
  background: #f3f4f6;
  border-radius: 0.5rem;
}
</style>
