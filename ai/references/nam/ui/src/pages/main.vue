<script lang="ts" setup>
import { ref, onMounted, defineExpose, onUnmounted, computed } from "vue";
import AudioVisualizer from "../components/AudioVisualizer.vue";
import TestAudio from "../components/TestAudio.vue";
import { useToast } from "vue-toastification";

const readyForReceive = ref(false);
const audioVisualizer = ref<null>(null);
const rawAudioVisualizer = ref<null>(null);

const reset = () => {
  readyForReceive.value = false;
  audioVisualizer.value?.reset();
  rawAudioVisualizer.value?.reset();
};

const showSpectrogram = () => {
  readyForReceive.value = false;
  audioVisualizer.value?.showSpectrogram();
  rawAudioVisualizer.value?.showSpectrogram();
};

const stopReceive = () => {
  readyForReceive.value = false;
};

const continueReceive = () => {
  readyForReceive.value = true;
};
</script>
<template>
  <el-container>
    <el-header style="font-size: 12px">
      <div class="toolbar">
        <el-button
          @click="reset"
          type="danger"
          large
          size="large"
          style="width: 150px"
          ><span>Đặt lại</span></el-button
        >
        <el-button
          @click="stopReceive"
          type="primary"
          large
          plain
          :disabled="!readyForReceive"
          size="large"
          style="width: 150px"
          ><span>Dừng</span></el-button
        >
        <el-button
          @click="continueReceive"
          type="primary"
          large
          plain
          :disabled="readyForReceive"
          size="large"
          style="width: 150px"
        >
          Bắt đầu
        </el-button>
        <el-button
          type="primary"
          large
          @click="showSpectrogram"
          plain
          :disabled="readyForReceive"
          size="large"
          style="width: 150px"
        >
          So sánh Spectogram
        </el-button>
      </div>
    </el-header>

    <el-main>
      <el-scrollbar>
        <AudioVisualizer
          ref="rawAudioVisualizer"
          :ready-for-receive="readyForReceive"
          :is-raw="true"
        />
        <AudioVisualizer
          ref="audioVisualizer"
          :ready-for-receive="readyForReceive"
          :is-raw="false"
        />
      </el-scrollbar>
    </el-main>
  </el-container>
</template>
