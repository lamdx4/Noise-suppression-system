<script lang="ts" setup>
import { ref } from "vue";
const logs = ref<{ message: string; date: string }[]>([]);
const addListener = () => {
  eel.trigger_log_event()(function (data: any) {
    logs.value.push({ message: JSON.stringify(data), date: Date.now() });
  });
};
</script>
<template>
  <el-container
    ><el-row>
      <el-col>
        <h3>Console</h3>
        <template v-for="log in logs" :key="log.date">
          <div>{{ log.date }}: {{ log.message }}</div>
        </template>
      </el-col>
    </el-row>
  </el-container>
</template>
