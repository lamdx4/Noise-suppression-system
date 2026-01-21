<script setup>
import Main from "./layouts/main.vue";
import { onMounted } from "vue";

onMounted(() => {
  eel.on_loaded();
});
</script>

<template>
  <div>
    <Main />
  </div>
</template>
