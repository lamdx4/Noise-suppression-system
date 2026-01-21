<script lang="ts" setup>
import useNavigation from "../composables/useNavigation";
const { items, changeActiveItemByKey, renderComponent, activeItemKey } =
  useNavigation();
const onSelectNavigationItem = (index: string) => {
  const key = items[index].key;
  changeActiveItemByKey(key);
};
</script>
<template>
  <el-container class="layout-container" style="height: 100vh">
    <el-aside width="200px">
      <el-scrollbar>
        <el-menu
          :default-active="
            items.findIndex((item) => item.key === activeItemKey)
          "
          @select="onSelectNavigationItem"
        >
          <template v-for="(item, index) in items" :key="item.key">
            <el-menu-item :index="index">
              {{ item.label }}
            </el-menu-item>
          </template>
        </el-menu>
      </el-scrollbar>
    </el-aside>

    <component :is="renderComponent" />
  </el-container>
</template>

<style>
.layout-container .el-header {
  position: relative;
  background-color: var(--el-color-primary-light-7);
  color: var(--el-text-color-primary);
}
.layout-container .el-aside {
  color: var(--el-text-color-primary);
}
.layout-container .el-menu {
  border-right: none;
}
.layout-container .el-main {
  padding: 0;
}
.layout-container .toolbar {
  display: inline-flex;
  justify-content: end;
  width: 100%;
  align-items: center;
  height: 100%;
}
</style>
