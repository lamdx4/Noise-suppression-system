import Main from "../pages/main.vue";
import Console from "../pages/console.vue";
import { computed, ref } from "vue";
const useNavigation = () => {
  const activeItemKey = ref("main");
  const changeActiveItemByKey = (key = "") => (activeItemKey.value = key);
  const items = [
    {
      key: "main",
      label: "Main",
      component: Main,
    },
    {
      key: "console",
      label: "Console",
      component: Console,
    },
  ];
  const renderComponent = computed(
    () => items.find((item) => item.key === activeItemKey.value).component
  );
  return { items, renderComponent, activeItemKey, changeActiveItemByKey };
};

export default useNavigation;
