<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-content">
        <div class="logo">
          <router-link to="/">
            <el-icon><DataAnalysis /></el-icon>
            <span>聚类分析系统</span>
          </router-link>
        </div>
        <nav class="main-nav">
          <router-link to="/" class="nav-item">
            <el-icon><HomeFilled /></el-icon>
            <span>首页</span>
          </router-link>
          <router-link to="/dataset" class="nav-item">
            <el-icon><DataAnalysis /></el-icon>
            <span>数据集聚类</span>
          </router-link>
          <router-link to="/about" class="nav-item">
            <el-icon><InfoFilled /></el-icon>
            <span>关于</span>
          </router-link>
        </nav>
      </div>
    </header>

    <main class="app-main">
      <router-view />
    </main>

    <footer class="app-footer">
      <p>© 2025 聚类分析与图像分割系统 | 基于Vue3和Spring Boot开发</p>
    </footer>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import { DataAnalysis, HomeFilled, InfoFilled } from "@element-plus/icons-vue";

export default defineComponent({
  name: "App",
  components: {
    DataAnalysis,
    HomeFilled,
    InfoFilled,
  },
});
</script>

<style>
#app {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  min-height: 100vh;
}

body {
  margin: 0;
  padding: 0;
  background-color: #f5f7fa;
}

.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.app-header {
  background-color: #fff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logo a {
  display: flex;
  align-items: center;
  text-decoration: none;
  color: #409eff;
  font-size: 20px;
  font-weight: bold;
}

.logo .el-icon {
  font-size: 24px;
  margin-right: 8px;
}

.main-nav {
  display: flex;
  gap: 20px;
}

.nav-item {
  display: flex;
  align-items: center;
  text-decoration: none;
  color: #606266;
  font-weight: 500;
  padding: 8px 12px;
  border-radius: 4px;
  transition: all 0.3s;
}

.nav-item .el-icon {
  margin-right: 5px;
}

.nav-item:hover {
  color: #409eff;
  background-color: #ecf5ff;
}

.nav-item.router-link-active {
  color: #409eff;
  background-color: #ecf5ff;
}

.app-main {
  flex: 1;
  padding: 20px 0;
}

.app-footer {
  background-color: #fff;
  padding: 20px;
  text-align: center;
  color: #909399;
  font-size: 14px;
  border-top: 1px solid #ebeef5;
}
</style>
