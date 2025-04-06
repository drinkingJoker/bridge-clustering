<template>
  <div class="home-container">
    <div class="hero-section">
      <h1 class="main-title">聚类分析与图像分割系统</h1>
      <p class="subtitle">
        基于Python实现的聚类算法，提供数据集聚类分析与图像智能分割功能
      </p>

      <div class="action-buttons">
        <router-link to="/dataset" class="action-button primary">
          <el-icon><DataAnalysis /></el-icon>
          开始使用
        </router-link>
        <router-link to="/about" class="action-button secondary">
          <el-icon><InfoFilled /></el-icon>
          了解更多
        </router-link>
      </div>
    </div>

    <div class="features-section">
      <h2 class="section-title">主要功能</h2>

      <el-row :gutter="30">
        <el-col :xs="24" :sm="12">
          <div class="feature-card">
            <div class="feature-icon">
              <el-icon><DataAnalysis /></el-icon>
            </div>
            <h3>数据集聚类分析</h3>
            <p>
              上传您的数据集，选择合适的聚类算法，系统将自动进行聚类分析并生成可视化结果。支持K-Means、DBSCAN和基于桥点的密度聚类等多种算法。
            </p>
            <router-link to="/dataset?activeTab=dataset" class="feature-link"
              >立即尝试 <el-icon><ArrowRight /></el-icon
            ></router-link>
          </div>
        </el-col>

        <el-col :xs="24" :sm="12">
          <div class="feature-card">
            <div class="feature-icon">
              <el-icon><Picture /></el-icon>
            </div>
            <h3>图像分割</h3>
            <p>
              上传您的图片，选择合适的分割算法，系统将自动进行图像分割并展示结果。支持基于聚类算法的分割、分水岭分割和GrabCut分割等多种算法。
            </p>
            <router-link to="/dataset?activeTab=image" class="feature-link"
              >立即尝试 <el-icon><ArrowRight /></el-icon
            ></router-link>
          </div>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import {
  DataAnalysis,
  Picture,
  InfoFilled,
  ArrowRight,
} from "@element-plus/icons-vue";

export default defineComponent({
  name: "HomeView",
  components: {
    DataAnalysis,
    Picture,
    InfoFilled,
    ArrowRight,
  },
});
</script>

<style scoped>
.home-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.hero-section {
  text-align: center;
  padding: 60px 20px;
  background-color: #f9f9f9;
  border-radius: 8px;
  margin-bottom: 40px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.main-title {
  font-size: 36px;
  color: #409eff;
  margin-bottom: 20px;
}

.subtitle {
  font-size: 18px;
  color: #606266;
  margin-bottom: 40px;
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 20px;
}

.action-button {
  display: inline-flex;
  align-items: center;
  padding: 12px 24px;
  border-radius: 4px;
  font-size: 16px;
  font-weight: bold;
  text-decoration: none;
  transition: all 0.3s;
}

.action-button .el-icon {
  margin-right: 8px;
}

.action-button.primary {
  background-color: #409eff;
  color: white;
}

.action-button.primary:hover {
  background-color: #66b1ff;
  transform: translateY(-2px);
}

.action-button.secondary {
  background-color: #ecf5ff;
  color: #409eff;
  border: 1px solid #d9ecff;
}

.action-button.secondary:hover {
  background-color: #d9ecff;
  transform: translateY(-2px);
}

.section-title {
  text-align: center;
  margin-bottom: 30px;
  color: #303133;
  font-size: 24px;
}

.feature-card {
  background-color: #f9f9f9;
  border-radius: 8px;
  padding: 30px;
  height: 100%;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  transition: all 0.3s;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.feature-icon {
  font-size: 40px;
  color: #409eff;
  margin-bottom: 20px;
}

.feature-card h3 {
  font-size: 20px;
  color: #303133;
  margin-bottom: 15px;
}

.feature-card p {
  color: #606266;
  margin-bottom: 20px;
  line-height: 1.6;
}

.feature-link {
  display: inline-flex;
  align-items: center;
  color: #409eff;
  font-weight: bold;
  text-decoration: none;
}

.feature-link .el-icon {
  margin-left: 5px;
  transition: transform 0.3s;
}

.feature-link:hover .el-icon {
  transform: translateX(3px);
}
</style>
