<template>
  <div class="about-container">
    <h1 class="about-title">关于聚类分析系统</h1>

    <el-card class="about-card">
      <template #header>
        <div class="card-header">
          <span>项目介绍</span>
        </div>
      </template>
      <div class="card-content">
        <p>
          本系统是一个聚类分析平台，旨在提供简单易用的聚类分析工具。系统集成了多种经典聚类算法，可以对用户上传的数据集进行聚类分析，并提供可视化结果展示。同时，系统还支持使用聚类算法实现的图像分割功能，可以对用户上传的图片进行智能分割处理。
        </p>
      </div>
    </el-card>

    <el-card class="about-card">
      <template #header>
        <div class="card-header">
          <span>功能特点</span>
        </div>
      </template>
      <div class="card-content">
        <el-row :gutter="20">
          <el-col :span="12">
            <div class="feature-item">
              <el-icon><DataAnalysis /></el-icon>
              <h3>数据集聚类</h3>
              <p>
                支持多种格式数据集上传，提供K-Means、DBSCAN和基于桥点的密度聚类等多种算法选择，可自定义算法参数，生成聚类结果可视化图像。
              </p>
            </div>
          </el-col>
          <el-col :span="12">
            <div class="feature-item">
              <el-icon><Picture /></el-icon>
              <h3>图像分割</h3>
              <p>
                支持常见图片格式上传，提供基于聚类算法的分割、分水岭分割和GrabCut分割等多种图像分割算法，分割结果直观展示。
              </p>
              <p style="color: red"></p>
            </div>
          </el-col>
        </el-row>
      </div>
    </el-card>

    <el-card class="about-card">
      <template #header>
        <div class="card-header">
          <span>使用说明</span>
        </div>
      </template>
      <div class="card-content">
        <h3>数据集聚类</h3>
        <ol>
          <li>上传数据集文件（支持CSV、TXT等格式）</li>
          <li>选择聚类算法</li>
          <li>设置算法参数</li>
          <li>点击"开始聚类"按钮</li>
          <li>查看聚类结果图像，可下载保存</li>
        </ol>

        <h3>图像分割</h3>
        <ol>
          <li>上传图片文件（支持JPG、PNG等常见格式）</li>
          <li>选择分割算法</li>
          <li>点击"开始分割"按钮</li>
          <li>查看分割结果图像，可下载保存</li>
        </ol>
      </div>
    </el-card>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import { DataAnalysis, Picture } from "@element-plus/icons-vue";

export default defineComponent({
  name: "AboutView",
  components: {
    DataAnalysis,
    Picture,
  },
});
</script>

<style scoped>
.about-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.about-title {
  margin-bottom: 30px;
  color: #409eff;
  font-size: 28px;
  border-bottom: 2px solid #eaeaea;
  padding-bottom: 15px;
  text-align: center;
}

.about-card {
  margin-bottom: 30px;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  align-items: center;
  font-size: 18px;
  font-weight: bold;
  color: #303133;
}

.card-content {
  padding: 10px 0;
  line-height: 1.6;
  color: #606266;
}

.feature-item,
.tech-item {
  padding: 15px;
  background-color: #f9f9f9;
  border-radius: 8px;
  height: 100%;
  transition: all 0.3s;
}

.feature-item:hover,
.tech-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.feature-item h3,
.tech-item h3 {
  margin: 10px 0;
  color: #303133;
  font-size: 18px;
}

.feature-item .el-icon {
  font-size: 30px;
  color: #409eff;
  margin-bottom: 10px;
}

.tech-item ul {
  padding-left: 20px;
  margin: 10px 0;
}

.tech-item li {
  margin-bottom: 5px;
}

ol {
  padding-left: 20px;
  margin: 10px 0 20px;
}

ol li {
  margin-bottom: 8px;
}
</style>
