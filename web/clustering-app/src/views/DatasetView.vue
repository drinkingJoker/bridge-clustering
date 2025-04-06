<template>
  <div class="dataset-container">
    <h1 class="main-title">数据集聚类分析</h1>

    <el-tabs v-model="activeTab" class="demo-tabs">
      <el-tab-pane label="数据集聚类" name="dataset">
        <div class="upload-section">
          <h3>上传数据集</h3>
          <el-upload
            class="upload-demo"
            drag
            action="#"
            :auto-upload="false"
            :on-change="handleDatasetFileChange"
            :limit="1"
            ref="datasetUploadRef"
          >
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              拖拽文件到此处或 <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                支持CSV、TXT、arff等格式的数据集文件
              </div>
            </template>
          </el-upload>
        </div>

        <div class="algorithm-section">
          <h3>选择聚类算法</h3>
          <el-select v-model="selectedAlgorithm" placeholder="请选择聚类算法">
            <el-option
              v-for="item in algorithms"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            >
            </el-option>
          </el-select>
        </div>

        <div class="params-section" v-if="selectedAlgorithm">
          <h3>算法参数设置</h3>
          <el-form :model="algorithmParams" label-width="120px">
            <template v-if="selectedAlgorithm === 'kmeans'">
              <el-form-item label="聚类数量(K)">
                <el-input-number
                  v-model="algorithmParams.k"
                  :min="2"
                  :max="20"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="最大迭代次数">
                <el-input-number
                  v-model="algorithmParams.maxIter"
                  :min="10"
                  :max="1000"
                  :step="10"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="初始化方法">
                <el-select v-model="algorithmParams.init">
                  <el-option label="k-means++" value="k-means++"></el-option>
                  <el-option label="随机" value="random"></el-option>
                </el-select>
              </el-form-item>
            </template>
            <template v-else-if="selectedAlgorithm === 'dbscan'">
              <el-form-item label="邻域半径(Eps)">
                <el-input-number
                  v-model="algorithmParams.eps"
                  :min="0.1"
                  :max="10"
                  :step="0.1"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="最小样本数(MinPts)">
                <el-input-number
                  v-model="algorithmParams.minPts"
                  :min="1"
                  :max="100"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="距离度量">
                <el-select v-model="algorithmParams.metric">
                  <el-option label="欧氏距离" value="euclidean"></el-option>
                  <el-option label="曼哈顿距离" value="manhattan"></el-option>
                  <el-option label="余弦距离" value="cosine"></el-option>
                </el-select>
              </el-form-item>
            </template>
            <template v-else-if="selectedAlgorithm === 'hierarchical'">
              <el-form-item label="聚类数量（1表示none）">
                <el-input-number
                  v-model="algorithmParams.clusters"
                  :min="2"
                  :max="20"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="连接方式">
                <el-select v-model="algorithmParams.linkage">
                  <el-option label="单连接" value="single"></el-option>
                  <el-option label="完全连接" value="complete"></el-option>
                  <el-option label="平均连接" value="average"></el-option>
                  <el-option label="Ward连接" value="ward"></el-option>
                </el-select>
              </el-form-item>
              <el-form-item label="距离度量">
                <el-select v-model="algorithmParams.metric">
                  <el-option label="欧氏距离" value="euclidean"></el-option>
                  <el-option label="曼哈顿距离" value="manhattan"></el-option>
                  <el-option label="余弦距离" value="cosine"></el-option>
                </el-select>
              </el-form-item>
            </template>
            <template v-else-if="selectedAlgorithm === 'spectral'">
              <el-form-item label="聚类数量">
                <el-input-number
                  v-model="algorithmParams.nClusters"
                  :min="2"
                  :max="20"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="相似度计算方式">
                <el-select v-model="algorithmParams.affinity">
                  <el-option
                    label="最近邻"
                    value="nearest_neighbors"
                  ></el-option>
                  <el-option label="径向基函数" value="rbf"></el-option>
                </el-select>
              </el-form-item>
              <el-form-item
                label="邻居数量"
                v-if="algorithmParams.affinity === 'nearest_neighbors'"
              >
                <el-input-number
                  v-model="algorithmParams.nNeighbors"
                  :min="5"
                  :max="50"
                ></el-input-number>
              </el-form-item>
            </template>
            <template v-else-if="selectedAlgorithm === 'bridge'">
              <el-form-item label="异常检测方法">
                <el-select v-model="algorithmParams.outlierDetection">
                  <el-option label="LOF" value="LOF"></el-option>
                  <el-option label="IF" value="IF"></el-option>
                </el-select>
              </el-form-item>
              <el-form-item label="异常比例">
                <el-input-number
                  v-model="algorithmParams.contamination"
                  :min="0.01"
                  :max="0.5"
                  :step="0.01"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="邻居数量(od)">
                <el-input-number
                  v-model="algorithmParams.n_neighbors_od"
                  :min="5"
                  :max="50"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="邻居数量">
                <el-input-number
                  v-model="algorithmParams.n_neighbors"
                  :min="5"
                  :max="50"
                ></el-input-number>
              </el-form-item>
            </template>
          </el-form>
        </div>

        <div class="action-section">
          <el-button
            type="primary"
            @click="startClustering"
            :disabled="!datasetFile || !selectedAlgorithm"
            >开始聚类</el-button
          >
        </div>

        <div class="result-section" v-if="clusteringResult">
          <h3>聚类结果</h3>
          <div class="result-image">
            <img :src="clusteringResult" alt="聚类结果图" />
          </div>

          <!-- 数据集信息展示 -->
          <div class="dataset-info" v-if="datasetInfo">
            <h4>数据集信息</h4>
            <el-descriptions :column="2" border>
              <el-descriptions-item label="样本数量">{{
                datasetInfo.sampleCount
              }}</el-descriptions-item>
              <el-descriptions-item label="维度">{{
                datasetInfo.dimensions
              }}</el-descriptions-item>
            </el-descriptions>

            <!-- 样本数据展示 -->
            <h4>样本数据</h4>
            <el-table :data="datasetInfo.samples" stripe style="width: 100%">
              <el-table-column prop="index" label="索引" width="80" />
              <el-table-column prop="x_y" label="数据">
                <!-- <template #default="scope">
                  <span v-if="scope.row.data === '...'" class="ellipsis">
                    ...
                  </span>
                  <span v-else>
                    {{
                      Array.isArray(scope.row.data)
                        ? scope.row.data.join(", ")
                        : scope.row.data
                    }}
                  </span>
                </template> -->
              </el-table-column>
              <!-- <el-table-column prop="cluster" label="簇" width="80" /> -->
            </el-table>
          </div>

          <el-button type="success" @click="downloadResult">下载结果</el-button>
        </div>
      </el-tab-pane>

      <el-tab-pane label="图像分割" name="image">
        <div class="upload-section">
          <h3>上传图片</h3>
          <el-upload
            class="upload-demo"
            drag
            action="#"
            :auto-upload="false"
            :on-change="handleImageFileChange"
            :limit="1"
            accept="image/*"
            ref="imageUploadRef"
          >
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              拖拽图片到此处或 <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">支持JPG、PNG等常见图片格式</div>
            </template>
          </el-upload>

          <div class="image-preview" v-if="imagePreview">
            <h4>预览图片</h4>
            <img
              :src="imagePreview"
              alt="预览图片"
              style="max-width: 100%; max-height: 300px"
            />
          </div>
        </div>

        <div class="algorithm-section">
          <h3>选择分割算法</h3>
          <el-select
            v-model="selectedSegmentAlgorithm"
            placeholder="请选择分割算法"
          >
            <el-option
              v-for="item in segmentAlgorithms"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            >
            </el-option>
          </el-select>
        </div>

        <div class="params-section" v-if="selectedSegmentAlgorithm">
          <h3>算法参数设置</h3>
          <el-form :model="segmentParams" label-width="120px">
            <template v-if="selectedSegmentAlgorithm === 'kmeans'">
              <el-form-item label="聚类数量(K)">
                <el-input-number
                  v-model="segmentParams.k"
                  :min="2"
                  :max="20"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="颜色空间">
                <el-select v-model="segmentParams.colorSpace">
                  <el-option label="RGB" value="rgb"></el-option>
                  <el-option label="HSV" value="hsv"></el-option>
                  <el-option label="LAB" value="lab"></el-option>
                </el-select>
              </el-form-item>
            </template>
            <template v-else-if="selectedSegmentAlgorithm === 'watershed'">
              <el-form-item label="标记距离">
                <el-input-number
                  v-model="segmentParams.distanceThreshold"
                  :min="1"
                  :max="100"
                  :step="1"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="平滑处理">
                <el-switch v-model="segmentParams.applySmoothing"></el-switch>
              </el-form-item>
            </template>
            <template v-else-if="selectedSegmentAlgorithm === 'grabcut'">
              <el-form-item label="迭代次数">
                <el-input-number
                  v-model="segmentParams.iterCount"
                  :min="1"
                  :max="10"
                  :step="1"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="模式">
                <el-select v-model="segmentParams.mode">
                  <el-option label="矩形模式" value="rect"></el-option>
                  <el-option label="掩码模式" value="mask"></el-option>
                </el-select>
              </el-form-item>
            </template>
            <template v-else-if="selectedSegmentAlgorithm === 'bridge'">
              <el-form-item label="聚类数量">
                <el-input-number
                  v-model="segmentParams.n_clusters"
                  :min="1"
                  :max="20"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="异常检测方法">
                <el-select v-model="segmentParams.outlierDetection">
                  <el-option label="LOF" value="LOF"></el-option>
                  <el-option label="IF" value="IF"></el-option>
                </el-select>
              </el-form-item>
              <el-form-item label="异常比例">
                <el-input-number
                  v-model="segmentParams.contamination"
                  :min="0.01"
                  :max="0.5"
                  :step="0.01"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="邻居数量(od)">
                <el-input-number
                  v-model="segmentParams.n_neighbors_od"
                  :min="5"
                  :max="50"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="邻居数量">
                <el-input-number
                  v-model="segmentParams.n_neighbors"
                  :min="5"
                  :max="50"
                ></el-input-number>
              </el-form-item>
              <el-form-item label="颜色空间">
                <el-select v-model="segmentParams.colorSpace">
                  <el-option label="RGB" value="rgb"></el-option>
                  <el-option label="HSV" value="hsv"></el-option>
                  <el-option label="LAB" value="lab"></el-option>
                </el-select>
              </el-form-item>
            </template>
          </el-form>
        </div>

        <div class="action-section">
          <el-button
            type="primary"
            @click="startSegmentation"
            :disabled="!imageFile || !selectedSegmentAlgorithm"
            >开始分割</el-button
          >
        </div>

        <div class="result-section" v-if="segmentationResult">
          <h3>分割结果</h3>
          <div class="result-image">
            <img :src="segmentationResult" alt="分割结果图" />
          </div>
          <el-button type="success" @click="downloadSegmentResult"
            >下载结果</el-button
          >
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted } from "vue";
import { UploadFilled } from "@element-plus/icons-vue";
import axios from "axios";
import { ElMessage } from "element-plus";
import { useRoute } from "vue-router";
import type { UploadInstance } from "element-plus";

export default defineComponent({
  name: "DatasetView",
  components: {
    UploadFilled,
  },
  setup() {
    const route = useRoute();
    const activeTab = ref("dataset");

    // 从URL查询参数中获取activeTab值
    onMounted(() => {
      if (route.query.activeTab) {
        activeTab.value = route.query.activeTab as string;
      }
    });

    const datasetUploadRef = ref<UploadInstance | null>(null);
    const imageUploadRef = ref<UploadInstance | null>(null);
    const datasetFile = ref(null);
    const imageFile = ref(null);
    const imagePreview = ref("");
    const selectedAlgorithm = ref("");
    const selectedSegmentAlgorithm = ref("");
    const clusteringResult = ref("");
    const segmentationResult = ref("");
    const datasetInfo = ref<{
      sampleCount: number;
      dimensions: number;
      samples: any[];
    } | null>(null);

    const algorithms = [
      { value: "kmeans", label: "K-Means聚类" },
      { value: "dbscan", label: "DBSCAN密度聚类" },
      { value: "hierarchical", label: "层次聚类" },
      { value: "bridge", label: "基于桥点的密度聚类" },
    ];

    const segmentAlgorithms = [
      { value: "kmeans", label: "基于kmeans算法的分割" },
      { value: "watershed", label: "分水岭分割" },
      { value: "grabcut", label: "GrabCut分割" },
      { value: "bridge", label: "基于桥点聚类算法的分割" },
    ];

    const segmentParams = ref({
      // K-Means分割参数
      k: 5,
      colorSpace: "rgb",

      // 分水岭分割参数
      distanceThreshold: 10,
      applySmoothing: true,

      // GrabCut分割参数
      iterCount: 5,
      mode: "rect",

      // 桥点密度聚类分割参数
      n_clusters: 4,
      outlierDetection: "LOF",
      contamination: 0.2,
      n_neighbors_od: 15,
      n_neighbors: 7,
    });

    const algorithmParams = ref({
      // K-Means参数
      k: 3,
      maxIter: 300,
      init: "k-means++",

      // DBSCAN参数
      eps: 0.5,
      minPts: 5,
      metric: "euclidean",

      // 层次聚类参数
      clusters: 3,
      linkage: "ward",

      // 谱聚类参数
      nClusters: 3,
      affinity: "nearest_neighbors",
      nNeighbors: 10,

      // 桥点密度聚类参数
      outlierDetection: "LOF",
      contamination: 0.2,
      n_neighbors_od: 15,
      n_neighbors: 7,
    });

    const handleDatasetFileChange = (file: any) => {
      // 清除之前的结果
      clusteringResult.value = "";
      datasetInfo.value = null;
      // 更新文件引用
      datasetFile.value = file.raw;
    };

    const handleImageFileChange = (file: any) => {
      // 清除之前的结果
      segmentationResult.value = "";
      // 更新文件引用
      imageFile.value = file.raw;
      // 创建预览
      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreview.value = e.target?.result as string;
      };
      reader.readAsDataURL(file.raw);
    };

    const startClustering = async () => {
      if (!datasetFile.value || !selectedAlgorithm.value) {
        ElMessage.warning("请上传数据集并选择聚类算法");
        return;
      }

      try {
        // 创建FormData对象
        const formData = new FormData();
        formData.append("file", datasetFile.value);
        formData.append("algorithm", selectedAlgorithm.value);

        // 添加算法参数
        if (selectedAlgorithm.value === "kmeans") {
          formData.append("k", algorithmParams.value.k.toString());
          formData.append("maxIter", algorithmParams.value.maxIter.toString());
          formData.append("init", algorithmParams.value.init);
        } else if (selectedAlgorithm.value === "dbscan") {
          formData.append("eps", algorithmParams.value.eps.toString());
          formData.append("minPts", algorithmParams.value.minPts.toString());
          formData.append("metric", algorithmParams.value.metric);
        } else if (selectedAlgorithm.value === "hierarchical") {
          formData.append(
            "clusters",
            algorithmParams.value.clusters.toString()
          );
          formData.append("linkage", algorithmParams.value.linkage);
          formData.append("metric", algorithmParams.value.metric);
        } else if (selectedAlgorithm.value === "spectral") {
          formData.append(
            "nClusters",
            algorithmParams.value.nClusters.toString()
          );
          formData.append("affinity", algorithmParams.value.affinity);
          if (algorithmParams.value.affinity === "nearest_neighbors") {
            formData.append(
              "nNeighbors",
              algorithmParams.value.nNeighbors.toString()
            );
          }
        } else if (selectedAlgorithm.value === "bridge") {
          formData.append(
            "outlierDetection",
            algorithmParams.value.outlierDetection
          );
          formData.append(
            "contamination",
            algorithmParams.value.contamination.toString()
          );
          formData.append(
            "n_neighbors_od",
            algorithmParams.value.n_neighbors_od.toString()
          );
          formData.append(
            "n_neighbors",
            algorithmParams.value.n_neighbors.toString()
          );
        }

        // 发送请求到后端
        // 注意：这里的URL需要替换为实际的后端API地址
        const response = await axios.post("/api/clustering", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          // 修改为JSON响应类型，不再是blob
          responseType: "json",
        });

        // 处理返回的JSON数据
        const result = response.data;
        console.log("response.data: ", result);
        // 设置数据集信息
        datasetInfo.value = {
          sampleCount: result.sampleCount,
          dimensions: result.dimensions,
          samples: result.samples,
        };
        console.log("datasetInfo.value: ", datasetInfo.value);

        // 将Base64图片数据转换为URL
        clusteringResult.value = `data:image/png;base64,${result.image}`;

        // 清除上传组件的文件列表，确保下次上传新文件
        if (datasetUploadRef.value) {
          datasetUploadRef.value.clearFiles();
        }

        ElMessage.success("聚类分析完成");
      } catch (error) {
        console.error("聚类分析失败:", error);
        ElMessage.error("聚类分析失败，请检查数据格式或稍后重试");
      }
    };

    const startSegmentation = async () => {
      if (!imageFile.value || !selectedSegmentAlgorithm.value) {
        ElMessage.warning("请上传图片并选择分割算法");
        return;
      }

      try {
        // 创建FormData对象
        const formData = new FormData();
        formData.append("file", imageFile.value);
        formData.append("algorithm", selectedSegmentAlgorithm.value);

        // 添加算法参数
        if (selectedSegmentAlgorithm.value === "kmeans") {
          formData.append("k", segmentParams.value.k.toString());
          formData.append("colorSpace", segmentParams.value.colorSpace);
        } else if (selectedSegmentAlgorithm.value === "watershed") {
          formData.append(
            "distanceThreshold",
            segmentParams.value.distanceThreshold.toString()
          );
          formData.append(
            "applySmoothing",
            segmentParams.value.applySmoothing.toString()
          );
        } else if (selectedSegmentAlgorithm.value === "grabcut") {
          formData.append(
            "iterCount",
            segmentParams.value.iterCount.toString()
          );
          formData.append("mode", segmentParams.value.mode);
        } else if (selectedSegmentAlgorithm.value === "bridge") {
          formData.append(
            "n_clusters",
            segmentParams.value.n_clusters.toString()
          );
          formData.append(
            "outlierDetection",
            segmentParams.value.outlierDetection
          );
          formData.append(
            "contamination",
            segmentParams.value.contamination.toString()
          );
          formData.append(
            "n_neighbors_od",
            segmentParams.value.n_neighbors_od.toString()
          );
          formData.append(
            "n_neighbors",
            segmentParams.value.n_neighbors.toString()
          );
          formData.append("colorSpace", segmentParams.value.colorSpace);
        }

        // 发送请求到后端
        // 注意：这里的URL需要替换为实际的后端API地址
        const response = await axios.post("/api/segmentation", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          responseType: "blob",
        });

        // 将返回的图片数据转换为URL
        const blob = new Blob([response.data], { type: "image/png" });
        segmentationResult.value = URL.createObjectURL(blob);

        // 清除上传组件的文件列表，确保下次上传新文件
        if (imageUploadRef.value) {
          imageUploadRef.value.clearFiles();
          // 清除预览图片
          imagePreview.value = "";
        }

        ElMessage.success("图像分割完成");
      } catch (error) {
        console.error("图像分割失败:", error);
        ElMessage.error("图像分割失败，请检查图片格式或稍后重试");
      }
    };

    const downloadResult = () => {
      if (clusteringResult.value) {
        const a = document.createElement("a");
        a.href = clusteringResult.value;
        a.download = "聚类结果.png";
        a.click();
      }
    };

    const downloadSegmentResult = () => {
      if (segmentationResult.value) {
        const a = document.createElement("a");
        a.href = segmentationResult.value;
        a.download = "分割结果.png";
        a.click();
      }
    };
    // const formatSamples = (samples: any[]) => {
    //   if (!samples || samples.length === 0) return [];

    //   if (samples.length <= 6) {
    //     return samples.map((sample, index) => ({
    //       index: index + 1,
    //       data: sample.data,
    //       // cluster: sample.cluster,
    //     }));
    //   }

    //   const formattedSamples = [
    //     ...samples.slice(0, 3).map((sample, index) => ({
    //       index: index + 1,
    //       data: sample.data,
    //       cluster: sample.cluster,
    //     })),
    //     { index: "...", data: "...", cluster: "..." },
    //     ...samples.slice(-3).map((sample, index) => ({
    //       index: samples.length - 2 + index,
    //       data: sample.data,
    //       // cluster: sample.cluster,
    //     })),
    //   ];

    //   return formattedSamples;
    // };
    return {
      activeTab,
      datasetFile,
      imageFile,
      imagePreview,
      selectedAlgorithm,
      selectedSegmentAlgorithm,
      algorithms,
      segmentAlgorithms,
      algorithmParams,
      segmentParams,
      clusteringResult,
      segmentationResult,
      datasetInfo,
      handleDatasetFileChange,
      handleImageFileChange,
      startClustering,
      startSegmentation,
      downloadResult,
      downloadSegmentResult,
      datasetUploadRef,
      imageUploadRef,
      // formatSamples,
    };
  },
});
</script>

<style scoped>
.dataset-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.main-title {
  margin-bottom: 30px;
  color: #409eff;
  font-size: 28px;
  border-bottom: 2px solid #eaeaea;
  padding-bottom: 15px;
}

.upload-section,
.algorithm-section,
.params-section,
.action-section,
.result-section {
  margin-bottom: 30px;
  background-color: #f9f9f9;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.upload-section h3,
.algorithm-section h3,
.params-section h3,
.action-section h3,
.result-section h3 {
  margin-top: 0;
  color: #303133;
  border-bottom: 1px solid #dcdfe6;
  padding-bottom: 10px;
  margin-bottom: 20px;
}

.result-image {
  margin: 20px 0;
  border: 1px solid #eee;
  padding: 10px;
  border-radius: 4px;
  background-color: #ffffff;
}

.result-image img {
  max-width: 100%;
  max-height: 500px;
  border-radius: 4px;
}

.image-preview {
  margin-top: 20px;
}

.action-section {
  display: flex;
  justify-content: center;
}

.action-section .el-button {
  padding: 12px 30px;
  font-size: 16px;
}

.el-upload {
  width: 100%;
}

.el-upload-dragger {
  width: 100%;
  height: 180px;
}

.el-select {
  width: 100%;
}
</style>


