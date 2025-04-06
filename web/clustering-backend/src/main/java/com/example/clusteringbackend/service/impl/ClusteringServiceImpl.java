package com.example.clusteringbackend.service.impl;

import com.example.clusteringbackend.dao.po.ClusteringResult;
import com.example.clusteringbackend.dao.po.Point;
import com.example.clusteringbackend.service.ClusteringService;
import com.example.clusteringbackend.util.PythonScriptRunner;
import org.apache.commons.io.FileUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;
import org.json.JSONObject;
import org.json.JSONArray;

@Service
public class ClusteringServiceImpl implements ClusteringService {

    @Value("${app.upload.dir}")
    private String uploadDir;

    @Value("${app.python.script.dir}")
    private String pythonScriptDir;

    @Autowired
    private PythonScriptRunner pythonScriptRunner;

    @Override
    public ClusteringResult performClustering(MultipartFile file, String algorithm, Map<String, String> params)
            throws IOException, Exception {
        // 创建上传目录
        Path uploadPath = Paths.get(uploadDir);
        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
        }

        // 生成唯一文件名
        String fileId = UUID.randomUUID().toString();
        String originalFilename = file.getOriginalFilename();
        String fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
        String datasetFilename = fileId + fileExtension;
        String outputFilename = fileId + "_result.png";
        String jsonFilename = fileId + "_data.json";

        // 保存上传的文件
        Path datasetPath = uploadPath.resolve(datasetFilename);
        Files.copy(file.getInputStream(), datasetPath);

        // 构建Python脚本路径
        String scriptPath = Paths.get(pythonScriptDir, "clustering.py").toString();

        // 构建命令参数
        StringBuilder commandArgs = new StringBuilder();
        commandArgs.append(scriptPath).append(" ")
                .append(datasetPath.toString()).append(" ")
                .append(uploadPath.resolve(outputFilename).toString()).append(" ")
                .append(algorithm).append(" ")
                .append(uploadPath.resolve(jsonFilename).toString());

        // 添加算法特定参数
        for (Map.Entry<String, String> entry : params.entrySet()) {
            if (!entry.getKey().equals("file") && !entry.getKey().equals("algorithm")) {
                commandArgs.append(" --").append(entry.getKey()).append(" ").append(entry.getValue());
            }
        }

        // 执行Python脚本
        pythonScriptRunner.runPythonScript(commandArgs.toString());

        // 读取结果图片
        File outputFile = uploadPath.resolve(outputFilename).toFile();
        byte[] imageBytes = FileUtils.readFileToByteArray(outputFile);

        // 读取JSON数据文件
        File jsonFile = uploadPath.resolve(jsonFilename).toFile();
//        Map<String, Object> result = new HashMap<>();
        ClusteringResult clusteringResult = new ClusteringResult();

        if (jsonFile.exists()) {
            StringBuilder jsonContent = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new FileReader(jsonFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    jsonContent.append(line);
                }
            }

            // 解析JSON数据
            JSONObject jsonData = new JSONObject(jsonContent.toString());

            // 添加数据集信息
//            result.put("sampleCount", jsonData.getInt("sampleCount"));
//            result.put("dimensions", jsonData.getInt("dimensions"));
            clusteringResult.setSampleCount(jsonData.getInt("sampleCount"));
            clusteringResult.setDimensions(jsonData.getInt("dimensions"));


            // 添加样本数据
            if (jsonData.has("samples")) {
                JSONArray samplesArray = jsonData.getJSONArray("samples");
//                Object[] samples = new Object[samplesArray.length()];
                for (int i = 0; i < samplesArray.length(); i++) {
//                    samples[i] = samplesArray.get(i);
                    JSONObject sample =  (JSONObject) samplesArray.get(i);
                    // 将sample中的index，data转换为point,
                    // 其中index是int，需要转换为字符串，也可能直接就是字符串"..."，所以需要先判断是int还是字符串
                    // 其中data是一个数组，需要转换为字符串，data也可能为"..."，直接就是字符串，所以需要先判断是数组还是字符串
                    String index;
                    if (sample.get("index") instanceof Integer) {
                        index = String.valueOf(sample.getInt("index"));
                    } else {
                        index = sample.getString("index");
                    }
                    String x_y;
                    if (sample.get("data") instanceof JSONArray) {
                        JSONArray dataArray = sample.getJSONArray("data");
                        x_y = dataArray.toString();
                    } else {
                        x_y = sample.getString("data");
                    }
                    Point point = new Point(index, x_y);
                    clusteringResult.getSamples().add(point);
                }
//                result.put("samples", samples);
            }
        }

        // 将图片转换为Base64编码
        String base64Image = Base64.getEncoder().encodeToString(imageBytes);
//        result.put("image", base64Image);
        clusteringResult.setImage(base64Image);

        // 清理临时文件
        Files.deleteIfExists(datasetPath);
        Files.deleteIfExists(outputFile.toPath());
        Files.deleteIfExists(jsonFile.toPath());

        return clusteringResult;
    }
}