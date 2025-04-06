package com.example.clusteringbackend.service.impl;

import com.example.clusteringbackend.service.SegmentationService;
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
import java.util.Map;
import java.util.UUID;

@Service
public class SegmentationServiceImpl implements SegmentationService {

    @Value("${app.upload.dir}")
    private String uploadDir;

    @Value("${app.python.script.dir}")
    private String pythonScriptDir;

    @Autowired
    private PythonScriptRunner pythonScriptRunner;

    @Override
    public byte[] performSegmentation(MultipartFile file, String algorithm, Map<String, String> params)
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
        String imageFilename = fileId + fileExtension;
        String outputFilename = fileId + "_result.png";

        // 保存上传的图片
        Path imagePath = uploadPath.resolve(imageFilename);
        Files.copy(file.getInputStream(), imagePath);

        // 构建Python脚本路径
        String scriptPath = Paths.get(pythonScriptDir, "segmentation.py").toString();

        // 构建命令参数
        StringBuilder commandArgs = new StringBuilder();
        commandArgs.append(scriptPath).append(" ")
                .append(imagePath.toString()).append(" ")
                .append(uploadPath.resolve(outputFilename).toString()).append(" ")
                .append(algorithm);

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

        // 清理临时文件
        Files.deleteIfExists(imagePath);
        Files.deleteIfExists(outputFile.toPath());

        return imageBytes;
    }
}