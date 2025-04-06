package com.example.clusteringbackend.controller;

import com.example.clusteringbackend.dao.po.ClusteringResult;
import com.example.clusteringbackend.service.ClusteringService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;
import java.util.HashMap;

@RestController
public class ClusteringController {

    @Autowired
    private ClusteringService clusteringService;

    @PostMapping("/clustering")
    public ResponseEntity<?> performClustering(
            @RequestParam("file") MultipartFile file,
            @RequestParam("algorithm") String algorithm,
            @RequestParam Map<String, String> params) {

        try {
            ClusteringResult result = clusteringService.performClustering(file, algorithm, params);
            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(result);
        } catch (IOException e) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("error", "文件处理错误: " + e.getMessage());
            return ResponseEntity.badRequest().body(errorResponse);
        } catch (Exception e) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("error", "服务器错误: " + e.getMessage());
            return ResponseEntity.internalServerError().body(errorResponse);
        }
    }
}