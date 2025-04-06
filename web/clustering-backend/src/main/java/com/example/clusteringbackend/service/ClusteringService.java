package com.example.clusteringbackend.service;

import com.example.clusteringbackend.dao.po.ClusteringResult;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

public interface ClusteringService {
    ClusteringResult performClustering(MultipartFile file, String algorithm, Map<String, String> params)
            throws IOException, Exception;
}