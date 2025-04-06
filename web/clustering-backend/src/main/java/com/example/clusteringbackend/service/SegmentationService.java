package com.example.clusteringbackend.service;

import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

public interface SegmentationService {
    byte[] performSegmentation(MultipartFile file, String algorithm, Map<String, String> params)
            throws IOException, Exception;
}