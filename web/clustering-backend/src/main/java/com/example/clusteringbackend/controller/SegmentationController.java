package com.example.clusteringbackend.controller;

import com.example.clusteringbackend.service.SegmentationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

@RestController
public class SegmentationController {

    @Autowired
    private SegmentationService segmentationService;

    @PostMapping("/segmentation")
    public ResponseEntity<byte[]> performSegmentation(
            @RequestParam("file") MultipartFile file,
            @RequestParam("algorithm") String algorithm,
            @RequestParam Map<String, String> params) {

        try {
            byte[] result = segmentationService.performSegmentation(file, algorithm, params);
            return ResponseEntity.ok()
                    .contentType(MediaType.IMAGE_PNG)
                    .body(result);
        } catch (IOException e) {
            return ResponseEntity.badRequest().build();
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }
}