package com.example.clusteringbackend.dao.po;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.json.JSONObject;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class ClusteringResult {
  String image;
  Integer sampleCount;
  Integer dimensions;
  List<Point> samples = new ArrayList<Point>();
}
