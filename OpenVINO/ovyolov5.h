#pragma once
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "openvino/openvino.hpp"

class OVDetector {
 private:
  std::string model_path_;
  int class_num_;
  float conf_thres_;

  ov::InferRequest infer_request_;

 public:
  OVDetector(std::string _model_path, int _class_num, float _conf_thres);
  void Init();
  void Infer(const cv::Mat& _img, std::vector<cv::Rect>& _origin_rect,
             std::vector<float>& _origin_rect_cof,
             std::vector<int>& _origin_rect_cls, std::vector<int>& _indices_id);
  ~OVDetector();
};
