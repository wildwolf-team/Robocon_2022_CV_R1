#pragma once

#include "TensorRTx/yolov5.hpp"

bool rectFilter(std::vector<Yolo::Detection> res, cv::Mat &img,
                cv::Rect &rect) {
  float max_conf = .0;
  int max_conf_res_id = -1;
  for (size_t i = 0; i < res.size(); i++) {
    if (0.8 > res[i].bbox[2] / res[i].bbox[3] &&
        res[i].bbox[2] / res[i].bbox[3] > 1.2)
      continue;
    if (res[i].conf > max_conf) {
      max_conf = res[i].conf;
      max_conf_res_id = i;
    }
  }
  if (max_conf_res_id != -1) {
    rect = get_rect(img, res[max_conf_res_id].bbox);
    return true;
  } else {
    return false;
  }
}