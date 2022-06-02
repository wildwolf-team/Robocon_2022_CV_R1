#pragma once
#include "ovyolov5.h"
#include "robo_data.h"

class Detector : public OVDetector {
 private:
 public:
  using OVDetector::OVDetector;
  void detect(const cv::Mat &_img, std::vector<myrobo::detection> &_ds) {
    std::vector<cv::Rect> origin_rect;
    std::vector<float> origin_rect_cof;
    std::vector<int> origin_rect_cls;
    std::vector<int> indices_id;

    this->Infer(_img, origin_rect, origin_rect_cof, origin_rect_cls,
                indices_id);

    if(_ds.size() != 0) _ds.clear();

    for(size_t i = 0; i < indices_id.size(); i++) {
      myrobo::detection dt;
      dt.rect = origin_rect[indices_id[i]];
      dt.conf = origin_rect_cof[indices_id[i]];
      dt.class_id = origin_rect_cls[indices_id[i]];
      _ds.emplace_back(dt);
    }
  };
};
