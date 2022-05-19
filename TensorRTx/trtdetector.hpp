#pragma once
#include <opencv2/opencv.hpp>

#include "yolov5.hpp"
#include "robo_data.h"

class TRTDetector : public YOLOv5TRT
{
private:

public:
  using YOLOv5TRT::YOLOv5TRT;
  void detect(cv::Mat &_img, std::vector<myrobo::detection> &_pred) {
    auto res = Detect(_img);
    for(auto &i : res) {
      myrobo::detection dt;
      dt.rect = get_rect(_img, i.bbox);
      dt.class_id = i.class_id;
      dt.conf = i.conf;
      _pred.emplace_back(dt);
    }
  }
};
