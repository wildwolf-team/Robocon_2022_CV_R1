#pragma once
#include "robo_data.h"
#include "ovyolov5.h"

class OVDetector : public Detector
{
private:

public:
  using Detector::Detector;
  void detect(Mat &_img, vector<myrobo::detection> &_ds) {
    vector<Object> objs;
    process_frame(_img, objs);
    for(auto &i : objs) {
      myrobo::detection d;
      d.rect = i.rect;
      d.class_id = i.id;
      d.conf = i.prob;
      _ds.emplace_back(d);
    }
  };
};
