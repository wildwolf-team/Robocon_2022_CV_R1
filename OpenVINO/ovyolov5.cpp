#include "ovyolov5.h"

OVDetector::OVDetector(std::string _model_path, int _class_num,
                       float _conf_thres) {
  model_path_ = _model_path;
  class_num_ = _class_num;
  conf_thres_ = _conf_thres;
}

void OVDetector::Init() {
  ov::Core core;
  std::shared_ptr<ov::Model> model = core.read_model(model_path_);
  ov::preprocess::PrePostProcessor ppp(model);
  ppp.input("images").tensor().set_layout("NCHW").set_element_type(
      ov::element::f16);
  ppp.input("images").model().set_layout("NCHW");
  model = ppp.build();
  ov::CompiledModel compiled_model = core.compile_model(
      model, "GPU",
      ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  infer_request_ = compiled_model.create_infer_request();
}

void OVDetector::Infer(const cv::Mat& _img, std::vector<cv::Rect>& _origin_rect,
                       std::vector<float>& _origin_rect_cof,
                       std::vector<int>& _origin_rect_cls,
                       std::vector<int>& _indices_id) {
  cv::Mat resize_img;
  cv::Mat resize_blob(cv::Size(640, 640), CV_16FC3);

  if (_img.cols != _img.rows) {
    if (_img.cols > _img.rows) {
      cv::resize(_img, resize_img, cv::Size(640, 640 * _img.rows / _img.cols));
      cv::copyMakeBorder(resize_img, resize_img, 0, 640 - resize_img.rows, 0, 0,
                         cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else if (_img.cols < _img.rows) {
      cv::resize(_img, resize_img, cv::Size(640 * _img.cols / _img.rows, 640));
      cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, 640 - resize_img.cols,
                         cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
  } else {
    cv::resize(_img, resize_img, cv::Size(640, 640));
  }

  cv::Mat img_blob =
      cv::dnn::blobFromImage(resize_img, 1. / 255, cv::Size(640, 640),
                             cv::Scalar(), true, false, CV_32F);
  img_blob.convertTo(resize_blob, CV_16FC3);

  ov::Tensor input_tensor(ov::element::f16, ov::Shape({1, 3, 640, 640}),
                          resize_blob.data);
  infer_request_.set_input_tensor(input_tensor);

  infer_request_.infer();

  const ov::Tensor& output_tensor = infer_request_.get_output_tensor(0);
  const ov::Shape& shape = output_tensor.get_shape();
  auto out_data = output_tensor.data<float>();

  cv::Size resize_img_shape = resize_img.size();

  for (int i = 0; i < 25200; i++) {
    if (out_data[4 + i * (class_num_ + 5)] > conf_thres_) {
      // scale_coords
      float gain =
          MIN(float(resize_img_shape.width) / _img.cols,
              float(resize_img_shape.height) / _img.rows);  // gain  = old / new
      // float padw = (img1_shape.width - im.cols * gain) / 2; // w padding
      // float padh = (img1_shape.width - im.rows * gain) / 2; // h padding
      cv::Rect box((out_data[i * (class_num_ + 5)] -
                    out_data[i * (class_num_ + 5) + 2] / 2) /
                       gain,
                   (out_data[i * (class_num_ + 5) + 1] -
                    out_data[i * (class_num_ + 5) + 3] / 2) /
                       gain,
                   out_data[i * (class_num_ + 5) + 2] / gain,
                   out_data[i * (class_num_ + 5) + 3] / gain);
      std::vector<float> conf;
      for (int j = 0; j < class_num_; j++) {
        conf.emplace_back(
            out_data[5 + i * (class_num_ + 5) + j] *
            out_data[4 + i * (class_num_ + 5)]);  // conf = cls_conf * obj_conf
      }
      auto conf_max_it = max_element(conf.begin(), conf.end());
      int cls = conf_max_it - conf.begin();
      if (*conf_max_it > conf_thres_) {
        _origin_rect.emplace_back(box);
        _origin_rect_cof.emplace_back(*conf_max_it);
        _origin_rect_cls.emplace_back(cls);
      }
    }
  }

  cv::dnn::NMSBoxes(_origin_rect, _origin_rect_cof, 0.25, 0.45, _indices_id);
}

OVDetector::~OVDetector() {}