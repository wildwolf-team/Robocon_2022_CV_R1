#pragma once
#include <thread>
#include <chrono>
#include <exception>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/float32.hpp"
#include <opencv2/opencv.hpp>
#define FMT_HEADER_ONLY
#include <fmt/color.h>
#include <fmt/core.h>

#include "angle/prediction/kalman_prediction.hpp"
#include "angle/solvePnP/solvePnP.hpp"
#include "devices/camera/mv_video_capture.hpp"
#include "devices/new_serial/serial.hpp"
#include "robo_data.h"
#include "streamer/mjpeg_streamer.hpp"
#include "TensorRTx/yolov5.hpp"

class RoboR1 {
 private:
  bool end_node_{false};
  bool is_kalman_open_{true};

  RoboCmd robo_cmd;
  RoboInf robo_inf;

  rclcpp::Node::SharedPtr n_;
  std::unique_ptr<RoboSerial> serial_;
  std::shared_ptr<mindvision::VideoCapture> camera_;
  std::unique_ptr<YOLOv5TRT> yolo_detection_;
  std::unique_ptr<solvepnp::PnP> pnp_;
  std::unique_ptr<KalmanPrediction> kalman_prediction_;
  std::unique_ptr<RoboStreamer> streamer_;

 public:
  RoboR1();
  void Init();
  void Spin();
  void uartWrite();
  void uartRead();
  void detection();
  void stop();
  ~RoboR1();
};
