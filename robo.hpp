#pragma once
#include <thread>
#include <chrono>
#include <exception>

#include <opencv2/opencv.hpp>
#include <fmt/color.h>
#include <fmt/core.h>

#include "angle/prediction/kalman_prediction.hpp"
#include "angle/solvePnP/solvePnP.hpp"
#include "devices/camera/mv_video_capture.hpp"
#include "devices/new_serial/serial.hpp"
#include "robo_data.h"
#include "robo_utils.hpp"
#include "streamer/mjpeg_streamer.hpp"
#include "TensorRTx/yolov5.hpp"

class RoboR1 {
 private:
  bool end_node_{false};

  RoboCmd robo_cmd;
  RoboInf robo_inf;

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

using namespace std::chrono_literals;

RoboR1::RoboR1() {
  yolo_detection_ = std::make_unique<YOLOv5TRT>(
    fmt::format("{}{}", SOURCE_PATH, "/models/ball.engine"));
  pnp_ = std::make_unique<solvepnp::PnP>(
    fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_273.xml"),
    fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));
  kalman_prediction_ = std::make_unique<KalmanPrediction>();

  try {
    int camera_exposure = 5000;
    mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_1280_X_1024,
                                            camera_exposure);
    camera_ = std::make_shared<mindvision::VideoCapture>(camera_params);
    serial_ = std::make_unique<RoboSerial>("/dev/ttyACM0", 115200);
    streamer_ = std::make_unique<RoboStreamer>();
  } catch (const std::exception &e) {
    fmt::print("[{}] {}", fmt::format(fg(fmt::color::red) |
               fmt::emphasis::bold, "construct"), e.what());
  }
}

void RoboR1::Init() {
  streamer_->setStopNodeFuncPtr(std::bind(&RoboR1::stop, this));
  streamer_->setCameraSetExposureFuncPtr(std::bind(&mindvision::VideoCapture::setCameraExposureTime, camera_, std::placeholders::_1));

  try {
    if(!streamer_->isRunning())
      streamer_->start(8080, SOURCE_PATH "/streamer/streamer.html");
    if(!camera_->isOpen())
      camera_->open();
  } catch (const std::exception &e) {
    fmt::print("[{}] {}", fmt::format(fg(fmt::color::red) |
               fmt::emphasis::bold, "init"), e.what());
  }
}

void RoboR1::Spin() {
  std::thread uartWriteThread(std::bind(&RoboR1::uartWrite,this));
  std::thread uartReadThread(std::bind(&RoboR1::uartRead,this));
  std::thread detectionThread(std::bind(&RoboR1::detection,this));

  while (!end_node_) {
    if (uartWriteThread.joinable())
      uartWriteThread.detach();
    if (uartReadThread.joinable())
      uartReadThread.detach();
    if (detectionThread.joinable())
      detectionThread.detach();
    std::this_thread::sleep_for(1000ms);
  }

  try {
    if(serial_->isOpen())
      serial_->close();
    if(streamer_->isRunning())
      streamer_->stop();
  } catch (const std::exception& e) {
    fmt::print("{}\n", e.what());
  }
}

void RoboR1::uartRead() {
  while (!end_node_) try {
      if(serial_->isOpen()) {
        serial_->ReceiveInfo(robo_inf);
        streamer_->publish_charts_value("echarta", robo_inf.yaw_angle.load());
        streamer_->publish_text_value("imu_angle",
                                      robo_inf.yaw_angle.load());
      }
      std::this_thread::sleep_for(1ms);
    } catch (const std::exception &e) {
      std::this_thread::sleep_for(1000ms);
    }
}

void RoboR1::uartWrite() {
  while (!end_node_) try {
      if(serial_->isOpen()) {
        serial_->WriteInfo(robo_cmd);
      } else {
        serial_->open();
      }
      std::this_thread::sleep_for(10ms);
    } catch (const std::exception &e) {
      serial_->close();
      static int serial_read_excepted_times{0};
      if (serial_read_excepted_times++ > 3) {
        fmt::print("[{}] serial excepted to many times, sleep 10s.\n",
                   serial::idntifier_red);
        std::this_thread::sleep_for(10000ms);
        serial_read_excepted_times = 0;
      }
      fmt::print("[{}] {}\n",
                 serial::idntifier_red, e.what());
      std::this_thread::sleep_for(1000ms);
    }
}

void RoboR1::detection() {
  cv::Mat src_img;
  cv::Rect target_rect;
  cv::Rect target_rect_predicted;
  cv::Rect target_rect_3d(0, 0, 150, 150);
  cv::Point2f detection_pnp_angle;
  cv::Point2f target_pnp_angle;
  cv::Point3f target_pnp_coordinate_mm;
  float depth;
  while (!end_node_) try {
    if(camera_->isOpen()) {
      *camera_ >> src_img;

      auto res = yolo_detection_->Detect(src_img);

      if (rectFilter(res, src_img, target_rect)) {
        target_rect.height = target_rect.width;
        pnp_->solvePnP(target_rect_3d, target_rect, detection_pnp_angle,
                             target_pnp_coordinate_mm, depth);

        // kalman 预测
        float kalman_yaw_compensate =
        kalman_prediction_->Prediction(robo_inf.yaw_angle.load() -
                                             detection_pnp_angle.y, depth);
        target_rect_predicted = target_rect;
        target_rect_predicted.x = target_rect.x - kalman_yaw_compensate;
        pnp_->solvePnP(target_rect_3d, target_rect_predicted,
                             target_pnp_angle, target_pnp_coordinate_mm,
                             depth);

        // 弹道补偿
        float pitch_compensate = depth / 1000 * 1.9529 + 5.9291;
        target_pnp_angle.x -= pitch_compensate;

        robo_cmd.pitch_angle.store(target_pnp_angle.x);
        robo_cmd.yaw_angle.store(target_pnp_angle.y);
        robo_cmd.depth.store(depth);
        robo_cmd.detect_object.store(true);

        // 画出所有检测目标
        for (long unsigned int i = 0; i < res.size(); i++)
          cv::rectangle(src_img, get_rect(src_img, res[i].bbox),
                        cv::Scalar(0, 255, 0), 2);
        // 画出十字线条
        cv::line(src_img, cv::Point(0, src_img.rows / 2),
                cv::Point(src_img.cols, src_img.rows / 2),
                cv::Scalar(0, 150, 255), 2);
        cv::line(src_img, cv::Point(src_img.cols / 2, 0),
                    cv::Point(src_img.cols / 2, src_img.rows),
                    cv::Scalar(0, 150, 255), 2);

        // 画出瞄准目标、经过预测的目标, yaw, pitch
        cv::rectangle(src_img, target_rect, cv::Scalar(0, 150, 255), 5);
        cv::rectangle(src_img, target_rect_predicted, cv::Scalar(255, 0, 150),
                      5);
        cv::putText(src_img, std::to_string(depth),
                    cv::Point(target_rect.x, target_rect.y - 1),
                    cv::FONT_HERSHEY_DUPLEX,
                    1.2, cv::Scalar(0, 150, 255), 2);
        cv::putText(src_img,
                    "pitch:" + std::to_string(target_pnp_angle.x) +
                    ", yaw:" + std::to_string(target_pnp_angle.y),
                    cv::Point(0, 50), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 150, 255), 2);
        streamer_->publish_text_value("yaw_angle",target_pnp_angle.y);
        streamer_->publish_text_value("pitch_angle",target_pnp_angle.x);
        // 目标不动和跟踪目标过程为发射时机
        if((abs(detection_pnp_angle.y) < 0.5f && abs(target_pnp_angle.y) < 0.5f) ||
           (detection_pnp_angle.y < -0.5f && target_pnp_angle.y > 0.5f) ||
           (detection_pnp_angle.y > 0.5f && target_pnp_angle.y < -0.5f))
          streamer_->call_html_js_function("setReadytoShootGreen");
        else
          streamer_->call_html_js_function("setReadytoShootRed");
      } else {
        robo_cmd.detect_object.store(false);
        kalman_prediction_->Prediction(robo_inf.yaw_angle.load(), depth);
      }
      if (!src_img.empty()) {
        std::vector<uchar> buff_bgr;
        cv::imencode(".jpg", src_img, buff_bgr);
        streamer_->publish("/pc",
                                 std::string(buff_bgr.begin(),
                                 buff_bgr.end()));
      }
      usleep(1);
    }
  } catch (const std::exception &e) {
    fmt::print("{}\n", e.what());
  }
}

void RoboR1::stop(){
  if (!end_node_) {
    end_node_ = true;
  }
  fmt::print("node stop.");
}

RoboR1::~RoboR1() {}