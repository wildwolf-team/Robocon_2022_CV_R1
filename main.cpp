#include <fmt/color.h>
#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "TensorRTx/yolov5.hpp"
#include "angle/prediction/kalman_prediction.hpp"
#include "angle/solvePnP/solvePnP.hpp"
#include "devices/camera/mv_video_capture.hpp"
#include "devices/serial/uart_serial.hpp"
#include "utils.hpp"
#include "devices/new_serial/serial.hpp"

using namespace std::chrono_literals;

void PTZCameraThread(RoboCmd &robo_cmd, RoboInf &robo_inf) {
  mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_1280_X_800,
                                        25000);
  auto mv_capture = std::make_shared<mindvision::VideoCapture>(camera_params);

  auto detect_ball = std::make_shared<YOLOv5TRT>(
    fmt::format("{}{}", SOURCE_PATH, "/models/RCBall3.engine"));

  auto pnp = std::make_shared<solvepnp::PnP>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_273.xml"),
      fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));

  auto kalman_prediction = std::make_shared<KalmanPrediction>();

  cv::Mat src_img;
  cv::Rect rect;
  cv::Rect rect_predicted;
  cv::Rect ball_3d_rect(0, 0, 165, 165);
  cv::Point2f angle;
  float depth;

  // To-do: 异常终端程序后相机自动
  while (cv::waitKey(1) != 'q') {
    if (mv_capture->isindustryimgInput()) {
      mv_capture->cameraReleasebuff();
      src_img = mv_capture->image();
      auto res = detect_ball->Detect(src_img);

#ifndef RELEASE
      cv::line(src_img, cv::Point(0, src_img.rows / 2),
               cv::Point(src_img.cols, src_img.rows / 2),
               cv::Scalar(0, 255, 190));
      cv::line(src_img, cv::Point(src_img.cols / 2, 0),
               cv::Point(src_img.cols / 2, src_img.rows),
               cv::Scalar(0, 255, 190));
#endif

      if (rectFilter(res, src_img, rect)) {
        rect.height = rect.width;
        pnp->solvePnP(ball_3d_rect, rect, angle, depth);

        float yaw_compensate = kalman_prediction->Prediction(robo_inf.yaw_angle.load(), depth);

        rect_predicted = rect;
        rect_predicted.x = rect.x + yaw_compensate;
        pnp->solvePnP(ball_3d_rect, rect_predicted, angle, depth);

        robo_cmd.pitch_angle.store(angle.x);
        robo_cmd.yaw_angle.store(angle.y);
        robo_cmd.depth.store(depth);
        robo_cmd.detect_object.store(true);

#ifndef RELEASE
        cv::rectangle(src_img, rect, cv::Scalar(0, 255, 190), 2);
        cv::rectangle(src_img, rect_predicted, cv::Scalar(0, 255, 0), 2);
        cv::putText(src_img, std::to_string(depth),
                    cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_DUPLEX, 1.2,
                    cv::Scalar(0, 255, 190), 2);
        cv::putText(src_img,
                    "pitch:" + std::to_string(angle.x) +
                        ", yaw:" + std::to_string(angle.y),
                    cv::Point(0, 50), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 255, 190));
#endif
      } else {
        robo_cmd.detect_object.store(false);
      }
#ifndef RELEASE
      if (!src_img.empty()) cv::imshow("img", src_img);
#endif
      if (cv::waitKey(1) == 'q') break;
    }
  }
}

// void uartReceiveThread(std::shared_ptr<uart::SerialPort> serial, RoboInf &robo_inf) {
//   while (true) {
//     try {
//       serial->updateReceiveInformation();
//       robo_inf.yaw_angle.store(serial->returnReceiveYaw());
//       std::this_thread::sleep_for(1ms);
//     } catch (...) {
//     }
//   }
// }
void uartReceiveThread(std::shared_ptr<RoboSerial> serial, RoboInf &robo_inf) {
  while (true) {
    try {

      std::this_thread::sleep_for(1ms);
    } catch (...) {
    }
  }
}

void uartThread(RoboCmd &robo_cmd, RoboInf &robo_inf) {
  auto new_serial = std::make_shared<RoboSerial>("/dev/ttyACM0", 921600);
  // auto serial = std::make_shared<uart::SerialPort>(
  //     fmt::format("{}{}", CONFIG_FILE_PATH, "/uart_serial_config.xml"));
  // std::thread uart_receive_thread(uartReceiveThread, new_serial, std::ref(robo_inf));
  // uart_receive_thread.detach();
  // serial::Serial my_serial("/dev/ttyACM0", 921600, serial::Timeout::simpleTimeout(1000));
  // uint8_t temp;
  // float temp1;

  while (true) {
    try {
      new_serial->ReceiveInfo(std::ref(robo_inf));
      // serial->updataWriteData(
      //     robo_cmd.yaw_angle.load(), robo_cmd.pitch_angle.load(),
      //     robo_cmd.depth.load(), robo_cmd.detect_object.load(), 0);
      std::this_thread::sleep_for(10ms);
    } catch (...) {
      // To-do: 串口掉线恢复
    }
  }
}

int main(int argc, char *argv[]) {
  RoboCmd robo_cmd;
  RoboInf robo_inf;
  std::thread camera_thread(PTZCameraThread, std::ref(robo_cmd), std::ref(robo_inf));
  camera_thread.detach();
  std::thread uart_thread(uartThread, std::ref(robo_cmd), std::ref(robo_inf));
  uart_thread.detach();
  if (std::cin.get() == 'q') {
    camera_thread.~thread();
    uart_thread.~thread();
  }
  return 0;
}