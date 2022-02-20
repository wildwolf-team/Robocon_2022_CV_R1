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
#include "devices/new_serial/serial.hpp"
#include "devices/serial/uart_serial.hpp"
#include "utils.hpp"

using namespace std::chrono_literals;

void PTZCameraThread(RoboCmd &robo_cmd, RoboInf &robo_inf) {
  mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_1280_X_800,
                                        10000);
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

        float yaw_compensate =
            kalman_prediction->Prediction(robo_inf.yaw_angle.load(), depth);

        rect_predicted = rect;
        rect_predicted.x = rect.x + yaw_compensate;
        pnp->solvePnP(ball_3d_rect, rect_predicted, angle, depth);

        robo_cmd.pitch_angle.store(angle.x);
        robo_cmd.yaw_angle.store(angle.y);
        robo_cmd.depth.store(depth);
        robo_cmd.detect_object.store(true);

#ifndef RELEASE
        cv::rectangle(src_img, rect, cv::Scalar(0, 255, 190), 2);
        cv::rectangle(src_img, rect_predicted, cv::Scalar(0, 255, 190), 2);
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

void uartWriteThread(const std::shared_ptr<RoboSerial>& serial, RoboCmd &robo_cmd) {
  while (true) {
    try {
      serial->WriteInfo(robo_cmd);
      std::this_thread::sleep_for(10ms);
    } catch (const std::exception &e) {
      fmt::print("[{}] serial exception: {} serial restarting...\n", idntifier_red, e.what());
      std::this_thread::sleep_for(1000ms);
    }
  }
}

void uartReadThread(const std::shared_ptr<RoboSerial>& serial, RoboInf &robo_inf) {
  while (true) {
    try {
      serial->ReceiveInfo(robo_inf);
      std::this_thread::sleep_for(1ms);
    } catch (const std::exception &e) {
      fmt::print("[{}] serial exception: {} serial restarting...\n", idntifier_red, e.what());
      std::this_thread::sleep_for(1000ms);
    }
  }
}

void uartThread(RoboCmd &robo_cmd, RoboInf &robo_inf) {
  auto serial = std::make_shared<RoboSerial>("/dev/ttyACM0", 115200);
  std::thread uart_write_thread(uartWriteThread, serial,
                                  std::ref(robo_cmd));
  uart_write_thread.detach();
  std::thread uart_read_thread(uartReadThread, serial,
                                  std::ref(robo_inf));
  uart_read_thread.detach();

  while (true)
  {
    try {
      serial->available();
    } catch (const std::exception &e) {
      static int change_serial_port_times {0};
      while (++change_serial_port_times)
      {
        if (change_serial_port_times > 5) {
          fmt::print("[{}] Serial restarted to many times, sleep 1min...\n", idntifier_red);
          std::this_thread::sleep_for(6000ms);
          change_serial_port_times = 0;
        }
        fmt::print("[{}] exception: {} serial restarting...\n", idntifier_red, e.what());
        std::string port = serial->getPort();
        port.pop_back();
        try { serial->close(); } catch(...) { }
        
        for (int i = 0; i < 5; i++) {
          fmt::print("[{}] try to change to {}{} port.\n", idntifier_red, port, i);
          try { serial->setPort(port + std::to_string(i)); } catch(...) { }
          try {
            serial->open();
          } catch (const std::exception &e1) {
            fmt::print("[{}] change {}{} serial failed.\n", idntifier_red, port, i);
          }
          if (serial->isOpen()) {
            fmt::print("[{}] change to {}{} serial successed.\n", idntifier_green, port, i);
            break;
          }
          std::this_thread::sleep_for(300ms);
        }
        if (serial->isOpen()) break;
      }
    }
    std::this_thread::sleep_for(1000ms);
  }
}

int main(int argc, char *argv[]) {
  RoboCmd robo_cmd;
  RoboInf robo_inf;
  std::thread camera_thread(PTZCameraThread, std::ref(robo_cmd),
                            std::ref(robo_inf));
  camera_thread.detach();
  std::thread uart_thread(uartThread, std::ref(robo_cmd), std::ref(robo_inf));
  uart_thread.detach();
  if (std::cin.get() == 'q') {
    camera_thread.~thread();
    uart_thread.~thread();
  }
  return 0;
}