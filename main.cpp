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
#include "angle/solvePnP/solvePnP.hpp"
#include "devices/camera/mv_video_capture.hpp"
#include "devices/serial/uart_serial.hpp"

using namespace std::chrono_literals;

struct roboCmd {
  std::atomic<float> pitch_angle = 0.f;
  std::atomic<float> yaw_angle = 0.f;
  std::atomic<float> depth = 0.f;
  std::atomic<bool> detect_object = false;
};

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

void PTZCameraThread(roboCmd &robo_cmd) {
  cv::Mat src_img;

  mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_1280_X_800,
                                        50000);
  auto mv_capture = std::make_shared<mindvision::VideoCapture>(camera_params);

  start(fmt::format("{}{}", SOURCE_PATH, "/models/RCBall3.engine"));

  auto pnp = std::make_shared<solvepnp::PnP>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_273.xml"),
      fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));

  // To-do: 异常终端程序后相机自动
  try {
    while (true) {
      mv_capture->isindustryimgInput();
      mv_capture->cameraReleasebuff();
      src_img = mv_capture->image();
      auto res = yolov5_v5_Rtx_start(src_img);

      cv::line(src_img, cv::Point(0, src_img.rows / 2),
               cv::Point(src_img.cols, src_img.rows / 2),
               cv::Scalar(0, 255, 190));
      cv::line(src_img, cv::Point(src_img.cols / 2, 0),
               cv::Point(src_img.cols / 2, src_img.rows),
               cv::Scalar(0, 255, 190));

      cv::Rect rect;
      if (rectFilter(res, src_img, rect)) {
        rect.height = rect.width;
        cv::Rect ball_3d_rect(0, 0, 165, 165);
        cv::Point2f angle;
        float depth;
        pnp->solvePnP(ball_3d_rect, rect, angle, depth);

        robo_cmd.pitch_angle.store(angle.x);
        robo_cmd.yaw_angle.store(angle.y);
        robo_cmd.depth.store(depth);
        robo_cmd.detect_object.store(true);

        cv::rectangle(src_img, rect, cv::Scalar(0, 255, 190), 2);
        cv::putText(src_img, std::to_string(depth),
                    cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_DUPLEX, 1.2,
                    cv::Scalar(0, 255, 190), 2);
        cv::putText(src_img,
                    "pitch:" + std::to_string(angle.x) +
                        ", yaw:" + std::to_string(angle.y),
                    cv::Point(0, 50), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 255, 190));
        fmt::print("pitch:{}, yaw:{} \n", angle.x, angle.y);
      } else {
        robo_cmd.detect_object.store(false);
      }
      if (!src_img.empty()) cv::imshow("img", src_img);
      if (cv::waitKey(1) == 'q') break;
      fmt::print("---------------\n");
    }
  } catch (...) {
    mv_capture->~VideoCapture();
    mindvision::VideoCapture mv_capture(camera_params);
  }
  destroy();
}

void uartReceiveThread(std::shared_ptr<uart::SerialPort> serial) {
  while (true) {
    try {
      serial->updateReceiveInformation();
      std::this_thread::sleep_for(1ms);
    } catch (...) {
    }
  }
}

void uartThread(roboCmd &robo_cmd) {
  auto serial = std::make_shared<uart::SerialPort>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/uart_serial_config.xml"));
  std::thread uart_receive_thread(uartReceiveThread, serial);
  uart_receive_thread.detach();

  while (true) {
    try {
      serial->updataWriteData(robo_cmd.yaw_angle.load(),
                             robo_cmd.pitch_angle.load(), robo_cmd.depth.load(),
                             robo_cmd.detect_object.load(), 0);
      std::this_thread::sleep_for(10ms);
    } catch (...) {
      // To-do: 串口掉线恢复
    }
  }
}

int main(int argc, char *argv[]) {
  roboCmd robo_cmd;
  std::thread camera_thread(PTZCameraThread, std::ref(robo_cmd));
  camera_thread.detach();
  std::thread uart_thread(uartThread, std::ref(robo_cmd));
  uart_thread.detach();
  if (std::cin.get() == 'q') {
    camera_thread.~thread();
    uart_thread.~thread();
  }
  return 0;
}