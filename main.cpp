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
#include "log/log.hpp"
#include "log/rqt_watcher.hpp"
#include "utils.hpp"
#include "utils/mjpeg_streamer.hpp"

using namespace std::chrono_literals;

void PTZCameraThread(
    RoboCmd &robo_cmd, RoboInf &robo_inf,
    const std::shared_ptr<nadjieb::MJPEGStreamer> &streamer_ptr) {
  int camera_exposure = 5000;
  mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_1280_X_1024,
                                        camera_exposure);
  auto mv_capture = std::make_shared<mindvision::VideoCapture>(camera_params);

  auto detect_ball = std::make_shared<YOLOv5TRT>(
      fmt::format("{}{}", SOURCE_PATH, "/models/BALL.engine"));

  auto pnp = std::make_shared<solvepnp::PnP>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_273.xml"),
      fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));

  auto kalman_prediction = std::make_shared<KalmanPrediction>();

  auto log =
      std::make_shared<Log::log>(fmt::format("{}{}", SOURCE_PATH, "/log.txt"));

  cv::Mat src_img;
  cv::Rect rect;
  cv::Rect rect_predicted;
  cv::Rect ball_3d_rect(0, 0, 150, 150);
  cv::Point2f angle;
  cv::Point3f coordinate_mm;
  float depth;

  // To-do: 异常终端程序后相机自动
  while (true) try {
    if (mv_capture->isindustryimgInput()) {
      mv_capture->cameraReleasebuff();
      src_img = mv_capture->image();

      auto res = detect_ball->Detect(src_img);

      if (rectFilter(res, src_img, rect)) {
        rect.height = rect.width;
        pnp->solvePnP(ball_3d_rect, rect, angle, coordinate_mm, depth);

        // kalman 预测
        float yaw_compensate =
            kalman_prediction->Prediction(robo_inf.yaw_angle.load() - angle.y, depth);

        rect_predicted = rect;
        rect_predicted.x = rect.x - yaw_compensate;
        pnp->solvePnP(ball_3d_rect, rect_predicted, angle, coordinate_mm,
                      depth);

        // 弹道补偿
        float pitch_compensate = depth / 1000 * 1.9529 + 6.9291;
        angle.x -= pitch_compensate;

        robo_cmd.pitch_angle.store(angle.x);
        robo_cmd.yaw_angle.store(angle.y);
        robo_cmd.depth.store(depth);
        robo_cmd.detect_object.store(true);

#ifndef RELEASE
        //画出中心点
        for (long unsigned int i = 0; i < res.size(); i++)
          cv::rectangle(src_img, get_rect(src_img, res[i].bbox),
                        cv::Scalar(0, 255, 0), 2);
        cv::line(src_img, cv::Point(0, src_img.rows / 2),
                 cv::Point(src_img.cols, src_img.rows / 2),
                 cv::Scalar(0, 150, 255), 2);
        cv::line(src_img, cv::Point(src_img.cols / 2, 0),
                 cv::Point(src_img.cols / 2, src_img.rows),
                 cv::Scalar(0, 150, 255), 2);

        cv::rectangle(src_img, rect, cv::Scalar(0, 150, 255), 5);
        cv::rectangle(src_img, rect_predicted, cv::Scalar(255, 0, 150), 5);
        cv::putText(src_img, std::to_string(depth),
                    cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_DUPLEX,
                    1.2, cv::Scalar(0, 150, 255), 2);
        cv::putText(src_img,
                    "pitch:" + std::to_string(angle.x) +
                        ", yaw:" + std::to_string(angle.y),
                    cv::Point(0, 50), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 150, 255), 2);
#endif
      } else {
        robo_cmd.detect_object.store(false);
      }
#ifndef RELEASE
      if (!src_img.empty()) {
        std::vector<uchar> buff_bgr;
        cv::imencode(".jpg", src_img, buff_bgr);
        streamer_ptr->publish("/pc",
                              std::string(buff_bgr.begin(), buff_bgr.end()));
      }
#endif
      usleep(1);
    }
  } catch (const std::exception &e) {
    fmt::print("{}\n", e.what());
  }
}

void uartWriteThread(const std::shared_ptr<RoboSerial> &serial,
                     RoboCmd &robo_cmd) {
  while (true) try {
      if(serial->isOpen()) {
        serial->WriteInfo(robo_cmd);
      } else {
        serial->open();
      }
      std::this_thread::sleep_for(10ms);
    } catch (const std::exception &e) {
      serial->close();
      static int serial_read_excepted_times{0};
      if (serial_read_excepted_times++ > 3) {
        std::this_thread::sleep_for(10000ms);
        fmt::print("[{}] read serial excepted to many times, sleep 10s.\n",
                   idntifier_red);
        serial_read_excepted_times = 0;
      }
      fmt::print("[{}] serial exception: {}\n",
                 idntifier_red, e.what());
      std::this_thread::sleep_for(1000ms);
    }
}

void uartReadThread(const std::shared_ptr<RoboSerial> &serial,
                    RoboInf &robo_inf) {
  while (true) try {
      if(serial->isOpen()) {
        serial->ReceiveInfo(robo_inf);
      }
      std::this_thread::sleep_for(1ms);
    } catch (const std::exception &e) {
      static int serial_read_excepted_times{0};
      if (serial_read_excepted_times++ > 3) {
        std::this_thread::sleep_for(10000ms);
        fmt::print("[{}] read serial excepted to many times, sleep 10s.\n",
                   idntifier_red);
        serial_read_excepted_times = 0;
      }
      fmt::print("[{}] serial exception: {}\n",
                 idntifier_red, e.what());
      std::this_thread::sleep_for(1000ms);
    }
}

void watchDogThread(std::thread &ptz_camera_thread,
                    std::thread &uartWriteThread) {
  while (true)
  {
    if(ptz_camera_thread.joinable()) {
      ptz_camera_thread.detach();
    }
    if(uartWriteThread.joinable()) {
      uartWriteThread.detach();
    }
    sleep(1);
  }
}

int main(int argc, char *argv[]) {
  RoboCmd robo_cmd;
  RoboInf robo_inf;

  auto streamer_ptr = std::make_shared<nadjieb::MJPEGStreamer>();
  streamer_ptr->start(8080);

  auto serial = std::make_shared<RoboSerial>("/dev/ch340g", 115200);

  std::thread uart_write_thread(uartWriteThread, serial, std::ref(robo_cmd));
  uart_write_thread.detach();

  std::thread uart_read_thread(uartReadThread, serial, std::ref(robo_inf));
  uart_read_thread.detach();

  std::thread ptz_camera_thread(PTZCameraThread, std::ref(robo_cmd),
                                std::ref(robo_inf), std::ref(streamer_ptr));
  ptz_camera_thread.detach();

  std::thread watch_dog_thread(watchDogThread, std::ref(ptz_camera_thread),
                               std::ref(uart_read_thread));
  watch_dog_thread.join();

  return 0;
}