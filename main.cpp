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

using namespace std::chrono_literals;

void PTZCameraThread(roboCmd &robo_cmd, roboInf &robo_inf) {
  cv::Mat src_img;

  mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_1280_X_800,
                                        50000);
  auto mv_capture = std::make_shared<mindvision::VideoCapture>(camera_params);

  auto detect_ball = std::make_shared<YOLOv5TRT>(
    fmt::format("{}{}", SOURCE_PATH, "/models/RCBall3.engine"));

  auto pnp = std::make_shared<solvepnp::PnP>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_273.xml"),
      fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));

  cv::Rect rect;
  cv::Rect rect_predicted;
  cv::Rect ball_3d_rect(0, 0, 165, 165);
  cv::Point2f angle;
  float depth;

  // auto kalman_prediction = std::make_shared<KalmanPrediction>();
  using _Kalman = Kalman<1, S>;
  _Kalman             kalman;
  _Kalman::Matrix_xxd A = _Kalman::Matrix_xxd::Identity();
  _Kalman::Matrix_zxd H;
  H(0, 0) = 1;
  _Kalman::Matrix_xxd R;
  R(0, 0) = 1;
  for (int i = 1; i < S; i++) {
    R(i, i) = 100;
  }
  _Kalman::Matrix_zzd Q{4};
  _Kalman::Matrix_x1d init{0, 0};
  kalman = _Kalman(A, H, R, Q, init, 0);

  auto start = std::chrono::system_clock::now();

  float compensate_w           = 0;
  float last_compensate_w      = 0;
  float last_last_compensate_w = 0;
  int a = 0;
  double last_yaw = 0;
  double m_yaw = 0;
  _Kalman::Matrix_x1d state;

  // To-do: 异常终端程序后相机自动
  while (cv::waitKey(1) != 'q') {
    if (mv_capture->isindustryimgInput()) {
      mv_capture->cameraReleasebuff();
      src_img = mv_capture->image();
      auto res = detect_ball->Detect(src_img);

      cv::line(src_img, cv::Point(0, src_img.rows / 2),
               cv::Point(src_img.cols, src_img.rows / 2),
               cv::Scalar(0, 255, 190));
      cv::line(src_img, cv::Point(src_img.cols / 2, 0),
               cv::Point(src_img.cols / 2, src_img.rows),
               cv::Scalar(0, 255, 190));

      if (rectFilter(res, src_img, rect)) {
        rect.height = rect.width;

      auto          end      = std::chrono::system_clock::now();
      // static double t_data[] = {serial_.returnReceiveYaw(), 0, 0};
      // static cv::Mat R_IM     = cv::Mat(3, 1, CV_64FC1, t_data);
      // int   depth = 0;
      // float pitch = 0;
      m_yaw = robo_inf.yaw_angle.load();
      // m_yaw       = serial_.returnReceiveYaw();
      if (std::fabs(last_yaw - m_yaw) > (10 / 180. * M_PI)) {
        kalman.reset(m_yaw, std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        last_yaw = m_yaw;
        std::cout << "reset" << std::endl;
      } else {
        last_yaw = m_yaw;
        Eigen::Matrix<double, 1, 1> z_k{m_yaw};
        state = kalman.update(z_k, std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
      }

        // depth        = pnp_.returnDepth();
        // pitch        = pnp_.returnPitchAngle();
        // cv::Mat m_pw = pnp_.returnTvec_() + R_IM;
        static double last_yaw = 0, last_speed = 0.0;
      
        last_speed                        = state(1, 0);
        double  c_speed                   = state(1, 0) * depth;
        double  predict_time              = depth * 0.001 / (/*serial_.returnReceiveBulletVelocity()*/18);
        double  s_yaw                     = atan2(predict_time * c_speed, depth);
        compensate_w                      = 8 * tan(s_yaw * 180 / CV_PI);
        compensate_w                      *= 1000;
        compensate_w                      = (last_last_compensate_w + last_compensate_w + compensate_w) * 0.333;
        last_last_compensate_w            = last_compensate_w;
        last_compensate_w                 = compensate_w;
        static cv::Point2f ss             = cv::Point2f(0, 0);
        ss                                = cv::Point2f(compensate_w, 0);

        // double ptz_yaw_angle = 0;
        // cv::Point2f ss = kalman_prediction->Prediction(ptz_yaw_angle, angle, depth);
        std::cout << ss << "\n";

        rect_predicted.x = rect.x + compensate_w;

        pnp->solvePnP(ball_3d_rect, rect_predicted, angle, depth);

        robo_cmd.pitch_angle.store(angle.x);
        robo_cmd.yaw_angle.store(angle.y);
        robo_cmd.depth.store(depth);
        robo_cmd.detect_object.store(true);

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
        fmt::print("pitch:{}, yaw:{} \n", angle.x, angle.y);
      } else {
        robo_cmd.detect_object.store(false);
      }
      if (!src_img.empty()) cv::imshow("img", src_img);
      if (cv::waitKey(1) == 'q') break;
      fmt::print("---------------\n");
    }
  }
}

void uartReceiveThread(std::shared_ptr<uart::SerialPort> serial, roboInf &robo_inf) {
  while (true) {
    try {
      serial->updateReceiveInformation();
      robo_inf.yaw_angle.store(serial->returnReceiveYaw());
      std::this_thread::sleep_for(1ms);
    } catch (...) {
    }
  }
}

void uartThread(roboCmd &robo_cmd, roboInf &robo_inf) {
  auto serial = std::make_shared<uart::SerialPort>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/uart_serial_config.xml"));
  std::thread uart_receive_thread(uartReceiveThread, serial, std::ref(robo_inf));
  uart_receive_thread.detach();

  while (true) {
    try {
      serial->updataWriteData(
          robo_cmd.yaw_angle.load(), robo_cmd.pitch_angle.load(),
          robo_cmd.depth.load(), robo_cmd.detect_object.load(), 0);
      std::this_thread::sleep_for(10ms);
    } catch (...) {
      // To-do: 串口掉线恢复
    }
  }
}

int main(int argc, char *argv[]) {
  roboCmd robo_cmd;
  roboInf robo_inf;
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