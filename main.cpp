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
using _Kalman = Kalman<1, S>;

void PTZCameraThread(RoboCmd &robo_cmd, RoboInf &robo_inf) {
  _Kalman             kalman;
  _Kalman::Matrix_xxd A = _Kalman::Matrix_xxd::Identity();
  _Kalman::Matrix_zxd H;
  H(0, 0) = 1;
  _Kalman::Matrix_xxd R;
  R(0, 0) = 20;
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
  int   a                      = 0;
  // int num =0;
  double c_speed = 0.0, last_speed = 0.0;
  float top_yaw = 0;
  float last_top_yaw = 0;
  float  pitch                  = 0;
  float  yaw                    = 0;
  int    num                    = 0;
  int    resettimes             = 0;
  int    angletimes1            = 0;
  int    angletimes2            = 0;
  int    angleyaw               = 0;
  bool   issame                 = 0;
  bool   isTop                  = 0;



  mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_1280_X_800,
                                        7500);
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

  while (cv::waitKey(1) != 'q') {
    if (mv_capture->isindustryimgInput()) {
      auto          end = std::chrono::system_clock::now();
      static double last_yaw = 0, last_speed = 0.0;
      angleyaw = robo_inf.yaw_angle.load() - last_yaw;
      issame = angleyaw >= 0 ? 1 : 0;
      if(issame){
        angletimes1++;
        angletimes2 = 0;
      }else{
        angletimes2++;
        angletimes1 = 0;
      }
      if (std::fabs(last_yaw - robo_inf.yaw_angle.load()) > (5 / 180. * M_PI)) {
        kalman.reset(robo_inf.yaw_angle.load(), std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 0.001);
        last_yaw = robo_inf.yaw_angle.load();
        std::cout << "reset" << std::endl;
        cv::putText(src_img, "reset", cv::Point(0, 200), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 150, 255));
        resettimes++;
      } else {
        std::cout <<"top================"<< isTop <<std::endl; 
        last_yaw = robo_inf.yaw_angle.load();
        Eigen::Matrix<double, 1, 1> z_k{robo_inf.yaw_angle.load()};
        _Kalman::Matrix_x1d         state = kalman.update(z_k, std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 0.001);
        // c_speed                           = state(1, 0) * 1.75;
        c_speed                           = state(1, 0) * 2.2;
        c_speed                           = (c_speed + last_speed) * 0.5;
        last_speed                        = c_speed;
        top_yaw                           = state(0, 0);
      }
      
      mv_capture->cameraReleasebuff();
      src_img = mv_capture->image();
      auto res = detect_ball->Detect(src_img);

#ifndef RELEASE
      cv::line(src_img, cv::Point(0, src_img.rows / 2),
               cv::Point(src_img.cols, src_img.rows / 2),
               cv::Scalar(0, 150, 255));
      cv::line(src_img, cv::Point(src_img.cols / 2, 0),
               cv::Point(src_img.cols / 2, src_img.rows),
               cv::Scalar(0, 150, 255));
#endif

      if (PTZCameraRectFilter(res, src_img, rect)) {
        rect.height = rect.width;
        pnp->solvePnP(ball_3d_rect, rect, angle, depth);

        std::cout << "c_speed = " << c_speed << std::endl;
        double predict_time = (depth * 0.001 / 10);
        double s_yaw        = atan2(predict_time * c_speed * depth * 0.001, 1);
        compensate_w        = 8 * tan(s_yaw);
        // compensate_w       *= 1000;
        compensate_w           = (last_last_compensate_w + last_compensate_w + compensate_w) * 0.333;
        last_last_compensate_w = last_compensate_w;
        last_compensate_w      = compensate_w;
        cv::putText(src_img, "compensate_w:" + std::to_string(compensate_w),
                    cv::Point(0, 150), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 150, 255), 1);

        // float yaw_compensate =
        //     kalman_prediction->Prediction(robo_inf.yaw_angle.load(), depth);

        rect_predicted = rect;
        rect_predicted.x = rect.x - compensate_w;
        // pnp->solvePnP(ball_3d_rect, rect_predicted, angle, depth);

        robo_cmd.pitch_angle.store(angle.x);
        robo_cmd.yaw_angle.store(angle.y);
        robo_cmd.depth.store(depth);
        robo_cmd.detect_object.store(true);

#ifndef RELEASE
        cv::rectangle(src_img, rect, cv::Scalar(0, 150, 255), 2);
        cv::rectangle(src_img, rect_predicted, cv::Scalar(255, 0, 150), 2);
        cv::putText(src_img, std::to_string(depth),
                    cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_DUPLEX, 1.2,
                    cv::Scalar(0, 150, 255), 2);
        cv::putText(src_img,
                    "pitch:" + std::to_string(angle.x) +
                        ", yaw:" + std::to_string(angle.y),
                    cv::Point(0, 50), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 150, 255));
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

void ClipBallThread(RoboCmd &robo_cmd) {
  auto clip_camera = std::make_shared<cv::VideoCapture>(0);
  auto detect_ball = std::make_shared<YOLOv5TRT>(
      fmt::format("{}{}", SOURCE_PATH, "/models/RCBall3.engine"));
  auto pnp = std::make_shared<solvepnp::PnP>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_273.xml"),
      fmt::format("{}{}", CONFIG_FILE_PATH, "/clip_pnp_config.xml"));
  cv::Mat src_img;
  cv::Rect ball_rect;
  cv::Rect ball_3d_rect(0, 0, 165, 165);
  cv::Point2f angle;
  float depth;

  while (cv::waitKey(1) != 'q') {
    if (clip_camera->isOpened())
    {
      clip_camera->read(src_img);
      auto res = detect_ball->Detect(src_img);
      if (ClipCameraRectFilter(res, src_img, ball_rect))
      {
        pnp->solvePnP(ball_3d_rect, ball_rect, angle, depth);
        robo_cmd.clip_x_angle.store(angle.x);

#ifndef RELEASE
        if (!src_img.empty()) cv::imshow("clip_img", src_img);
#endif
      }
    }
  }
}

void uartWriteThread(const std::shared_ptr<RoboSerial>& serial, RoboCmd &robo_cmd) {
  while (true) {
    try {
      serial->WriteInfo(robo_cmd);
      std::this_thread::sleep_for(10ms);
    } catch (const std::exception &e) {
      static int serial_write_excepted_times {0};
      if (serial_write_excepted_times++ > 5) {
        std::this_thread::sleep_for(10000ms);
        fmt::print("[{}] write serial excepted to many times, sleep 10s.\n", idntifier_red);
        serial_write_excepted_times = 0;
      }
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
      static int serial_read_excepted_times {0};
      if (serial_read_excepted_times++ > 5) {
        std::this_thread::sleep_for(10000ms);
        fmt::print("[{}] read serial excepted to many times, sleep 10s.\n", idntifier_red);
        serial_read_excepted_times = 0;
      }
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
      if (change_serial_port_times++ > 5) {
        fmt::print("[{}] Serial restarted to many times, sleep 1min...\n", idntifier_red);
        std::this_thread::sleep_for(10000ms);
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
  // std::thread clip_camera_thread(ClipBallThread, std::ref(robo_cmd));
  // clip_camera_thread.detach();
  if (std::cin.get() == 'q') {
    camera_thread.~thread();
    uart_thread.~thread();
    // clip_camera_thread.~thread();
  }
  return 0;
}