#pragma once
#include <chrono>

#include "kalman.hpp"
#include "utils.hpp"

//上海交通大学的 Kalman 预测

constexpr int S = 2;
using _Kalman = Kalman<1, S>;

class KalmanPrediction : public Kalman<1, S> {
 public:
  KalmanPrediction() {
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
    _Kalman(A, H, R, Q, init, 0);
    start = std::chrono::system_clock::now();
  }

  cv::Point2f Prediction(double ptz_yaw_angle, cv::Point2f angle, float depth) {
    auto end = std::chrono::system_clock::now();
    m_yaw = ptz_yaw_angle;
    if (std::fabs(last_yaw - m_yaw) > (10 / 180. * M_PI)) {
      this->reset(m_yaw, std::chrono::duration_cast<std::chrono::milliseconds>(
                              end - start)
                              .count());
      last_yaw = m_yaw;
      std::cout << "reset" << std::endl;
    } else {
      last_yaw = m_yaw;
      Eigen::Matrix<double, 1, 1> z_k{m_yaw};
      state = this->update(
          z_k,
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count());
    }
    static double last_yaw = 0, last_speed = 0.0;

    last_speed = state(1, 0);
    double c_speed = state(1, 0) * depth;
    double predict_time =
        depth * 0.001 / (/*serial_.returnReceiveBulletVelocity()*/ 18);
    double s_yaw = atan2(predict_time * c_speed, static_cast<int>(depth));
    compensate_w = 8 * tan(s_yaw * 180 / CV_PI);
    compensate_w *= 1000;
    compensate_w =
        (last_last_compensate_w + last_compensate_w + compensate_w) * 0.333;
    last_last_compensate_w = last_compensate_w;
    last_compensate_w = compensate_w;
    static cv::Point2f ss = cv::Point2f(0, 0);
    ss = cv::Point2f(compensate_w, 0);
    return ss;
  }

 private:
//   _Kalman kalman;
  std::chrono::time_point<std::chrono::system_clock> start;
  std::chrono::time_point<std::chrono::system_clock> end;

  float compensate_w = 0;
  float last_compensate_w = 0;
  float last_last_compensate_w = 0;
  int a = 0;
  double last_yaw = 0;
  double m_yaw = 0;
  _Kalman::Matrix_x1d state;
};