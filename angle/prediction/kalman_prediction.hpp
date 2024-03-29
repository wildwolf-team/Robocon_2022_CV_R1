#pragma once
#include <chrono>

#include "kalman.hpp"
#include "robo.hpp"

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
    R(0, 0) = 0.1;
    for (int i = 1; i < S; i++) {
      R(i, i) = 0.01;
    }
    _Kalman::Matrix_zzd Q{1};
    _Kalman::Matrix_x1d init{0, 0};
    this->reset(A, H, R, Q, init, 0);
    start = std::chrono::system_clock::now();
  }

  float Prediction(double ptz_yaw_angle, float depth) {
    auto end = std::chrono::system_clock::now();
    m_yaw = ptz_yaw_angle;
    if (std::fabs(last_yaw - m_yaw) > 7.5 ) {
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

    // double c_yaw = state(0, 0); // yaw 滤波数值，单位弧度
    double c_speed = state(1, 0) * depth * 0.001; // 角速度 转 线速度 单位 m/s
    // c_speed = (c_speed + last_speed) * 0.5;
    // last_speed = c_speed;
    double predict_time = depth * 0.001 * 98.331 + 23.871;
    predict_time = predict_time / 1000;
    double p_yaw        = atan2(predict_time * c_speed, depth * 0.001); // 预测出的近似弧度

    compensate_w = p_yaw * 180 / CV_PI;

    return compensate_w;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start;

  float compensate_w = 0;
  float last_compensate_w = 0;
  float last_last_compensate_w = 0;
  int a = 0;
  double last_yaw = 0;
  double last_speed =0;
  double m_yaw = 0;
  _Kalman::Matrix_x1d state;
};