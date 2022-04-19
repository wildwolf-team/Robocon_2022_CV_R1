#pragma once
#include <atomic>

struct RoboCmd {
  std::atomic<float> yaw_angle = 0.f;
  std::atomic<float> pitch_angle = 0.f;
  std::atomic<float> depth = 0.f;
  std::atomic<uint8_t> detect_object = false;
};

struct RoboInf {
  std::atomic<float> yaw_angle = 0;
  std::atomic<bool> following = false;
};

struct RoboCmdUartBuff{
  uint8_t start = (unsigned)'S';
  float yaw_angle;
  float pitch_angle;
  float depth;
  uint8_t detect_object;
  uint8_t end = (unsigned)'E';
} __attribute__((packed));

struct RoboInfUartBuff {
  float yaw_angle = 0;
  bool following = false;
} __attribute__((packed));