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
  uint8_t crc8x;
  uint8_t end = (unsigned)'E';
} __attribute__((packed));

struct RoboInfUartBuff {
  float yaw_angle = 0;
  bool following = false;
  uint8_t crc8x{0x00};
  uint8_t end{0x00};
} __attribute__((packed));

struct RoboCatchBallUartBuff {
  uint8_t start = (unsigned)'S';
  int code {0};
  uint8_t crc8x{0x00};
  uint8_t end = (unsigned)'E';
} __attribute__((packed));