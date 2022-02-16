#pragma once
#include "serial/serial.h"
#include "utils.hpp"

auto idntifier_green = fmt::format(fg(fmt::color::green) | fmt::emphasis::bold, "serial");
auto idntifier_red   = fmt::format(fg(fmt::color::red)   | fmt::emphasis::bold, "serial");

class RoboSerial : public serial::Serial {
 public:
  RoboSerial(std::string port, unsigned long baud) {
    auto timeout = serial::Timeout::simpleTimeout(serial::Timeout::max());
    this->setPort(port);
    this->setBaudrate(baud);
    this->setTimeout(timeout);
    try {
      this->open();
      fmt::print("[{}] Serial init successed.\n", idntifier_green);
    } catch(const std::exception& e) {
      fmt::print("[{}] Serial init failed, {}.\n", idntifier_red, e.what());
    }
  }

  void WriteInfo(RoboCmd &robo_cmd) {
    // float yaw_angle = robo_cmd.yaw_angle.load();
    // float pitch_angle = robo_cmd.pitch_angle.load();
    // float depth = robo_cmd.depth.load();
    // uint8_t detect_object = robo_cmd.detect_object.load();
    float yaw_angle = 1.1;
    float pitch_angle = 1.1;
    float depth = 1.1;
    uint8_t detect_object = 1;
    this->write((uint8_t *)&robo_cmd.start_flag, 1);
    this->write((uint8_t *)&yaw_angle, 4);
    this->write((uint8_t *)&pitch_angle, 4);
    this->write((uint8_t *)&depth, 4);
    this->write((uint8_t *)&detect_object, 1);
    this->write((uint8_t *)&robo_cmd.end_flag, 1);
  }

  void ReceiveInfo(RoboInf &robo_inf) {
    RoboInf temp_info;
      uint8_t temp;
      this->read(&temp, 1);
      while (temp != 'S')
        this->read(&temp, 1);
      this->read((uint8_t *)&temp_info, sizeof(temp_info));
  }

 private:
};