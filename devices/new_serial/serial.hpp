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
    struct temp_send_info_struct{
      uint8_t start = (unsigned)'S';
      float yaw_angle;
      float pitch_angle;
      float depth;
      uint8_t detect_object;
      uint8_t end = (unsigned)'E';
    } __attribute__((packed));
    temp_send_info_struct t1;
    t1.yaw_angle = robo_cmd.yaw_angle.load();
    t1.pitch_angle = robo_cmd.pitch_angle.load();
    t1.depth = robo_cmd.depth.load();
    t1.detect_object = robo_cmd.detect_object.load();

    this->write((uint8_t *)&t1, sizeof(t1));
  }

  void ReceiveInfo(RoboInf &robo_inf) {
    RoboInf temp_info;
    uint8_t temp;
    this->read(&temp, 1);
    while (temp != 'S')
      this->read(&temp, 1);
    this->read((uint8_t *)&temp_info, sizeof(temp_info));
    robo_inf.yaw_angle.store(temp_info.yaw_angle);
  }

 private:
};