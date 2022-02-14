#pragma once
#include "serial/serial.h"
#include "utils.hpp"

class RoboSerial : public serial::Serial {
 public:
  RoboSerial(std::string port, unsigned long baud) {
    this->close();
    auto timeout = serial::Timeout::simpleTimeout(serial::Timeout::max());
    this->setPort(port);
    this->setBaudrate(baud);
    this->setTimeout(timeout);
    this->open();
    if (this->isOpen())
      fmt::print("Serial open successed.");
    else {
      fmt::print("Serial open failed.");
      for (int i = 0; i < 10; i++) {
        port.pop_back();
        this->setPort(port + std::to_string(i));
        this->open();
        if (this->isOpen()) break;
      }
      if (!this->isOpen()) {
        for (const auto &port_info : serial::list_ports()) {
          this->setPort(port_info.port);
          this->open();
          if (this->isOpen()) break;
        }
      }
    }
  }

  bool WriteInfo(RoboCmd &robo_cmd) {
    if (this->isOpen()) {
      this->write((uint8_t *)&robo_cmd, sizeof(robo_cmd));
      return true;
    } else
      return false;
  }

  void ReceiveInfo(RoboInf &robo_inf) {
    RoboInf temp_info;
    if (this->isOpen()) {
      uint8_t temp;
      this->read(&temp, 1);
      while (temp != 'S') {
        this->read(&temp, 1);
      }
      this->read((uint8_t *)&temp_info, sizeof(temp_info));
    }
    std::cout << "yaw_angle:" << temp_info.yaw_angle << "\n";
  }

 private:
};