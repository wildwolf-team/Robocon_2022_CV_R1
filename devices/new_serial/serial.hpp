#pragma once
#include "serial/serial.h"
#include "robo_data.h"

namespace serial {
auto idntifier_green = fmt::format(fg(fmt::color::green) | fmt::emphasis::bold, "serial");
auto idntifier_red   = fmt::format(fg(fmt::color::red)   | fmt::emphasis::bold, "serial");
}

class RoboSerial : public serial::Serial {
 public:
  RoboSerial(std::string port, unsigned long baud) {
    auto timeout = serial::Timeout::simpleTimeout(serial::Timeout::max());
    this->setPort(port);
    this->setBaudrate(baud);
    this->setTimeout(timeout);
    try {
      this->open();
      fmt::print("[{}] Serial init successed.\n", serial::idntifier_green);
    } catch(const std::exception& e) {
      fmt::print("[{}] Serial init failed, {}.\n", serial::idntifier_red, e.what());
    }
  }

  void WriteInfo(RoboCmd &robo_cmd) {
    RoboCmdUartBuff robo_cmd_uart_temp;
    robo_cmd_uart_temp.yaw_angle = robo_cmd.yaw_angle.load();
    robo_cmd_uart_temp.pitch_angle = robo_cmd.pitch_angle.load();
    robo_cmd_uart_temp.depth = robo_cmd.depth.load();
    robo_cmd_uart_temp.detect_object = robo_cmd.detect_object.load();

    this->write((uint8_t *)&robo_cmd_uart_temp, sizeof(robo_cmd_uart_temp));
  }

  void ReceiveInfo(RoboInf &robo_inf) {
    RoboInfUartBuff robo_inf_uart_temp;
    uint8_t temp;
    this->read(&temp, 1);
    while (temp != 'S')
      this->read(&temp, 1);
    this->read((uint8_t *)&robo_inf_uart_temp, sizeof(robo_inf_uart_temp));
    robo_inf.yaw_angle.store(robo_inf_uart_temp.yaw_angle);
  }

 private:
};