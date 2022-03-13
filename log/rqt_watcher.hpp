#pragma once
#include "devices/new_serial/serial/serial.h"

class rqtWatcher : public serial::Serial {
 public:
  rqtWatcher(std::string serial_port, unsigned long baud) {
    auto timeout = serial::Timeout::simpleTimeout(serial::Timeout::max());
    this->setPort(serial_port);
    this->setBaudrate(baud);
    this->setTimeout(timeout);
    try {
      this->open();
      fmt::print("[{}] rqt watcher serial init successed.\n", idntifier_green);
    } catch (const std::exception &e) {
      fmt::print("[{}] rqt watcher serial init failed, {}.\n", idntifier_red,
                 e.what());
    }
  }

  void SendData(float v, char topic_flag) {
    if (this->isOpen())
    {
      this->write((uint8_t *)'S', 1);
      this->write((uint8_t *)&topic_flag, 1);
      this->write((uint8_t *)&v, sizeof(v));
    }
  }
};