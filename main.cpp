#include "robo.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_unique<RoboR1>();
  node->Init();
  node->Spin();
  return 0;
}