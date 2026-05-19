#include "custom_codes_cpp/racer_node.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<custom_codes_cpp::RacerNode>());
  rclcpp::shutdown();
  return 0;
}
