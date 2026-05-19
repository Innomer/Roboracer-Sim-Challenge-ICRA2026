#ifndef CUSTOM_CODES_CPP__RACER_NODE_HPP_
#define CUSTOM_CODES_CPP__RACER_NODE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/float32.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace custom_codes_cpp
{

struct ScoreBreakdown
{
  double progress_score{0.0};
  double clearance_score{0.0};
  double low_clearance_score{0.0};
  double wall_clearance_score{0.0};
  double smoothness_score{0.0};
  double steering_penalty{0.0};
  double gap_alignment{0.0};
  double loop_penalty{0.0};
  double end_wall_clearance{0.0};
  double turn_commitment{0.0};
  double anticipation_bonus{0.0};
  std::string reason{"score"};
};

struct PathEval
{
  bool collision{false};
  double min_clearance{999.0};
  double score_progress{0.0};
  double forward_progress{0.0};
  double loop_penalty{0.0};
  double end_wall_clearance{999.0};
};

struct CandidatePath
{
  std::vector<std::pair<double, double>> path;
  double score{-999999.0};
  double steer{0.0};
  bool collision{false};
  double clearance{0.0};
  double score_progress{0.0};
  double forward_progress{0.0};
  double loop_penalty{0.0};
  double end_wall_clearance{0.0};
  double turn_bias_{0.0};
  ScoreBreakdown breakdown;
};

struct GapDebug
{
  double angle{0.0};
  double smoothed_angle{0.0};
  double threshold{0.0};
  std::vector<bool> free_mask;
  std::size_t best_start{0};
  std::size_t best_len{0};
};

struct DecisionDebug
{
  double best_score{-999999.0};
  double best_steering{0.0};
  double best_progress{0.0};
  double best_forward_progress{0.0};
  double best_clearance{0.0};
  double throttle{0.0};
  double forward_min{0.0};
  double rollout_distance{0.0};
  double gap_angle{0.0};
  double turn_prediction{0.0};
  std::string throttle_state{"CLEAR"};
  ScoreBreakdown breakdown;
};

class RacerNode : public rclcpp::Node
{
public:
  RacerNode();

private:
  static constexpr double CAR_WIDTH = 0.2700;
  static constexpr double WHEELBASE = 0.3240;
  static constexpr double TRACK_WIDTH = 0.2360;
  static constexpr double WHEEL_RADIUS = 0.0590;
  static constexpr int ENCODER_PPR = 16;
  static constexpr int ENCODER_CR = 120;
  static constexpr int TICKS_PER_REV = ENCODER_PPR * ENCODER_CR;
  static constexpr double WHEEL_CIRC = 2.0 * M_PI * WHEEL_RADIUS;
  static constexpr double DIST_PER_TICK = WHEEL_CIRC / TICKS_PER_REV;
  static constexpr double MAX_STEER_RAD = 0.5236;
  static constexpr int LIDAR_RAYS = 1080;

  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void leftEncoderCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void rightEncoderCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void lidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);

  void validateLidarSlice();
  void updateOdometry();

  std::vector<double> removeDoubleWallArtifacts(const std::vector<double> & ranges);
  std::vector<double> filterDuctGapArtifacts(const std::vector<double> & ranges);
  std::vector<double> applyDisparityExtender(
    const std::vector<double> & ranges,
    double angle_inc,
    double radius);
  GapDebug getGapDebug(const std::vector<double> & ranges, const std::vector<double> & angles);
  std::vector<std::pair<double, double>> generateArcPath(double steering_angle, double distance);
  PathEval evaluatePath(
    const std::vector<std::pair<double, double>> & path_points,
    const std::vector<double> & lidar_ranges,
    const std::vector<double> & lidar_angles,
    double radius);
  std::pair<double, std::pair<double, double>> computePursuitSteering(
    const std::vector<std::pair<double, double>> & path_points,
    double lookahead);
  double predictTurnDirection(const std::vector<double> & ranges);
  double chooseEscapeSteering(
    const std::vector<double> & ranges,
    const std::vector<double> & angles,
    double gap_angle,
    double turn_prediction);
  double chooseHardCornerSteering(
    const std::vector<double> & ranges,
    const std::vector<double> & angles,
    double gap_angle,
    double turn_prediction) const;
  double computeThrottle(
    double forward_progress,
    double clearance,
    double steering,
    double forward_min,
    double max_t,
    double min_t,
    double roll_dist);
  std::string forwardState(double forward_min);
  bool hasReverseLoopRisk(
    const std::vector<std::pair<double, double>> & path,
    double rollout_distance) const;
  double historyPenaltyForPath(const std::vector<std::pair<double, double>> & path) const;
  double reverseHistoryPenaltyForEndpoint(double local_x, double local_y) const;

  void publishProcessedLidar(
    const sensor_msgs::msg::LaserScan & scan,
    const std::vector<double> & processed_ranges);
  void publishDebugMarkers(
    const sensor_msgs::msg::LaserScan & scan,
    const std::vector<CandidatePath> & paths,
    const std::vector<std::pair<double, double>> & best_path,
    const std::pair<double, double> * pursuit_target,
    const std::vector<double> & raw_ranges,
    const std::vector<double> & processed_ranges,
    const std::vector<double> & angles,
    const GapDebug & gap_debug,
    const DecisionDebug & decision_debug);

  geometry_msgs::msg::Point makePoint(double x, double y, double z) const;
  std_msgs::msg::ColorRGBA makeColor(double r, double g, double b, double a) const;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_enc_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr right_enc_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr steer_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr throttle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr processed_lidar_pub_;

  double current_x_{0.0};
  double current_y_{0.0};
  double current_yaw_{0.0};
  bool left_ticks_prev_valid_{false};
  bool right_ticks_prev_valid_{false};
  double left_ticks_prev_{0.0};
  double right_ticks_prev_{0.0};
  double left_ticks_{0.0};
  double right_ticks_{0.0};
  double imu_yaw_rate_{0.0};
  rclcpp::Time last_odom_time_;
  std::deque<std::pair<double, double>> path_history_;
  std::size_t max_history_{120};
  double prev_steering_{0.0};
  double prev_gap_angle_{0.0};
  double max_range_cap_{9.0};
  std::vector<double> prev_processed_ranges_;
  int lidar_start_idx_{240};
  int lidar_end_idx_{840};
  double current_throttle_{0.0};
  int consecutive_blocked_frames_{0};
  int consecutive_low_progress_frames_{0};
  bool stuck_corner_mode_{false};
};

}  // namespace custom_codes_cpp

#endif  // CUSTOM_CODES_CPP__RACER_NODE_HPP_
