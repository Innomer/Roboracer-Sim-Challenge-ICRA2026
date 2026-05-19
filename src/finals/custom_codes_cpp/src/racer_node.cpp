#include "custom_codes_cpp/racer_node.hpp"

#include <numeric>
#include <sstream>

namespace custom_codes_cpp
{

namespace
{

double clip(double value, double lower, double upper)
{
  return std::max(lower, std::min(value, upper));
}

double meanMasked(const std::vector<double> & values, const std::vector<bool> & mask)
{
  double sum = 0.0;
  std::size_t count = 0;
  for (std::size_t i = 0; i < values.size() && i < mask.size(); ++i) {
    if (mask[i]) {
      sum += values[i];
      ++count;
    }
  }
  return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

double maxMasked(const std::vector<double> & values, const std::vector<bool> & mask)
{
  double result = 0.0;
  for (std::size_t i = 0; i < values.size() && i < mask.size(); ++i) {
    if (mask[i]) {
      result = std::max(result, values[i]);
    }
  }
  return result;
}

double medianFive(
  double a,
  double b,
  double c,
  double d,
  double e)
{
  std::array<double, 5> values{a, b, c, d, e};
  std::sort(values.begin(), values.end());
  return values[2];
}

}  // namespace

RacerNode::RacerNode()
: Node("robust_rollout_racer_cpp"), last_odom_time_(this->get_clock()->now())
{
  declare_parameter("max_throttle", 0.25);
  declare_parameter("min_throttle", 0.04);
  declare_parameter("throttle_ramp_rate", 0.02);
  declare_parameter("safety_margin", 0.075);
  declare_parameter("preferred_path_clearance", 0.5);
  declare_parameter("preferred_endpoint_clearance", 0.5);
  declare_parameter("low_clearance_penalty_weight", 50.0);
  declare_parameter("wall_clearance_weight", 40.0);
  declare_parameter("pursuit_lookahead", 1.7);
  declare_parameter("pursuit_blend", 0.0);
  declare_parameter("progress_weight", 3.0);
  declare_parameter("clearance_weight", 1.5);
  declare_parameter("smoothness_weight", -1.2);
  declare_parameter("steering_penalty_weight", -0.4);
  declare_parameter("gap_alignment_weight", 1.5);
  declare_parameter("turn_commitment_weight", 5.0);
  declare_parameter("rollout_distance", 2.5);
  declare_parameter("max_steering_delta", 0.07);
  declare_parameter("temporal_blend", 0.15);
  declare_parameter("use_disparity_extender", false);
  declare_parameter("disparity_threshold", 0.75);
  declare_parameter("max_jump_distance", 1.5);
  declare_parameter("history_penalty_radius", 1.2);
  declare_parameter("history_penalty_weight", 4.0);
  declare_parameter("gap_threshold_scale", 0.45);
  declare_parameter("max_gap_threshold", 2.5);
  declare_parameter("turn_prediction_gain", 0.25);
  declare_parameter("forward_critical_distance", 0.5);
  declare_parameter("forward_warning_distance", 1.0);
  declare_parameter("forward_caution_distance", 1.20);
  declare_parameter("enable_visualizations", true);
  declare_parameter("debug_top_candidate_count", 5);
  declare_parameter("lidar_half_fov_degrees", 180.0);
  declare_parameter("candidate_steer_count", 32);
  declare_parameter("stuck_blocked_frame_threshold", 3);
  declare_parameter("stuck_low_progress_frame_threshold", 12);
  declare_parameter("stuck_forward_progress_threshold", 0.1);
  declare_parameter("stuck_clearance_threshold", 0.1);
  declare_parameter("stuck_exit_forward_progress", 0.80);
  declare_parameter("stuck_exit_clearance", 0.15);
  declare_parameter("stuck_mode_steering_delta", 0.52);
  declare_parameter("escape_alignment_weight", 1.0);
  declare_parameter("reverse_history_penalty_weight", 12.0);
  declare_parameter("enable_reverse_loop_guard", false);
  declare_parameter("min_forward_progress_fraction", 0.15);
  declare_parameter("max_forward_backtrack", 0.40);
  declare_parameter("history_path_penalty_weight", 0.0);
  declare_parameter("enable_corner_turn_assist", false);
  declare_parameter("corner_turn_front_distance", 1.35);
  declare_parameter("corner_turn_gap_angle", 0.18);
  declare_parameter("corner_turn_min_steer", 0.36);
  declare_parameter("enable_hard_corner_override", false);
  declare_parameter("hard_corner_front_distance", 0.75);
  declare_parameter("hard_corner_min_steer", 0.50);
  declare_parameter("hard_corner_turn_direction", 0);

  lidar_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
    "/autodrive/roboracer_1/lidar",
    rclcpp::SensorDataQoS(),
    std::bind(&RacerNode::lidarCallback, this, std::placeholders::_1));
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    "/autodrive/roboracer_1/imu",
    10,
    std::bind(&RacerNode::imuCallback, this, std::placeholders::_1));
  left_enc_sub_ = create_subscription<sensor_msgs::msg::JointState>(
    "/autodrive/roboracer_1/left_encoder",
    10,
    std::bind(&RacerNode::leftEncoderCallback, this, std::placeholders::_1));
  right_enc_sub_ = create_subscription<sensor_msgs::msg::JointState>(
    "/autodrive/roboracer_1/right_encoder",
    10,
    std::bind(&RacerNode::rightEncoderCallback, this, std::placeholders::_1));

  steer_pub_ = create_publisher<std_msgs::msg::Float32>(
    "/autodrive/roboracer_1/steering_command", 10);
  throttle_pub_ = create_publisher<std_msgs::msg::Float32>(
    "/autodrive/roboracer_1/throttle_command", 10);
  marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    "/autodrive/debug_markers", 10);
  processed_lidar_pub_ = create_publisher<sensor_msgs::msg::LaserScan>(
    "/autodrive/debug_processed_lidar", 10);

  current_throttle_ = get_parameter("min_throttle").as_double();
  validateLidarSlice();
  RCLCPP_INFO(get_logger(), "C++ Rollout Racer initialized");
}

void RacerNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  imu_yaw_rate_ = msg->angular_velocity.z;
}

void RacerNode::leftEncoderCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  if (!msg->position.empty()) {
    left_ticks_ = msg->position[0];
  }
}

void RacerNode::rightEncoderCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  if (!msg->position.empty()) {
    right_ticks_ = msg->position[0];
  }
}

void RacerNode::validateLidarSlice()
{
  const int centre_ray = LIDAR_RAYS / 2;
  double half_fov_degrees = get_parameter("lidar_half_fov_degrees").as_double();
  half_fov_degrees = clip(half_fov_degrees, 1.0, (LIDAR_RAYS * 0.25) / 2.0);
  const int half_arc_idx = static_cast<int>(half_fov_degrees / 0.25);
  lidar_start_idx_ = std::max(0, centre_ray - half_arc_idx);
  lidar_end_idx_ = std::min(LIDAR_RAYS, centre_ray + half_arc_idx);
}

void RacerNode::updateOdometry()
{
  const auto now = get_clock()->now();
  const double dt = (now - last_odom_time_).seconds();
  last_odom_time_ = now;

  if (dt <= 0.0 || dt > 0.5) {
    return;
  }
  if (!left_ticks_prev_valid_ || !right_ticks_prev_valid_) {
    left_ticks_prev_ = left_ticks_;
    right_ticks_prev_ = right_ticks_;
    left_ticks_prev_valid_ = true;
    right_ticks_prev_valid_ = true;
    return;
  }

  const double d_left = (left_ticks_ - left_ticks_prev_) * DIST_PER_TICK;
  const double d_right = (right_ticks_ - right_ticks_prev_) * DIST_PER_TICK;
  left_ticks_prev_ = left_ticks_;
  right_ticks_prev_ = right_ticks_;

  const double d_center = (d_left + d_right) / 2.0;
  const double yaw_enc = (d_right - d_left) / TRACK_WIDTH;
  const double yaw_imu = imu_yaw_rate_ * dt;
  const double d_yaw = 0.60 * yaw_enc + 0.40 * yaw_imu;
  const double heading = current_yaw_ + d_yaw / 2.0;

  current_x_ += d_center * std::cos(heading);
  current_y_ += d_center * std::sin(heading);
  current_yaw_ += d_yaw;

  path_history_.push_back({current_x_, current_y_});
  if (path_history_.size() > max_history_) {
    path_history_.pop_front();
  }
}

void RacerNode::lidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  updateOdometry();
  validateLidarSlice();

  const double max_throttle = get_parameter("max_throttle").as_double();
  const double min_throttle = get_parameter("min_throttle").as_double();
  const double rollout_distance =
    get_parameter("rollout_distance").as_double() + 2.0 * (current_throttle_ / max_throttle);
  const double safety_margin = get_parameter("safety_margin").as_double();
  const double pursuit_lookahead = get_parameter("pursuit_lookahead").as_double();
  const double pursuit_blend = get_parameter("pursuit_blend").as_double();
  const double w_progress = get_parameter("progress_weight").as_double();
  const double w_clearance = get_parameter("clearance_weight").as_double();
  const double w_wall_clearance = get_parameter("wall_clearance_weight").as_double();
  const double preferred_path_clearance = get_parameter("preferred_path_clearance").as_double();
  const double preferred_endpoint_clearance =
    get_parameter("preferred_endpoint_clearance").as_double();
  const double w_low_clearance = get_parameter("low_clearance_penalty_weight").as_double();
  const double w_smoothness = get_parameter("smoothness_weight").as_double();
  const double w_steering = get_parameter("steering_penalty_weight").as_double();
  const double w_gap = get_parameter("gap_alignment_weight").as_double();
  const double w_commitment = get_parameter("turn_commitment_weight").as_double();
  const double vehicle_radius = (CAR_WIDTH / 2.0) + safety_margin;
  const bool enable_visualizations = get_parameter("enable_visualizations").as_bool();

  const std::size_t raw_size = msg->ranges.size();
  const int start_idx = std::max(0, std::min(lidar_start_idx_, static_cast<int>(raw_size)));
  const int end_idx = std::max(start_idx, std::min(lidar_end_idx_, static_cast<int>(raw_size)));
  const std::size_t slice_len = static_cast<std::size_t>(end_idx - start_idx);

  std::vector<double> raw_ranges(raw_size, 0.0);
  for (std::size_t i = 0; i < raw_size; ++i) {
    double r = static_cast<double>(msg->ranges[i]);
    if (std::isnan(r)) {
      r = 0.0;
    } else if (!std::isfinite(r) && r > 0.0) {
      r = max_range_cap_;
    } else if (!std::isfinite(r)) {
      r = 0.0;
    }
    raw_ranges[i] = clip(r, 0.0, max_range_cap_);
  }

  std::vector<double> ranges;
  std::vector<double> raw_slice;
  std::vector<double> angles;
  ranges.reserve(slice_len);
  raw_slice.reserve(slice_len);
  angles.reserve(slice_len);
  const double angle_min = msg->angle_min + start_idx * msg->angle_increment;
  for (std::size_t i = 0; i < slice_len; ++i) {
    const double r = raw_ranges[start_idx + static_cast<int>(i)];
    ranges.push_back(r);
    raw_slice.push_back(r);
    angles.push_back(angle_min + static_cast<double>(i) * msg->angle_increment);
  }

  ranges = removeDoubleWallArtifacts(ranges);
  ranges = filterDuctGapArtifacts(ranges);

  std::vector<double> processed_ranges;
  if (get_parameter("use_disparity_extender").as_bool()) {
    processed_ranges = applyDisparityExtender(ranges, msg->angle_increment, vehicle_radius);
  } else {
    processed_ranges = ranges;
  }

  if (prev_processed_ranges_.size() == processed_ranges.size()) {
    const double temporal_blend = get_parameter("temporal_blend").as_double();
    for (std::size_t i = 0; i < processed_ranges.size(); ++i) {
      processed_ranges[i] =
        temporal_blend * prev_processed_ranges_[i] + (1.0 - temporal_blend) * processed_ranges[i];
    }
  }
  prev_processed_ranges_ = processed_ranges;

  const std::size_t mid = processed_ranges.size() / 2;
  const std::size_t f_start = mid > 60 ? mid - 60 : 0;
  const std::size_t f_end = std::min(processed_ranges.size(), mid + 60);
  double forward_min = max_range_cap_;
  for (std::size_t i = f_start; i < f_end; ++i) {
    forward_min = std::min(forward_min, processed_ranges[i]);
  }

  double best_score = -999999.0;
  double best_steering = 0.0;
  double best_progress = 0.0;
  double best_forward_progress = 0.0;
  double best_clearance = 0.0;
  bool best_collision = true;
  std::vector<std::pair<double, double>> best_path_points;
  std::vector<CandidatePath> candidate_paths;
  const int candidate_count = std::max(
    2,
    static_cast<int>(get_parameter("candidate_steer_count").as_int()));
  candidate_paths.reserve(static_cast<std::size_t>(candidate_count));

  GapDebug gap_debug = getGapDebug(processed_ranges, angles);
  // double best_gap_angle = 0.85 * prev_gap_angle_ + 0.15 * gap_debug.angle;
  double gap_delta = std::abs(gap_debug.angle - prev_gap_angle_);
double gap_alpha = gap_delta > 0.30 ? 0.45 : 0.15;  // react faster to sudden gap shifts
double best_gap_angle = (1.0 - gap_alpha) * prev_gap_angle_ + gap_alpha * gap_debug.angle;
  gap_debug.smoothed_angle = best_gap_angle;
  prev_gap_angle_ = best_gap_angle;
  const double turn_prediction = predictTurnDirection(processed_ranges);

  ScoreBreakdown selected_breakdown;
  int viable_path_count = 0;
  for (int i = 0; i < candidate_count; ++i) {
    const double ratio = static_cast<double>(i) / static_cast<double>(candidate_count - 1);
    const double steering_angle = -MAX_STEER_RAD + ratio * (2.0 * MAX_STEER_RAD);
    CandidatePath candidate;
    candidate.path = generateArcPath(steering_angle, rollout_distance);
    PathEval eval = evaluatePath(candidate.path, processed_ranges, angles, vehicle_radius);
    const bool reverse_loop_risk = hasReverseLoopRisk(candidate.path, rollout_distance);
    if (reverse_loop_risk) {
      eval.collision = true;
    }
    candidate.steer = steering_angle;
    candidate.collision = eval.collision;
    candidate.clearance = eval.min_clearance;
    candidate.score_progress = eval.score_progress;
    candidate.forward_progress = eval.forward_progress;
    candidate.loop_penalty = eval.loop_penalty;
    candidate.end_wall_clearance = eval.end_wall_clearance;

    if (eval.collision) {
      candidate.score = -9999.0;
      candidate.breakdown.reason = reverse_loop_risk ? "reverse loop guard" : "collision clearance";
      candidate.breakdown.loop_penalty = eval.loop_penalty;
      candidate.breakdown.end_wall_clearance = eval.end_wall_clearance;
    } else {
      ++viable_path_count;
      const double progress_score = w_progress * eval.score_progress;
      const double clearance_score = w_clearance * eval.min_clearance;
      const double low_clearance_score =
        -w_low_clearance * std::max(preferred_path_clearance - eval.min_clearance, 0.0);
      const double wall_clearance_score =
        -w_wall_clearance * std::max(
        preferred_endpoint_clearance - eval.end_wall_clearance, 0.0);
      const double smoothness_score =
        w_smoothness * std::pow(std::abs(steering_angle - prev_steering_), 1.4);
      const double steering_penalty = w_steering * std::pow(std::abs(steering_angle), 1.2);
      const double anticipation_bonus = 1.6 * std::cos(steering_angle - turn_prediction);
      const double predicted_angle = best_gap_angle + turn_prediction;
      const double gap_alignment = w_gap * std::cos(steering_angle - predicted_angle);
      double turn_commitment = 0.0;
      if (std::abs(best_gap_angle) > 0.08) {
        turn_commitment = w_commitment * (1.0 - std::abs(steering_angle - best_gap_angle));
      }
      candidate.turn_bias_ =
    0.95 * candidate.turn_bias_ +
    0.05 * best_steering;
    const double turn_consistency =
    5.0 *
    std::cos(
        steering_angle - candidate.turn_bias_);
      candidate.score =
        progress_score + clearance_score + low_clearance_score + wall_clearance_score +
        smoothness_score + steering_penalty + gap_alignment - eval.loop_penalty +
        turn_commitment + anticipation_bonus + turn_consistency;
      candidate.breakdown.progress_score = progress_score;
      candidate.breakdown.clearance_score = clearance_score;
      candidate.breakdown.low_clearance_score = low_clearance_score;
      candidate.breakdown.wall_clearance_score = wall_clearance_score;
      candidate.breakdown.smoothness_score = smoothness_score;
      candidate.breakdown.steering_penalty = steering_penalty;
      candidate.breakdown.gap_alignment = gap_alignment;
      candidate.breakdown.loop_penalty = eval.loop_penalty;
      candidate.breakdown.end_wall_clearance = eval.end_wall_clearance;
      candidate.breakdown.turn_commitment = turn_commitment;
      candidate.breakdown.anticipation_bonus = anticipation_bonus;
      candidate.breakdown.reason = "highest weighted score";
    }

    const bool prefer_candidate =
      (!candidate.collision && best_collision) ||
      (candidate.collision == best_collision && candidate.score > best_score);

    if (prefer_candidate) {
      best_score = candidate.score;
      best_steering = steering_angle;
      best_progress = eval.score_progress;
      best_forward_progress = eval.forward_progress;
      best_clearance = eval.min_clearance;
      best_path_points = candidate.path;
      selected_breakdown = candidate.breakdown;
      best_collision = candidate.collision;
    }

    candidate_paths.push_back(std::move(candidate));
  }

  bool all_paths_colliding = viable_path_count == 0;
  bool recovery_escape = false;
  if (all_paths_colliding) {
    ++consecutive_blocked_frames_;
  } else {
    consecutive_blocked_frames_ = 0;
  }
  if (best_forward_progress < get_parameter("stuck_forward_progress_threshold").as_double() && best_clearance < get_parameter("stuck_clearance_threshold").as_double()) {
    ++consecutive_low_progress_frames_;
  } else {
    consecutive_low_progress_frames_ = 0;
  }

  if (
    consecutive_blocked_frames_ >=
    static_cast<int>(get_parameter("stuck_blocked_frame_threshold").as_int()) ||
    consecutive_low_progress_frames_ >=
    static_cast<int>(get_parameter("stuck_low_progress_frame_threshold").as_int()))
  {
    stuck_corner_mode_ = true;
  }
  if (
    stuck_corner_mode_ &&
    !all_paths_colliding &&
    best_forward_progress > get_parameter("stuck_exit_forward_progress").as_double() &&
    best_clearance > get_parameter("stuck_exit_clearance").as_double())
  {
    stuck_corner_mode_ = false;
    consecutive_low_progress_frames_ = 0;
  }
  if (all_paths_colliding) {
    best_steering = chooseEscapeSteering(
      processed_ranges,
      angles,
      best_gap_angle,
      turn_prediction);
    best_score = -9999.0;
    best_progress = 0.0;
    best_forward_progress = 0.0;
    best_clearance = 0.0;
    best_path_points = generateArcPath(best_steering, std::min(rollout_distance, 1.2));
    selected_breakdown = ScoreBreakdown{};
    selected_breakdown.reason = "all rollouts blocked: escape to clearer side";
    recovery_escape = true;
  }

  if (
    stuck_corner_mode_ &&
    !all_paths_colliding &&
    best_forward_progress < get_parameter("stuck_exit_forward_progress").as_double())
  {
    best_steering = chooseEscapeSteering(
      processed_ranges,
      angles,
      best_gap_angle,
      turn_prediction);
    best_path_points = generateArcPath(best_steering, std::min(rollout_distance, 1.2));
    selected_breakdown.reason = "stuck corner: forward escape";
    recovery_escape = true;
  }

  if (get_parameter("enable_hard_corner_override").as_bool() &&
    (forward_min < get_parameter("hard_corner_front_distance").as_double() || stuck_corner_mode_))
  {
    best_steering = chooseHardCornerSteering(
      processed_ranges,
      angles,
      best_gap_angle,
      turn_prediction);
    best_path_points = generateArcPath(best_steering, std::min(rollout_distance, 1.4));
    selected_breakdown.reason = "hard corner override";
    recovery_escape = true;
  } else if (!recovery_escape && get_parameter("enable_corner_turn_assist").as_bool()) {
    const double desired_turn = best_gap_angle + turn_prediction;
    const double front_distance = get_parameter("corner_turn_front_distance").as_double();
    const double min_gap_angle = get_parameter("corner_turn_gap_angle").as_double();
    const double min_corner_steer = get_parameter("corner_turn_min_steer").as_double();
    if (forward_min < front_distance && std::abs(desired_turn) > min_gap_angle) {
      const double forced_steer =
        std::copysign(std::max(std::abs(best_steering), min_corner_steer), desired_turn);
      best_steering = clip(forced_steer, -MAX_STEER_RAD, MAX_STEER_RAD);
      best_path_points = generateArcPath(best_steering, std::min(rollout_distance, 1.6));
      selected_breakdown.reason = "corner turn assist";
    }
  }

  bool has_pursuit_target = false;
  std::pair<double, double> pursuit_target{0.0, 0.0};
  if (!best_path_points.empty() && !recovery_escape) {
    const auto pursuit = computePursuitSteering(best_path_points, pursuit_lookahead);
    best_steering = (1.0 - pursuit_blend) * best_steering + pursuit_blend * pursuit.first;
    pursuit_target = pursuit.second;
    has_pursuit_target = true;
  }

  double max_steering_delta = get_parameter("max_steering_delta").as_double();
  if (stuck_corner_mode_) {
    max_steering_delta = std::max(
      max_steering_delta,
      get_parameter("stuck_mode_steering_delta").as_double());
  }
  if (
    get_parameter("enable_corner_turn_assist").as_bool() &&
    forward_min < get_parameter("corner_turn_front_distance").as_double())
  {
    max_steering_delta = std::max(max_steering_delta, 0.24);
  }
  const double delta = clip(best_steering - prev_steering_, -max_steering_delta, max_steering_delta);
  best_steering = prev_steering_ + delta;
  if (recovery_escape) {
    best_steering = 0.35 * prev_steering_ + 0.65 * best_steering;
    best_path_points = generateArcPath(best_steering, std::min(rollout_distance, 1.2));
  }

  prev_steering_ = best_steering;
  prev_gap_angle_ = 0.92 * prev_gap_angle_ + 0.08 * best_steering;
  double throttle = computeThrottle(
    best_forward_progress, best_clearance, best_steering, forward_min, max_throttle, min_throttle,
    rollout_distance);
  if (stuck_corner_mode_) {
    throttle = min_throttle;
    current_throttle_ = min_throttle;
  }

  std_msgs::msg::Float32 steer_msg;
  steer_msg.data = static_cast<float>(clip(best_steering / MAX_STEER_RAD, -1.0, 1.0));
  steer_pub_->publish(steer_msg);

  std_msgs::msg::Float32 throttle_msg;
  throttle_msg.data = static_cast<float>(throttle);
  throttle_pub_->publish(throttle_msg);

  if (!enable_visualizations) {
    return;
  }

  DecisionDebug decision_debug;
  decision_debug.best_score = best_score;
  decision_debug.best_steering = best_steering;
  decision_debug.best_progress = best_progress;
  decision_debug.best_forward_progress = best_forward_progress;
  decision_debug.best_clearance = best_clearance;
  decision_debug.throttle = throttle;
  decision_debug.forward_min = forward_min;
  decision_debug.rollout_distance = rollout_distance;
  decision_debug.gap_angle = best_gap_angle;
  decision_debug.turn_prediction = turn_prediction;
  decision_debug.throttle_state = forwardState(forward_min);
  decision_debug.breakdown = selected_breakdown;

  publishProcessedLidar(*msg, processed_ranges);
  publishDebugMarkers(
    *msg,
    candidate_paths,
    best_path_points,
    has_pursuit_target ? &pursuit_target : nullptr,
    raw_slice,
    processed_ranges,
    angles,
    gap_debug,
    decision_debug);
}

std::vector<double> RacerNode::removeDoubleWallArtifacts(const std::vector<double> & ranges)
{
  std::vector<double> filtered = ranges;
  const double jump_distance = get_parameter("max_jump_distance").as_double();
  for (std::size_t i = 2; i + 2 < ranges.size(); ++i) {
    if (
      std::abs(ranges[i] - ranges[i - 1]) > jump_distance &&
      std::abs(ranges[i] - ranges[i + 1]) > jump_distance)
    {
      filtered[i] = (ranges[i - 1] + ranges[i + 1]) / 2.0;
    }
  }

  std::vector<double> smoothed = filtered;
  for (std::size_t i = 2; i + 2 < filtered.size(); ++i) {
    smoothed[i] = medianFive(
      filtered[i - 2],
      filtered[i - 1],
      filtered[i],
      filtered[i + 1],
      filtered[i + 2]);
  }
  return smoothed;
}

std::vector<double> RacerNode::filterDuctGapArtifacts(const std::vector<double> & ranges)
{
  std::vector<double> result = ranges;
  const double gap_thresh = 0.80 * max_range_cap_;
  const double wall_near = 2.0;
  for (std::size_t i = 2; i + 2 < ranges.size(); ++i) {
    if (ranges[i] > gap_thresh) {
      const std::array<double, 4> neighbors{ranges[i - 2], ranges[i - 1], ranges[i + 1], ranges[i + 2]};
      int near_count = 0;
      double near_sum = 0.0;
      for (const double r : neighbors) {
        if (r < wall_near) {
          ++near_count;
          near_sum += r;
        }
      }
      if (near_count >= 3) {
        result[i] = near_sum / near_count;
      }
    }
  }
  return result;
}

std::vector<double> RacerNode::applyDisparityExtender(
  const std::vector<double> & ranges,
  double angle_inc,
  double radius)
{
  std::vector<double> new_ranges = ranges;
  const double disparity_threshold = get_parameter("disparity_threshold").as_double();
  for (std::size_t i = 1; i < ranges.size(); ++i) {
    const double diff = ranges[i] - ranges[i - 1];
    if (std::abs(diff) > disparity_threshold) {
      const double dist = std::min(ranges[i], ranges[i - 1]);
      const double bubble_angle = std::atan2(radius, std::max(dist, 0.1));
      const int num_indices = static_cast<int>(bubble_angle / angle_inc);
      std::size_t start = 0;
      std::size_t end = 0;
      if (diff > 0.0) {
        start = i;
        end = std::min(i + static_cast<std::size_t>(std::max(num_indices, 0)), ranges.size());
      } else {
        const std::size_t n = static_cast<std::size_t>(std::max(num_indices, 0));
        start = i > n ? i - n : 0;
        end = i;
      }
      for (std::size_t j = start; j < end; ++j) {
        const double reduction = 0.92 - 0.25 *
          (std::abs(static_cast<int>(j) - static_cast<int>(i)) /
          static_cast<double>(std::max(1, num_indices)));
        new_ranges[j] = std::min(new_ranges[j], dist * reduction);
      }
    }
  }
  return new_ranges;
}

GapDebug RacerNode::getGapDebug(
  const std::vector<double> & ranges,
  const std::vector<double> & angles)
{
  GapDebug debug;
  if (ranges.empty()) {
    return debug;
  }
  const double max_range = *std::max_element(ranges.begin(), ranges.end());
  debug.threshold = std::min(
    max_range * get_parameter("gap_threshold_scale").as_double(),
    get_parameter("max_gap_threshold").as_double());
  debug.free_mask.reserve(ranges.size());

  std::size_t cur_start = 0;
  std::size_t cur_len = 0;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const bool free = ranges[i] > debug.threshold;
    debug.free_mask.push_back(free);
    if (free) {
      if (cur_len == 0) {
        cur_start = i;
      }
      ++cur_len;
      if (cur_len > debug.best_len) {
        debug.best_len = cur_len;
        debug.best_start = cur_start;
      }
    } else {
      cur_len = 0;
    }
  }
  if (debug.best_len > 0 && !angles.empty()) {
    const std::size_t idx = std::min(debug.best_start + debug.best_len / 2, angles.size() - 1);
    debug.angle = angles[idx];
    debug.smoothed_angle = debug.angle;
  }
  return debug;
}

std::vector<std::pair<double, double>> RacerNode::generateArcPath(
  double steering_angle,
  double distance)
{
  std::vector<std::pair<double, double>> path;
  path.reserve(static_cast<std::size_t>(std::ceil(distance / 0.08)) + 1);
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double d = 0.0;
  const double ds = 0.08;
  while (d < distance) {
    const double steer = d < distance * 0.5 ? steering_angle : steering_angle * 0.35;
    x += ds * std::cos(yaw);
    y += ds * std::sin(yaw);
    yaw += ds * std::tan(steer) / WHEELBASE;
    path.push_back({x, y});
    d += ds;
  }
  return path;
}

PathEval RacerNode::evaluatePath(
  const std::vector<std::pair<double, double>> & path_points,
  const std::vector<double> & lidar_ranges,
  const std::vector<double> & lidar_angles,
  double radius)
{
  PathEval eval;
  std::vector<double> obstacle_x;
  std::vector<double> obstacle_y;
  obstacle_x.reserve(lidar_ranges.size());
  obstacle_y.reserve(lidar_ranges.size());
  for (std::size_t i = 0; i < lidar_ranges.size() && i < lidar_angles.size(); ++i) {
    if (lidar_ranges[i] > 0.01) {
      obstacle_x.push_back(lidar_ranges[i] * std::cos(lidar_angles[i]));
      obstacle_y.push_back(lidar_ranges[i] * std::sin(lidar_angles[i]));
    }
  }

  if (obstacle_x.empty() || path_points.empty()) {
    eval.score_progress = 0.0;
    eval.forward_progress = 0.0;
    eval.loop_penalty = 0.0;
    return eval;
  }

  for (const auto & point : path_points) {
    const double px = point.first;
    const double py = point.second;
    double nearest_sq = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < obstacle_x.size(); ++i) {
      const double dx = obstacle_x[i] - px;
      const double dy = obstacle_y[i] - py;
      nearest_sq = std::min(nearest_sq, dx * dx + dy * dy);
    }
    const double clearance = std::sqrt(nearest_sq) - radius;
    eval.min_clearance = std::min(eval.min_clearance, clearance);
    if (clearance < 0.0) {
      eval.collision = true;
      break;
    }
    eval.forward_progress = px;
  }

  const auto final = path_points.back();
  const double final_x = final.first;
  const double final_y = final.second;
  eval.score_progress = final_x + 0.20 * std::abs(final_y);

  double final_nearest_sq = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < obstacle_x.size(); ++i) {
    const double dx = obstacle_x[i] - final_x;
    const double dy = obstacle_y[i] - final_y;
    final_nearest_sq = std::min(final_nearest_sq, dx * dx + dy * dy);
  }
  eval.end_wall_clearance = std::sqrt(final_nearest_sq) - 2.0 * radius;

  const double world_x =
    current_x_ + final_x * std::cos(current_yaw_) - final_y * std::sin(current_yaw_);
  const double world_y =
    current_y_ + final_x * std::sin(current_yaw_) + final_y * std::cos(current_yaw_);
  const double history_penalty_radius = get_parameter("history_penalty_radius").as_double();
  const double history_penalty_weight = get_parameter("history_penalty_weight").as_double();
  for (const auto & history_point : path_history_) {
    const double dist = std::hypot(world_x - history_point.first, world_y - history_point.second);
    if (dist < history_penalty_radius) {
      eval.loop_penalty += (history_penalty_radius - dist) * history_penalty_weight;
    }
  }
  eval.loop_penalty += reverseHistoryPenaltyForEndpoint(final_x, final_y);
  eval.loop_penalty += historyPenaltyForPath(path_points);
  return eval;
}

std::pair<double, std::pair<double, double>> RacerNode::computePursuitSteering(
  const std::vector<std::pair<double, double>> & path_points,
  double lookahead)
{
  if (path_points.size() < 2) {
    return {0.0, {0.0, 0.0}};
  }
  const std::size_t target_idx = static_cast<std::size_t>(0.75 * path_points.size());
  const auto target = path_points[std::min(target_idx, path_points.size() - 1)];
  const double alpha = std::atan2(target.second, target.first);
  const double pursuit_steering = std::atan2(2.0 * WHEELBASE * std::sin(alpha), lookahead);
  return {clip(pursuit_steering, -MAX_STEER_RAD, MAX_STEER_RAD), target};
}

// double RacerNode::predictTurnDirection(const std::vector<double> & ranges)
// {
//   if (ranges.size() < 240) {
//     return 0.0;
//   }
//   const std::size_t mid = ranges.size() / 2;
//   double left_score = 0.0;
//   double right_score = 0.0;
//   constexpr std::size_t sector_len = 110;
//   for (std::size_t k = 0; k < sector_len; ++k) {
//     const double weight = 1.0 + (1.5 * k) / static_cast<double>(sector_len - 1);
//     left_score = std::max(left_score, weight * ranges[mid + 10 + k]);
//     right_score = std::max(right_score, (2.5 - (1.5 * k) / static_cast<double>(sector_len - 1)) *
//       ranges[mid - 120 + k]);
//   }
//   double left_forward = 0.0;
//   double right_forward = 0.0;
//   for (std::size_t k = 0; k < 50; ++k) {
//     left_forward += ranges[mid + 10 + k];
//     right_forward += ranges[mid - 60 + k];
//   }
//   left_forward /= 50.0;
//   right_forward /= 50.0;

//   const double drift = left_forward - right_forward;
//   const double diff = 0.7 * (left_score - right_score) + 1.8 * drift;
//   const double gain = get_parameter("turn_prediction_gain").as_double();
//   return gain * std::tanh(diff * 0.12);
// }

double RacerNode::predictTurnDirection(const std::vector<double> & ranges)
{
    if (ranges.size() < 240) return 0.0;
    const std::size_t mid = ranges.size() / 2;

    // Use mean instead of max — more stable in V-shaped corners
    double left_sum = 0.0, right_sum = 0.0;
    constexpr std::size_t sector_len = 80;  // tighter window, less wall bleed
    for (std::size_t k = 0; k < sector_len; ++k) {
        left_sum  += ranges[mid + 15 + k];
        right_sum += ranges[mid - 95 + k];
    }
    const double left_mean  = left_sum  / sector_len;
    const double right_mean = right_sum / sector_len;

    // Forward drift as before
    double left_forward = 0.0, right_forward = 0.0;
    for (std::size_t k = 0; k < 50; ++k) {
        left_forward  += ranges[mid + 10 + k];
        right_forward += ranges[mid - 60 + k];
    }
    left_forward  /= 50.0;
    right_forward /= 50.0;

    const double drift = left_forward - right_forward;
    const double diff  = 0.7 * (left_mean - right_mean) + 1.8 * drift;
    return get_parameter("turn_prediction_gain").as_double() * std::tanh(diff * 0.12);
}

double RacerNode::chooseEscapeSteering(
  const std::vector<double> & ranges,
  const std::vector<double> & angles,
  double gap_angle,
  double turn_prediction)
{
  std::vector<bool> left_mask(ranges.size(), false);
  std::vector<bool> right_mask(ranges.size(), false);
  for (std::size_t i = 0; i < ranges.size() && i < angles.size(); ++i) {
    left_mask[i] = angles[i] > 0.05 && angles[i] < 1.40;
    right_mask[i] = angles[i] < -0.05 && angles[i] > -1.40;
  }
  const double left_clearance = 0.65 * meanMasked(ranges, left_mask) + 0.35 * maxMasked(ranges, left_mask);
  const double right_clearance = 0.65 * meanMasked(ranges, right_mask) + 0.35 * maxMasked(ranges, right_mask);

  const double left_steer = 0.75 * MAX_STEER_RAD;
  const double right_steer = -0.75 * MAX_STEER_RAD;
  const double desired_angle = gap_angle + turn_prediction;
  const double alignment_weight = get_parameter("escape_alignment_weight").as_double();
  const auto left_path = generateArcPath(left_steer, 1.2);
  const auto right_path = generateArcPath(right_steer, 1.2);
  const double left_reverse_penalty = left_path.empty() ? 0.0 :
    reverseHistoryPenaltyForEndpoint(left_path.back().first, left_path.back().second);
  const double right_reverse_penalty = right_path.empty() ? 0.0 :
    reverseHistoryPenaltyForEndpoint(right_path.back().first, right_path.back().second);

  const double left_score =
    left_clearance + alignment_weight * std::cos(left_steer - desired_angle) -
    left_reverse_penalty;
  const double right_score =
    right_clearance + alignment_weight * std::cos(right_steer - desired_angle) -
    right_reverse_penalty;

  if (left_score > right_score) {
    return 0.75 * MAX_STEER_RAD;
  }
  if (right_score > left_score) {
    return -0.75 * MAX_STEER_RAD;
  }
  if (std::abs(prev_steering_) > 0.02) {
    return -std::copysign(0.75 * MAX_STEER_RAD, prev_steering_);
  }
  return 0.75 * MAX_STEER_RAD;
}

double RacerNode::chooseHardCornerSteering(
  const std::vector<double> & ranges,
  const std::vector<double> & angles,
  double gap_angle,
  double turn_prediction) const
{
  const int forced_direction = static_cast<int>(get_parameter("hard_corner_turn_direction").as_int());
  const double min_steer = get_parameter("hard_corner_min_steer").as_double();
  if (forced_direction != 0) {
    return clip(
      std::copysign(std::max(min_steer, 0.75 * MAX_STEER_RAD), static_cast<double>(forced_direction)),
      -MAX_STEER_RAD,
      MAX_STEER_RAD);
  }

  std::vector<bool> left_front_mask(ranges.size(), false);
  std::vector<bool> right_front_mask(ranges.size(), false);
  for (std::size_t i = 0; i < ranges.size() && i < angles.size(); ++i) {
    left_front_mask[i] = angles[i] > 0.20 && angles[i] < 1.25;
    right_front_mask[i] = angles[i] < -0.20 && angles[i] > -1.25;
  }

  const double left_clearance =
    0.75 * meanMasked(ranges, left_front_mask) + 0.25 * maxMasked(ranges, left_front_mask);
  const double right_clearance =
    0.75 * meanMasked(ranges, right_front_mask) + 0.25 * maxMasked(ranges, right_front_mask);
  const double sensor_direction = left_clearance >= right_clearance ? 1.0 : -1.0;
  const double predicted_direction =
    std::abs(gap_angle + turn_prediction) > 0.10 ?
    std::copysign(1.0, gap_angle + turn_prediction) :
    sensor_direction;
  const double direction =
    std::abs(left_clearance - right_clearance) > 0.15 ? sensor_direction : predicted_direction;

  return clip(
    std::copysign(std::max(min_steer, 0.85 * MAX_STEER_RAD), direction),
    -MAX_STEER_RAD,
    MAX_STEER_RAD);
}

double RacerNode::computeThrottle(
  double forward_progress,
  double clearance,
  double steering,
  double forward_min,
  double max_t,
  double min_t,
  double roll_dist)
{
  const double progress_ratio = clip(forward_progress / roll_dist, 0.0, 1.0);
  const double clearance_ratio = clip(clearance / 1.1, 0.0, 1.0);
  const double steering_ratio = std::abs(steering) / MAX_STEER_RAD;
  const double speed_score = 0.5 * progress_ratio + 0.5 * clearance_ratio;
  double target = min_t + speed_score * (max_t - min_t);
  if (forward_min < get_parameter("forward_critical_distance").as_double()) {
    target *= 0.55;
  } else if (forward_min < get_parameter("forward_warning_distance").as_double()) {
    target *= 0.72;
  }
  // } else if (forward_min < get_parameter("forward_caution_distance").as_double()) {
  //   target *= 0.88;
  // }
  target *= 1.0 - 0.42 * std::pow(steering_ratio, 1.2);
  target = clip(target, min_t, max_t);
  if (target > current_throttle_) {
    current_throttle_ = std::min(target, current_throttle_ + get_parameter("throttle_ramp_rate").as_double());
  } else {
    current_throttle_ = target;
  }
  return current_throttle_;
}

std::string RacerNode::forwardState(double forward_min)
{
  if (forward_min < get_parameter("forward_critical_distance").as_double()) {
    // RCLCPP_INFO(get_logger(), "Critical %f", forward_min);
    return "CRITICAL";
  }
  if (forward_min < get_parameter("forward_warning_distance").as_double()) {
    // RCLCPP_INFO(get_logger(), "Warning %f", forward_min);
    return "WARNING";
  }
  if (forward_min < get_parameter("forward_caution_distance").as_double()) {
    // RCLCPP_INFO(get_logger(), "Caution %f", forward_min);
    return "CAUTION";
  }
  return "CLEAR";
}

bool RacerNode::hasReverseLoopRisk(
  const std::vector<std::pair<double, double>> & path,
  double rollout_distance) const
{
  if (!get_parameter("enable_reverse_loop_guard").as_bool() || path.size() < 4) {
    return false;
  }

  const double min_forward_progress =
    get_parameter("min_forward_progress_fraction").as_double() * rollout_distance;
  const double max_forward_backtrack = get_parameter("max_forward_backtrack").as_double();
  const double final_x = path.back().first;

  double max_x = path.front().first;
  for (std::size_t i = 1; i < path.size(); ++i) {
    max_x = std::max(max_x, path[i].first);
  }

  const std::size_t heading_start_idx = path.size() * 3 / 4;
  const auto & heading_start = path[heading_start_idx];
  const double heading_dx = final_x - heading_start.first;
  const double heading_dy = path.back().second - heading_start.second;
  const double heading_norm = std::hypot(heading_dx, heading_dy);
  const double forward_heading = heading_norm > 0.01 ? heading_dx / heading_norm : 1.0;
  const double backtrack = max_x - final_x;
  const bool clearly_heading_back = forward_heading < -0.25;
  const bool curled_back = clearly_heading_back && backtrack > max_forward_backtrack;
  const bool ended_behind_start = clearly_heading_back && final_x < 0.0;
  const bool stalled_after_big_loop =
    final_x < min_forward_progress && backtrack > 2.0 * max_forward_backtrack;

  return curled_back || ended_behind_start || stalled_after_big_loop;
}

double RacerNode::historyPenaltyForPath(
  const std::vector<std::pair<double, double>> & path) const
{
  if (path_history_.size() < 8 || path.empty()) {
    return 0.0;
  }

  const double history_penalty_radius = get_parameter("history_penalty_radius").as_double();
  const double history_path_penalty_weight = get_parameter("history_path_penalty_weight").as_double();
  if (history_penalty_radius <= 0.0 || history_path_penalty_weight <= 0.0) {
    return 0.0;
  }

  double penalty = 0.0;
  const std::size_t path_start = path.size() / 4;
  const std::size_t history_end = path_history_.size() > 8 ? path_history_.size() - 8 : 0;
  const std::size_t stride = std::max<std::size_t>(1, path.size() / 12);
  for (std::size_t i = path_start; i < path.size(); i += stride) {
    const double local_x = path[i].first;
    const double local_y = path[i].second;
    const double world_x =
      current_x_ + local_x * std::cos(current_yaw_) - local_y * std::sin(current_yaw_);
    const double world_y =
      current_y_ + local_x * std::sin(current_yaw_) + local_y * std::cos(current_yaw_);

    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t h = 0; h < history_end; ++h) {
      const auto & history_point = path_history_[h];
      nearest = std::min(
        nearest,
        std::hypot(world_x - history_point.first, world_y - history_point.second));
    }
    if (nearest < history_penalty_radius) {
      const double proximity = (history_penalty_radius - nearest) / history_penalty_radius;
      penalty += proximity * history_path_penalty_weight;
    }
  }

  return penalty;
}

double RacerNode::reverseHistoryPenaltyForEndpoint(double local_x, double local_y) const
{
  if (path_history_.size() < 8) {
    return 0.0;
  }

  const auto & recent = path_history_.back();
  const auto & older = path_history_[path_history_.size() - 8];
  const double motion_x = recent.first - older.first;
  const double motion_y = recent.second - older.second;
  const double motion_norm = std::hypot(motion_x, motion_y);
  if (motion_norm < 0.05) {
    return 0.0;
  }

  const double world_x =
    current_x_ + local_x * std::cos(current_yaw_) - local_y * std::sin(current_yaw_);
  const double world_y =
    current_y_ + local_x * std::sin(current_yaw_) + local_y * std::cos(current_yaw_);
  const double endpoint_x = world_x - current_x_;
  const double endpoint_y = world_y - current_y_;
  const double endpoint_norm = std::hypot(endpoint_x, endpoint_y);
  if (endpoint_norm < 0.05) {
    return 0.0;
  }

  const double alignment =
    (motion_x * endpoint_x + motion_y * endpoint_y) / (motion_norm * endpoint_norm);
  const double reverse_amount = std::max(-alignment, 0.0);
  if (reverse_amount <= 0.0) {
    return 0.0;
  }

  const double history_penalty_radius = get_parameter("history_penalty_radius").as_double();
  double proximity = 0.0;
  for (const auto & history_point : path_history_) {
    const double dist = std::hypot(world_x - history_point.first, world_y - history_point.second);
    if (dist < history_penalty_radius) {
      proximity = std::max(proximity, (history_penalty_radius - dist) / history_penalty_radius);
    }
  }

  return get_parameter("reverse_history_penalty_weight").as_double() *
    reverse_amount * proximity;
}

void RacerNode::publishProcessedLidar(
  const sensor_msgs::msg::LaserScan & scan,
  const std::vector<double> & processed_ranges)
{
  sensor_msgs::msg::LaserScan processed_scan;
  processed_scan.header = scan.header;
  processed_scan.angle_min = scan.angle_min + lidar_start_idx_ * scan.angle_increment;
  processed_scan.angle_max = scan.angle_min + (lidar_end_idx_ - 1) * scan.angle_increment;
  processed_scan.angle_increment = scan.angle_increment;
  processed_scan.time_increment = scan.time_increment;
  processed_scan.scan_time = scan.scan_time;
  processed_scan.range_min = scan.range_min;
  processed_scan.range_max = std::min(scan.range_max, static_cast<float>(max_range_cap_));
  processed_scan.ranges.reserve(processed_ranges.size());
  for (const double range : processed_ranges) {
    processed_scan.ranges.push_back(static_cast<float>(range));
  }
  processed_lidar_pub_->publish(processed_scan);
}

void RacerNode::publishDebugMarkers(
  const sensor_msgs::msg::LaserScan & scan,
  const std::vector<CandidatePath> & paths,
  const std::vector<std::pair<double, double>> & best_path,
  const std::pair<double, double> * pursuit_target,
  const std::vector<double> & raw_ranges,
  const std::vector<double> & processed_ranges,
  const std::vector<double> & angles,
  const GapDebug & gap_debug,
  const DecisionDebug & decision_debug)
{
  visualization_msgs::msg::MarkerArray ma;
  ma.markers.reserve(paths.size() + 24);

  int top_count = static_cast<int>(get_parameter("debug_top_candidate_count").as_int());
  top_count = std::max(0, std::min(top_count, std::min(static_cast<int>(paths.size()), 8)));
  std::vector<std::size_t> ranked(paths.size());
  std::iota(ranked.begin(), ranked.end(), 0);
  std::sort(ranked.begin(), ranked.end(), [&paths](std::size_t a, std::size_t b) {
    return paths[a].score > paths[b].score;
  });

  std::vector<int> top_rank(paths.size(), -1);
  for (int rank = 0; rank < top_count; ++rank) {
    top_rank[ranked[static_cast<std::size_t>(rank)]] = rank;
  }

  visualization_msgs::msg::Marker raw_scan;
  raw_scan.header = scan.header;
  raw_scan.header.stamp = rclcpp::Time(0, 0, get_clock()->get_clock_type());
  raw_scan.ns = "lidar_raw_slice";
  raw_scan.id = 0;
  raw_scan.type = visualization_msgs::msg::Marker::LINE_STRIP;
  raw_scan.action = visualization_msgs::msg::Marker::ADD;
  raw_scan.scale.x = 0.018;
  raw_scan.color = makeColor(0.45, 0.45, 0.45, 0.45);

  visualization_msgs::msg::Marker processed_scan;
  processed_scan.header = raw_scan.header;
  processed_scan.ns = "lidar_processed_mask";
  processed_scan.id = 0;
  processed_scan.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  processed_scan.action = visualization_msgs::msg::Marker::ADD;
  processed_scan.scale.x = 0.045;
  processed_scan.scale.y = 0.045;
  processed_scan.scale.z = 0.045;

  constexpr std::size_t stride = 3;
  for (std::size_t i = 0; i < angles.size() && i < raw_ranges.size() &&
    i < processed_ranges.size() && i < gap_debug.free_mask.size(); i += stride)
  {
    raw_scan.points.push_back(makePoint(
      raw_ranges[i] * std::cos(angles[i]),
      raw_ranges[i] * std::sin(angles[i]),
      0.025));
    processed_scan.points.push_back(makePoint(
      processed_ranges[i] * std::cos(angles[i]),
      processed_ranges[i] * std::sin(angles[i]),
      0.08));
    processed_scan.colors.push_back(
      gap_debug.free_mask[i] ? makeColor(0.0, 0.95, 0.25, 0.85) :
      makeColor(1.0, 0.20, 0.05, 0.85));
  }
  ma.markers.push_back(raw_scan);
  ma.markers.push_back(processed_scan);

  visualization_msgs::msg::Marker threshold_arc;
  threshold_arc.header = raw_scan.header;
  threshold_arc.ns = "lidar_gap_threshold";
  threshold_arc.id = 0;
  threshold_arc.type = visualization_msgs::msg::Marker::LINE_STRIP;
  threshold_arc.action = visualization_msgs::msg::Marker::ADD;
  threshold_arc.scale.x = 0.025;
  threshold_arc.color = makeColor(1.0, 0.80, 0.05, 0.75);
  for (std::size_t i = 0; i < angles.size(); i += 8) {
    threshold_arc.points.push_back(makePoint(
      gap_debug.threshold * std::cos(angles[i]),
      gap_debug.threshold * std::sin(angles[i]),
      0.05));
  }
  ma.markers.push_back(threshold_arc);

  const std::array<double, 3> vector_angles{
    gap_debug.smoothed_angle,
    decision_debug.turn_prediction,
    decision_debug.best_steering};
  const std::array<std_msgs::msg::ColorRGBA, 3> vector_colors{
    makeColor(0.0, 1.0, 0.20, 1.0),
    makeColor(0.10, 0.55, 1.0, 1.0),
    makeColor(1.0, 1.0, 1.0, 1.0)};
  for (int id = 0; id < 3; ++id) {
    visualization_msgs::msg::Marker arrow;
    arrow.header = raw_scan.header;
    arrow.ns = "decision_vectors";
    arrow.id = id;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    arrow.points.push_back(makePoint(0.0, 0.0, 0.33));
    arrow.points.push_back(makePoint(
      1.25 * std::cos(vector_angles[static_cast<std::size_t>(id)]),
      1.25 * std::sin(vector_angles[static_cast<std::size_t>(id)]),
      0.33));
    arrow.scale.x = 0.035;
    arrow.scale.y = 0.11;
    arrow.scale.z = 0.14;
    arrow.color = vector_colors[static_cast<std::size_t>(id)];
    ma.markers.push_back(arrow);
  }

  for (std::size_t i = 0; i < paths.size(); ++i) {
    visualization_msgs::msg::Marker path_marker;
    path_marker.header = raw_scan.header;
    path_marker.ns = "paths";
    path_marker.id = static_cast<int>(i);
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.scale.x = 0.012;
    if (top_rank[i] >= 0 && !paths[i].collision) {
      const int rank = top_rank[i];
      path_marker.scale.x = 0.020 + 0.010 * (top_count - rank);
      path_marker.color = makeColor(0.05 * rank, 1.0, 0.15, 0.28);
    } else if (paths[i].collision) {
      path_marker.color = makeColor(1.0, 0.0, 0.0, 0.18);
    } else {
      path_marker.color = makeColor(0.0, 0.0, 1.0, 0.08);
    }
    path_marker.points.reserve(paths[i].path.size());
    for (const auto & point : paths[i].path) {
      path_marker.points.push_back(makePoint(point.first, point.second, 0.01));
    }
    ma.markers.push_back(path_marker);
  }

  for (int label_id = 0; label_id < top_count; ++label_id) {
    const auto & candidate = paths[ranked[static_cast<std::size_t>(label_id)]];
    if (candidate.path.empty()) {
      continue;
    }
    visualization_msgs::msg::Marker label;
    label.header = raw_scan.header;
    label.ns = "candidate_scores";
    label.id = label_id;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    const auto end = candidate.path.back();
    label.pose.position.x = end.first;
    label.pose.position.y = end.second;
    label.pose.position.z = 0.45 + 0.04 * label_id;
    label.scale.z = 0.11;
    label.color = makeColor(1.0, 1.0, 1.0, 0.95);
    std::ostringstream text;
    text << "#" << (label_id + 1) << " " << candidate.score << "\n"
         << "steer " << (candidate.steer * 180.0 / M_PI) << " deg\n"
         << "clr " << candidate.clearance << " end " << candidate.end_wall_clearance;
    label.text = text.str();
    ma.markers.push_back(label);
  }
  for (int label_id = top_count; label_id < 8; ++label_id) {
    visualization_msgs::msg::Marker label;
    label.header = raw_scan.header;
    label.ns = "candidate_scores";
    label.id = label_id;
    label.action = visualization_msgs::msg::Marker::DELETE;
    ma.markers.push_back(label);
  }

  if (!best_path.empty()) {
    visualization_msgs::msg::Marker outline;
    outline.header = raw_scan.header;
    outline.ns = "selected_path";
    outline.id = 2;
    outline.type = visualization_msgs::msg::Marker::LINE_STRIP;
    outline.action = visualization_msgs::msg::Marker::ADD;
    outline.scale.x = 0.145;
    outline.color = makeColor(0.0, 0.0, 0.0, 0.95);
    for (const auto & point : best_path) {
      outline.points.push_back(makePoint(point.first, point.second, 0.14));
    }
    ma.markers.push_back(outline);

    visualization_msgs::msg::Marker selected = outline;
    selected.id = 0;
    selected.scale.x = 0.095;
    selected.color = makeColor(0.0, 1.0, 0.18, 1.0);
    selected.points.clear();
    for (const auto & point : best_path) {
      selected.points.push_back(makePoint(point.first, point.second, 0.16));
    }
    ma.markers.push_back(selected);

    visualization_msgs::msg::Marker beads;
    beads.header = raw_scan.header;
    beads.ns = "selected_path";
    beads.id = 3;
    beads.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    beads.action = visualization_msgs::msg::Marker::ADD;
    beads.scale.x = 0.075;
    beads.scale.y = 0.075;
    beads.scale.z = 0.075;
    beads.color = makeColor(1.0, 1.0, 1.0, 1.0);
    const std::size_t bead_stride = std::max<std::size_t>(1, best_path.size() / 12);
    for (std::size_t i = 0; i < best_path.size(); i += bead_stride) {
      beads.points.push_back(makePoint(best_path[i].first, best_path[i].second, 0.20));
    }
    ma.markers.push_back(beads);

    if (best_path.size() >= 2) {
      visualization_msgs::msg::Marker arrow;
      arrow.header = raw_scan.header;
      arrow.ns = "selected_path";
      arrow.id = 4;
      arrow.type = visualization_msgs::msg::Marker::ARROW;
      arrow.action = visualization_msgs::msg::Marker::ADD;
      const auto start = best_path[best_path.size() - 2];
      const auto end = best_path.back();
      arrow.points.push_back(makePoint(start.first, start.second, 0.22));
      arrow.points.push_back(makePoint(end.first, end.second, 0.22));
      arrow.scale.x = 0.055;
      arrow.scale.y = 0.17;
      arrow.scale.z = 0.20;
      arrow.color = makeColor(1.0, 1.0, 1.0, 1.0);
      ma.markers.push_back(arrow);
    }
  } else {
    for (const int id : {0, 2, 3, 4}) {
      visualization_msgs::msg::Marker marker;
      marker.header = raw_scan.header;
      marker.ns = "selected_path";
      marker.id = id;
      marker.action = visualization_msgs::msg::Marker::DELETE;
      ma.markers.push_back(marker);
    }
  }

  if (pursuit_target != nullptr) {
    visualization_msgs::msg::Marker target;
    target.header = raw_scan.header;
    target.ns = "selected_path";
    target.id = 1;
    target.type = visualization_msgs::msg::Marker::SPHERE;
    target.action = visualization_msgs::msg::Marker::ADD;
    target.pose.position.x = pursuit_target->first;
    target.pose.position.y = pursuit_target->second;
    target.pose.position.z = 0.26;
    target.scale.x = 0.20;
    target.scale.y = 0.20;
    target.scale.z = 0.20;
    target.color = makeColor(1.0, 1.0, 1.0, 1.0);
    ma.markers.push_back(target);
  } else {
    visualization_msgs::msg::Marker target;
    target.header = raw_scan.header;
    target.ns = "selected_path";
    target.id = 1;
    target.action = visualization_msgs::msg::Marker::DELETE;
    ma.markers.push_back(target);
  }

  visualization_msgs::msg::Marker debug_text;
  debug_text.header = raw_scan.header;
  debug_text.ns = "decision_readout";
  debug_text.id = 0;
  debug_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  debug_text.action = visualization_msgs::msg::Marker::ADD;
  debug_text.pose.position.x = 0.60;
  debug_text.pose.position.y = -1.35;
  debug_text.pose.position.z = 0.80;
  debug_text.scale.z = 0.13;
  debug_text.color = makeColor(1.0, 1.0, 1.0, 1.0);
  const auto & b = decision_debug.breakdown;
  std::ostringstream text;
  text << "Decision: " << b.reason << "\n"
       << "steer " << (decision_debug.best_steering * 180.0 / M_PI)
       << " deg  throttle " << decision_debug.throttle << " " << decision_debug.throttle_state
       << "\nscore " << decision_debug.best_score
       << "  clearance " << decision_debug.best_clearance
       << "  score_prog " << decision_debug.best_progress
       << "  fwd_prog " << decision_debug.best_forward_progress
       << "\ngap " << (decision_debug.gap_angle * 180.0 / M_PI)
       << " deg  turn bias " << (decision_debug.turn_prediction * 180.0 / M_PI)
       << " deg  front min " << decision_debug.forward_min
       << "\ncomponents: prog " << b.progress_score
       << " clr " << b.clearance_score
       << " low " << b.low_clearance_score
       << " end " << b.wall_clearance_score
       << " gap " << b.gap_alignment
       << " commit " << b.turn_commitment
       << "\nsmooth " << b.smoothness_score
       << " steer " << b.steering_penalty
       << " loop -" << b.loop_penalty
       << " anticip " << b.anticipation_bonus;
  debug_text.text = text.str();
  ma.markers.push_back(debug_text);

  marker_pub_->publish(ma);
}

geometry_msgs::msg::Point RacerNode::makePoint(double x, double y, double z) const
{
  geometry_msgs::msg::Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  return point;
}

std_msgs::msg::ColorRGBA RacerNode::makeColor(double r, double g, double b, double a) const
{
  std_msgs::msg::ColorRGBA color;
  color.r = static_cast<float>(r);
  color.g = static_cast<float>(g);
  color.b = static_cast<float>(b);
  color.a = static_cast<float>(a);
  return color;
}

}  // namespace custom_codes_cpp
