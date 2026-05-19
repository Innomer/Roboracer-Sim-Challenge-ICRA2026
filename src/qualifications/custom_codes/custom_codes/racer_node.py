import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Imu, JointState
from std_msgs.msg import Float32

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import numpy as np


class RacerNode(Node):

    CAR_WIDTH = 0.2700
    WHEELBASE = 0.3240
    TRACK_WIDTH = 0.2360
    WHEEL_RADIUS = 0.0590

    LIDAR_OFFSET_X = 0.0

    ENCODER_PPR = 16
    ENCODER_CR = 120

    TICKS_PER_REV = ENCODER_PPR * ENCODER_CR

    WHEEL_CIRC = 2.0 * np.pi * WHEEL_RADIUS
    DIST_PER_TICK = WHEEL_CIRC / TICKS_PER_REV

    MAX_STEER_RAD = 0.5236

    LIDAR_RAYS = 1080

    def __init__(self):

        super().__init__("robust_rollout_racer")

        self.declare_parameter("max_throttle", 0.205)
        self.declare_parameter("min_throttle", 0.02)
        self.declare_parameter("throttle_ramp_rate", 0.02)

        self.declare_parameter("safety_margin", 0.075)

        self.declare_parameter("pursuit_lookahead", 1.7)
        self.declare_parameter("pursuit_blend", 0.60)

        self.declare_parameter("progress_weight", 3.0)
        self.declare_parameter("clearance_weight", 1.5)
        self.declare_parameter("smoothness_weight", -1.2)
        self.declare_parameter("steering_penalty_weight", -0.4)
        self.declare_parameter("gap_alignment_weight", 1.2)
        self.declare_parameter("turn_commitment_weight", 3.0)

        self.declare_parameter("rollout_distance", 2.5)

        self.declare_parameter("max_steering_delta", 0.14)
        self.declare_parameter("temporal_blend", 0.15)
        self.declare_parameter("disparity_threshold", 0.50)
        self.declare_parameter("max_jump_distance", 1.5)

        self.declare_parameter("history_penalty_radius", 1.2)
        self.declare_parameter("history_penalty_weight", 4.0)

        self.declare_parameter("gap_threshold_scale", 0.45)
        self.declare_parameter("max_gap_threshold", 2.5)

        self.declare_parameter("turn_prediction_gain", 0.25)

        self.declare_parameter("forward_critical_distance", 0.50)
        self.declare_parameter("forward_warning_distance", 0.80)
        self.declare_parameter("forward_caution_distance", 1.20)

        self.lidar_sub = self.create_subscription(
            LaserScan,
            "/autodrive/roboracer_1/lidar",
            self.lidar_callback,
            10,
        )

        self.imu_sub = self.create_subscription(
            Imu,
            "/autodrive/roboracer_1/imu",
            self.imu_callback,
            10,
        )

        self.left_enc_sub = self.create_subscription(
            JointState,
            "/autodrive/roboracer_1/left_encoder",
            self.left_encoder_callback,
            10,
        )

        self.right_enc_sub = self.create_subscription(
            JointState,
            "/autodrive/roboracer_1/right_encoder",
            self.right_encoder_callback,
            10,
        )

        self.steer_pub = self.create_publisher(
            Float32,
            "/autodrive/roboracer_1/steering_command",
            10,
        )

        self.throttle_pub = self.create_publisher(
            Float32,
            "/autodrive/roboracer_1/throttle_command",
            10,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/autodrive/debug_markers",
            10,
        )
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self._left_ticks_prev = None
        self._right_ticks_prev = None

        self._left_ticks = 0.0
        self._right_ticks = 0.0

        self._imu_yaw_rate = 0.0

        self._last_odom_time = self.get_clock().now()

        self.path_history = []
        self.max_history = 120

        self.prev_steering = 0.0

        self.max_range_cap = 9.0

        self.prev_processed_ranges = None

        self.lidar_start_idx = 240
        self.lidar_end_idx = 840

        self._validate_lidar_slice()

        self._current_throttle = self.get_parameter("min_throttle").value

        self.get_logger().info("Updated Rollout Racer Initialized")

    def imu_callback(self, msg: Imu):
        self._imu_yaw_rate = msg.angular_velocity.z

    def left_encoder_callback(self, msg: JointState):
        if msg.position:
            self._left_ticks = float(msg.position[0])

    def right_encoder_callback(self, msg: JointState):
        if msg.position:
            self._right_ticks = float(msg.position[0])

    def _validate_lidar_slice(self):

        centre_ray = self.LIDAR_RAYS // 2
        half_arc_idx = int(75.0 / 0.25)

        self.lidar_start_idx = max(
            0,
            centre_ray - half_arc_idx,
        )

        self.lidar_end_idx = min(
            self.LIDAR_RAYS,
            centre_ray + half_arc_idx,
        )

    def _update_odometry(self):

        now = self.get_clock().now()

        dt = (now - self._last_odom_time).nanoseconds * 1e-9

        self._last_odom_time = now

        if dt <= 0 or dt > 0.5:
            return

        if self._left_ticks_prev is None:

            self._left_ticks_prev = self._left_ticks
            self._right_ticks_prev = self._right_ticks

            return

        d_left = (self._left_ticks - self._left_ticks_prev) * self.DIST_PER_TICK

        d_right = (self._right_ticks - self._right_ticks_prev) * self.DIST_PER_TICK

        self._left_ticks_prev = self._left_ticks
        self._right_ticks_prev = self._right_ticks

        d_center = (d_left + d_right) / 2.0

        yaw_enc = (d_right - d_left) / self.TRACK_WIDTH

        yaw_imu = self._imu_yaw_rate * dt

        d_yaw = 0.60 * yaw_enc + 0.40 * yaw_imu

        heading = self.current_yaw + d_yaw / 2.0

        self.current_x += d_center * np.cos(heading)
        self.current_y += d_center * np.sin(heading)

        self.current_yaw += d_yaw

        self.path_history.append((self.current_x, self.current_y))

        if len(self.path_history) > self.max_history:
            self.path_history.pop(0)

    def lidar_callback(self, data: LaserScan):

        self._update_odometry()

        # --- PARAMS ---

        max_throttle = self.get_parameter("max_throttle").value

        min_throttle = self.get_parameter("min_throttle").value

        safety_margin = self.get_parameter("safety_margin").value

        pursuit_lookahead = self.get_parameter("pursuit_lookahead").value

        pursuit_blend = self.get_parameter("pursuit_blend").value

        rollout_distance = self.get_parameter("rollout_distance").value

        w_progress = self.get_parameter("progress_weight").value

        w_clearance = self.get_parameter("clearance_weight").value

        w_smoothness = self.get_parameter("smoothness_weight").value

        w_steering = self.get_parameter("steering_penalty_weight").value

        w_gap = self.get_parameter("gap_alignment_weight").value

        w_commitment = self.get_parameter("turn_commitment_weight").value

        vehicle_radius = (self.CAR_WIDTH / 2.0) + safety_margin

        raw_ranges = np.array(data.ranges)

        raw_ranges = np.nan_to_num(
            raw_ranges,
            nan=0.0,
            posinf=self.max_range_cap,
            neginf=0.0,
        )

        raw_ranges = np.clip(
            raw_ranges,
            0.0,
            self.max_range_cap,
        )

        ranges = raw_ranges[self.lidar_start_idx : self.lidar_end_idx]

        angle_min = data.angle_min + self.lidar_start_idx * data.angle_increment

        angle_inc = data.angle_increment

        angles = angle_min + np.arange(len(ranges)) * angle_inc

        ranges = self.remove_double_wall_artifacts(ranges)
        ranges = self.filter_duct_gap_artifacts(ranges)

        processed_ranges = self.apply_disparity_extender(
            ranges,
            angle_inc,
            vehicle_radius,
        )

        if self.prev_processed_ranges is not None and len(
            self.prev_processed_ranges
        ) == len(processed_ranges):

            temporal_blend = self.get_parameter("temporal_blend").value

            processed_ranges = (
                temporal_blend * self.prev_processed_ranges
                + (1.0 - temporal_blend) * processed_ranges
            )

        self.prev_processed_ranges = processed_ranges.copy()

        mid = len(processed_ranges) // 2

        forward_min = np.min(processed_ranges[mid - 60 : mid + 60])

        (
            best_score,
            best_steering,
            best_progress,
            best_clearance,
            best_path_points,
        ) = (
            -999999.0,
            0.0,
            0.0,
            0.0,
            None,
        )

        candidate_paths = []

        candidate_steers = np.linspace(
            -self.MAX_STEER_RAD,
            self.MAX_STEER_RAD,
            41,
        )

        best_gap_angle = self._find_best_gap_angle(
            processed_ranges,
            angles,
        )

        turn_prediction = self.predict_turn_direction(processed_ranges)

        for steering_angle in candidate_steers:

            path_points = self.generate_arc_path(
                steering_angle,
                rollout_distance,
            )

            (
                collision,
                clearance,
                progress,
                loop_penalty,
            ) = self.evaluate_path(
                path_points,
                processed_ranges,
                angles,
                vehicle_radius,
            )

            if collision:
                score = -9999.0

            else:

                progress_score = w_progress * progress

                clearance_score = w_clearance * clearance

                smoothness_score = w_smoothness * abs(
                    steering_angle - self.prev_steering
                )

                steering_penalty = w_steering * abs(steering_angle)

                anticipation_bonus = 1.0 * np.cos(steering_angle - turn_prediction)

                predicted_angle = best_gap_angle + turn_prediction

                gap_alignment = w_gap * np.cos(steering_angle - predicted_angle)

                turn_commitment = 0.0

                if abs(best_gap_angle) > 0.15:

                    turn_commitment = w_commitment * (
                        1.0 - abs(steering_angle - best_gap_angle)
                    )

                score = (
                    progress_score
                    + clearance_score
                    + smoothness_score
                    + steering_penalty
                    + gap_alignment
                    - loop_penalty
                    + turn_commitment
                    + anticipation_bonus
                )

            candidate_paths.append(
                (
                    path_points,
                    score,
                    steering_angle,
                    collision,
                )
            )

            if score > best_score:

                (
                    best_score,
                    best_steering,
                    best_progress,
                    best_clearance,
                    best_path_points,
                ) = (
                    score,
                    steering_angle,
                    progress,
                    clearance,
                    path_points,
                )

        pursuit_target = None

        if best_path_points is not None:

            (
                pursuit_steer,
                pursuit_target,
            ) = self.compute_pursuit_steering(
                best_path_points,
                pursuit_lookahead,
            )

            best_steering = (
                1.0 - pursuit_blend
            ) * best_steering + pursuit_blend * pursuit_steer

        max_steering_delta = self.get_parameter("max_steering_delta").value

        delta = np.clip(
            best_steering - self.prev_steering,
            -max_steering_delta,
            max_steering_delta,
        )

        best_steering = self.prev_steering + delta

        self.prev_steering = best_steering

        throttle = self._compute_throttle(
            best_progress,
            best_clearance,
            best_steering,
            forward_min,
            max_throttle,
            min_throttle,
            rollout_distance,
        )

        self.publish_debug_markers(
            data,
            candidate_paths,
            best_steering,
            pursuit_target,
        )

        self.steer_pub.publish(
            Float32(
                data=float(
                    np.clip(
                        best_steering / self.MAX_STEER_RAD,
                        -1.0,
                        1.0,
                    )
                )
            )
        )

        self.throttle_pub.publish(Float32(data=float(throttle)))

    def compute_pursuit_steering(self, path_points, lookahead):

        if len(path_points) < 2:
            return 0.0, None

        target_idx = int(0.75 * len(path_points))

        tx, ty = path_points[target_idx]

        alpha = np.arctan2(ty, tx)

        pursuit_steering = np.arctan2(
            2.0 * self.WHEELBASE * np.sin(alpha),
            lookahead,
        )

        return (
            np.clip(
                pursuit_steering,
                -self.MAX_STEER_RAD,
                self.MAX_STEER_RAD,
            ),
            (tx, ty),
        )

    def predict_turn_direction(self, ranges):

        mid = len(ranges) // 2

        left_sector = ranges[mid + 10 : mid + 120]
        right_sector = ranges[mid - 120 : mid - 10]

        weights = np.linspace(
            1.0,
            2.5,
            len(left_sector),
        )

        left_score = np.max(weights * left_sector)

        right_score = np.max(weights[::-1] * right_sector)

        # Corridor drift
        left_forward = np.mean(ranges[mid + 10 : mid + 60])

        right_forward = np.mean(ranges[mid - 60 : mid - 10])

        drift = left_forward - right_forward

        diff = 0.7 * (left_score - right_score) + 1.8 * drift

        gain = self.get_parameter("turn_prediction_gain").value

        bias = gain * np.tanh(diff * 0.12)

        if bias > 0.05:
            turn_prediction = "LEFT"

        elif bias < -0.05:
            turn_prediction = "RIGHT"

        else:
            turn_prediction = "STRAIGHT"

        self.get_logger().debug(
            f"[TURN PREDICT] "
            f"{turn_prediction} | "
            f"drift={drift:.2f} | "
            f"diff={diff:.2f} | "
            f"bias={bias:.3f}"
        )

        return bias

    def _compute_throttle(
        self,
        progress,
        clearance,
        steering,
        forward_min,
        max_t,
        min_t,
        roll_dist,
    ):

        progress_ratio = progress / roll_dist

        clearance_ratio = np.clip(
            clearance / 2.0,
            0.0,
            1.0,
        )

        steering_ratio = abs(steering) / self.MAX_STEER_RAD

        speed_score = (
            0.50 * progress_ratio
            + 0.5 * clearance_ratio
            + 0.0 * (1.0 - steering_ratio)
        )
        # self.get_logger().info(f"{speed_score}, {progress_ratio}, {clearance_ratio}, {1.0 -steering_ratio}")

        target = min_t + speed_score * (max_t - min_t)

        # if forward_min < self.get_parameter("forward_critical_distance").value:
        #     self.get_logger().info("Critical")
        #     target *= 0.5

        # elif forward_min < self.get_parameter("forward_warning_distance").value:
        #     self.get_logger().info("Warning")
        #     target *= 0.85

        # elif forward_min < self.get_parameter("forward_caution_distance").value:
        #     self.get_logger().info("Caution")
        #     target *= 0.9

        target *= 1.0 - 0.45 * (abs(steering) / self.MAX_STEER_RAD) ** 1.5

        target = np.clip(target, min_t, max_t)

        if target > self._current_throttle:

            self._current_throttle = min(
                target,
                self._current_throttle + self.get_parameter("throttle_ramp_rate").value,
            )

        else:
            self._current_throttle = target

        return float(self._current_throttle)

    def remove_double_wall_artifacts(self, ranges):

        filtered = ranges.copy()

        jump_distance = self.get_parameter("max_jump_distance").value

        for i in range(2, len(ranges) - 2):

            if (
                abs(ranges[i] - ranges[i - 1]) > jump_distance
                and abs(ranges[i] - ranges[i + 1]) > jump_distance
            ):

                filtered[i] = (ranges[i - 1] + ranges[i + 1]) / 2.0

        smoothed = filtered.copy()

        for i in range(2, len(filtered) - 2):
            smoothed[i] = np.median(filtered[i - 2 : i + 3])

        return smoothed

    def filter_duct_gap_artifacts(self, ranges):

        result = ranges.copy()

        gap_thresh = 0.80 * self.max_range_cap
        wall_near = 2.0

        for i in range(2, len(ranges) - 2):

            if ranges[i] > gap_thresh:

                neighbors = [
                    ranges[i - 2],
                    ranges[i - 1],
                    ranges[i + 1],
                    ranges[i + 2],
                ]

                if sum(1 for r in neighbors if r < wall_near) >= 3:

                    result[i] = np.mean([r for r in neighbors if r < wall_near])

        return result

    def apply_disparity_extender(
        self,
        ranges,
        angle_inc,
        radius,
    ):

        new_ranges = ranges.copy()

        disparity_threshold = self.get_parameter("disparity_threshold").value

        for i in range(1, len(ranges)):

            diff = ranges[i] - ranges[i - 1]

            if abs(diff) > disparity_threshold:

                dist = min(ranges[i], ranges[i - 1])

                bubble_angle = np.arctan2(
                    radius,
                    max(dist, 0.1),
                )

                num_indices = int(bubble_angle / angle_inc)

                if diff > 0:

                    start = i
                    end = min(
                        i + num_indices,
                        len(ranges),
                    )

                else:

                    start = max(
                        0,
                        i - num_indices,
                    )

                    end = i

                for j in range(start, end):

                    reduction = 0.92 - 0.25 * (abs(j - i) / max(1, num_indices))

                    new_ranges[j] = min(
                        new_ranges[j],
                        dist * reduction,
                    )

        return new_ranges

    def _find_best_gap_angle(self, ranges, angles):

        threshold = min(
            np.max(ranges) * self.get_parameter("gap_threshold_scale").value,
            self.get_parameter("max_gap_threshold").value,
        )

        free_mask = ranges > threshold

        best_start = 0
        best_len = 0

        cur_start = 0
        cur_len = 0

        for i, free in enumerate(free_mask):

            if free:

                if cur_len == 0:
                    cur_start = i

                cur_len += 1

                if cur_len > best_len:

                    best_len = cur_len
                    best_start = cur_start

            else:
                cur_len = 0

        if best_len > 0:
            return float(angles[best_start + best_len // 2])

        return 0.0

    def generate_arc_path(self, steering_angle, distance):

        path = []

        x = 0.0
        y = 0.0
        yaw = 0.0
        d = 0.0

        while d < distance:

            x += 0.08 * np.cos(yaw)
            y += 0.08 * np.sin(yaw)

            yaw += 0.08 * np.tan(steering_angle) / self.WHEELBASE

            path.append((x, y))

            d += 0.08

        return path

    def evaluate_path(
        self,
        path_points,
        lidar_ranges,
        lidar_angles,
        radius,
    ):

        min_clearance = 999.0
        collision = False
        progress = 0.0

        for px, py in path_points:

            point_dist = np.hypot(px, py)
            point_angle = np.arctan2(py, px)

            lidar_idx = np.argmin(np.abs(lidar_angles - point_angle))

            clearance = lidar_ranges[lidar_idx] - point_dist

            min_clearance = min(
                min_clearance,
                clearance,
            )

            if clearance < radius:

                collision = True

                break

            progress = point_dist

        final_x, final_y = path_points[-1]

        world_x = (
            self.current_x
            + final_x * np.cos(self.current_yaw)
            - final_y * np.sin(self.current_yaw)
        )

        world_y = (
            self.current_y
            + final_x * np.sin(self.current_yaw)
            + final_y * np.cos(self.current_yaw)
        )

        history_penalty_radius = self.get_parameter("history_penalty_radius").value

        history_penalty_weight = self.get_parameter("history_penalty_weight").value

        loop_penalty = sum(
            (
                history_penalty_radius
                - np.hypot(
                    world_x - hx,
                    world_y - hy,
                )
            )
            * history_penalty_weight
            for hx, hy in self.path_history
            if np.hypot(
                world_x - hx,
                world_y - hy,
            )
            < history_penalty_radius
        )

        return (
            collision,
            min_clearance,
            progress,
            loop_penalty,
        )

    def publish_debug_markers(
        self,
        scan,
        paths,
        best_steer,
        pursuit_target,
    ):

        ma = MarkerArray()

        for i, (
            path,
            score,
            steer,
            coll,
        ) in enumerate(paths):

            m = Marker(
                header=scan.header,
                ns="paths",
                id=i,
                type=Marker.LINE_STRIP,
                action=Marker.ADD,
            )

            m.header.stamp.sec = 0
            m.header.stamp.nanosec = 0

            m.scale.x = 0.025

            if abs(steer - best_steer) < 0.001:

                m.color.g = 1.0
                m.color.a = 1.0

            elif coll:

                m.color.r = 1.0
                m.color.a = 0.35

            else:

                m.color.b = 1.0
                m.color.a = 0.18

            for px, py in path:
                m.points.append(
                    Point(
                        x=px,
                        y=py,
                        z=0.02,
                    )
                )

            ma.markers.append(m)

        if pursuit_target:

            m = Marker(
                header=scan.header,
                ns="pursuit",
                id=999,
                type=Marker.SPHERE,
                action=Marker.ADD,
            )

            m.header.stamp.sec = 0
            m.header.stamp.nanosec = 0

            (
                m.pose.position.x,
                m.pose.position.y,
                m.pose.position.z,
            ) = (
                pursuit_target[0],
                pursuit_target[1],
                0.08,
            )

            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = 0.15

            m.color.r = 1.0
            m.color.g = 1.0
            m.color.a = 1.0

            ma.markers.append(m)

        self.marker_pub.publish(ma)


def main(args=None):

    rclpy.init(args=args)

    node = RacerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
