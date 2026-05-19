import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Int32, Float32
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import (
    Parameter,
    ParameterValue,
    ParameterType,
)

import time
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor


class RaceTeamDashboard(Node):

    def __init__(self):

        super().__init__("race_team_dashboard")

        # ========================================================= #
        # RESET PUBLISHER
        # ========================================================= #

        self.reset_pub = self.create_publisher(
            Bool,
            "/autodrive/reset_command",
            10,
        )

        # ========================================================= #
        # PARAM CLIENT
        # ========================================================= #

        self.param_client = self.create_client(
            SetParameters,
            "/robust_rollout_racer/set_parameters",
        )

        # ========================================================= #
        # METRIC SUBSCRIPTIONS
        # ========================================================= #

        self.create_subscription(
            Int32,
            "/autodrive/roboracer_1/collision_count",
            self.collision_callback,
            10,
        )

        self.create_subscription(
            Int32,
            "/autodrive/roboracer_1/lap_count",
            self.lap_callback,
            10,
        )

        self.create_subscription(
            Float32,
            "/autodrive/roboracer_1/lap_time",
            self.time_callback,
            10,
        )

        # ========================================================= #
        # STATE
        # ========================================================= #

        self.current_collisions = 0
        self.current_laps = 0
        self.current_lap_time = 0.0

        self.initial_laps = None
        self.initial_collisions = None

        self.results = []

        # ========================================================= #
        # PARAM GRID
        # ========================================================= #

        self.param_grid = {
            # SPEED
            "max_throttle": np.arange(0.15, 0.46, 0.05).tolist(),
            "min_throttle": [0.03, 0.05, 0.07],
            "throttle_ramp_rate": [0.005, 0.01, 0.015, 0.02, 0.03],
            # SAFETY
            "safety_margin": np.arange(0.08, 0.23, 0.02).tolist(),
            # ROLLOUT
            "rollout_distance": np.arange(1.5, 3.6, 0.5).tolist(),
            "pursuit_lookahead": np.arange(1.2, 2.6, 0.3).tolist(),
            "pursuit_blend": np.arange(0.2, 0.81, 0.1).tolist(),
            # SCORING
            "progress_weight": np.arange(1.0, 6.1, 1.0).tolist(),
            "clearance_weight": np.arange(1.0, 5.1, 1.0).tolist(),
            "smoothness_weight": np.arange(-0.5, -2.6, -0.5).tolist(),
            "steering_penalty_weight": np.arange(-0.1, -1.1, -0.2).tolist(),
            "gap_alignment_weight": np.arange(0.5, 2.6, 0.5).tolist(),
            "turn_commitment_weight": np.arange(1.0, 7.1, 1.0).tolist(),
            "max_steering_delta": [0.04, 0.06, 0.08, 0.10, 0.12],
            "temporal_blend": [0.05, 0.10, 0.15, 0.20, 0.30],
            "disparity_threshold": [0.25, 0.40, 0.50, 0.70, 1.0],
            "max_jump_distance": [0.8, 1.2, 1.5, 2.0],
            "history_penalty_radius": [0.8, 1.0, 1.2, 1.5, 2.0],
            "history_penalty_weight": [1.0, 2.0, 4.0, 6.0, 8.0],
            "gap_threshold_scale": [0.25, 0.35, 0.45, 0.60],
            "max_gap_threshold": [1.5, 2.0, 2.5, 3.0],
            "turn_prediction_gain": [0.10, 0.20, 0.25, 0.35, 0.50],
            "forward_critical_distance": [0.30, 0.40, 0.50, 0.70],
            "forward_warning_distance": [0.60, 0.80, 1.0, 1.3],
            "forward_caution_distance": [1.0, 1.2, 1.5, 2.0],
        }

        # ========================================================= #
        # WAIT FOR SERVICE
        # ========================================================= #

        self.get_logger().info("RACE DASHBOARD ONLINE")

        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for racer node...")

    # ============================================================= #
    # CALLBACKS
    # ============================================================= #

    def collision_callback(self, msg):
        self.current_collisions = msg.data

    def lap_callback(self, msg):
        self.current_laps = msg.data

    def time_callback(self, msg):
        self.current_lap_time = msg.data

    # ============================================================= #
    # RESET SIM
    # ============================================================= #

    def reset_simulation(self):

        msg = Bool(data=True)

        for _ in range(3):
            self.reset_pub.publish(msg)
            time.sleep(0.1)

        time.sleep(2.5)

        msg.data = False
        self.reset_pub.publish(msg)

        time.sleep(1.5)

        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)

        self.initial_laps = self.current_laps
        self.initial_collisions = self.current_collisions

    # ============================================================= #
    # PARAM SETTER
    # ============================================================= #

    def set_params(self, param_dict):

        req = SetParameters.Request()

        for k, v in param_dict.items():

            pv = ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE,
                double_value=float(v),
            )

            req.parameters.append(
                Parameter(
                    name=k,
                    value=pv,
                )
            )

        self.param_client.call_async(req)

        time.sleep(0.5)

    # ============================================================= #
    # MAIN TUNER
    # ============================================================= #

    def run_exhaustive_optimization(self, iterations=1000):

        self.get_logger().info(f"STARTING {iterations} TRIALS")

        for i in range(iterations):

            # ===================================================== #
            # RANDOM PARAM SAMPLE
            # ===================================================== #

            p = {k: np.random.choice(v) for k, v in self.param_grid.items()}

            self.get_logger().info(
                f"TRIAL {i+1}/{iterations} | "
                f"Throttle={p['max_throttle']:.2f} | "
                f"Safety={p['safety_margin']:.2f} | "
                f"Rollout={p['rollout_distance']:.2f}"
            )

            # ===================================================== #
            # APPLY PARAMS
            # ===================================================== #

            self.set_params(p)

            # ===================================================== #
            # RESET
            # ===================================================== #

            self.reset_simulation()

            # ===================================================== #
            # RUN CONFIG
            # ===================================================== #

            start_t = time.time()

            max_runtime = 135.0
            target_laps = 15

            lap_timestamps = []

            prev_lap_count = self.current_laps

            while rclpy.ok():

                rclpy.spin_once(self, timeout_sec=0.1)

                elapsed = time.time() - start_t

                delta_collisions = self.current_collisions - self.initial_collisions

                delta_laps = self.current_laps - self.initial_laps

                # ================================================= #
                # LAP TIMESTAMPS
                # ================================================= #

                if self.current_laps != prev_lap_count:

                    lap_timestamps.append(elapsed)

                    prev_lap_count = self.current_laps

                # ================================================= #
                # PERIODIC STATUS
                # ================================================= #

                if int(elapsed) % 15 == 0:

                    self.get_logger().debug(
                        f"Runtime={elapsed:.1f}s | "
                        f"Laps={delta_laps} | "
                        f"Collisions={delta_collisions}"
                    )

                # ================================================= #
                # END CONDITIONS
                # ================================================= #

                if delta_laps >= target_laps:
                    break

                if elapsed >= max_runtime:
                    break

                if delta_collisions >= 50:
                    break

            # ===================================================== #
            # FINAL METRICS
            # ===================================================== #

            total_runtime = time.time() - start_t

            delta_collisions = self.current_collisions - self.initial_collisions

            delta_laps = self.current_laps - self.initial_laps

            success = int(delta_laps >= target_laps)

            avg_lap_time = total_runtime / delta_laps if delta_laps > 0 else 999.0

            collisions_per_lap = delta_collisions / max(delta_laps, 1)

            # ===================================================== #
            # LAP CONSISTENCY
            # ===================================================== #

            lap_deltas = []

            if len(lap_timestamps) >= 2:

                for j in range(1, len(lap_timestamps)):

                    lap_deltas.append(lap_timestamps[j] - lap_timestamps[j - 1])

            lap_consistency = np.std(lap_deltas) if len(lap_deltas) > 1 else 999.0

            # ===================================================== #
            # COMPOSITE SCORE
            # ===================================================== #

            endurance_score = (
                (delta_laps * 15.0)
                - (delta_collisions * 20.0)
                - (avg_lap_time * 2.5)
                - (lap_consistency * 5.0)
            )

            # ===================================================== #
            # STORE
            # ===================================================== #

            res = p.copy()

            res.update(
                {
                    "laps_completed": delta_laps,
                    "runtime": total_runtime,
                    "avg_lap_time": avg_lap_time,
                    "lap_consistency": lap_consistency,
                    "collisions": delta_collisions,
                    "collisions_per_lap": collisions_per_lap,
                    "success": success,
                    "endurance_score": endurance_score,
                }
            )

            self.results.append(res)

            # ===================================================== #
            # SAVE EVERY RUN
            # ===================================================== #

            pd.DataFrame(self.results).to_csv(
                "race_telemetry_partial.csv",
                index=False,
            )

            # ===================================================== #
            # PRINT RESULT
            # ===================================================== #

            self.get_logger().info(
                f"RESULT | "
                f"Laps={delta_laps} | "
                f"Collisions={delta_collisions} | "
                f"AvgLap={avg_lap_time:.2f} | "
                f"Consistency={lap_consistency:.2f} | "
                f"Score={endurance_score:.2f}"
            )

        # ========================================================= #
        # FINAL REPORT
        # ========================================================= #

        self.generate_final_report()

    # ============================================================= #
    # FINAL REPORT
    # ============================================================= #

    def generate_final_report(self):

        if len(self.results) == 0:
            print("No results.")
            return

        df = pd.DataFrame(self.results)

        df.to_csv(
            "race_telemetry_exhaustive.csv",
            index=False,
        )

        # ========================================================= #
        # RANDOM FOREST FEATURE IMPORTANCE
        # ========================================================= #

        numeric_df = df.select_dtypes(include=[np.number])

        feature_cols = [
            c
            for c in numeric_df.columns
            if c
            not in [
                "endurance_score",
                "success",
            ]
        ]

        X = numeric_df[feature_cols]
        y = numeric_df["endurance_score"]

        model = RandomForestRegressor(
            n_estimators=150,
            random_state=42,
        )

        model.fit(X, y)

        importance_df = pd.DataFrame(
            {
                "parameter": feature_cols,
                "importance": model.feature_importances_,
            }
        ).sort_values(
            by="importance",
            ascending=False,
        )

        importance_df.to_csv(
            "parameter_importance.csv",
            index=False,
        )

        # ========================================================= #
        # PLOTS
        # ========================================================= #

        sns.set_theme(
            style="whitegrid",
            palette="muted",
        )

        fig = plt.figure(figsize=(30, 22))

        # ========================================================= #
        # 1. CORRELATION
        # ========================================================= #

        plt.subplot(3, 2, 1)

        corr = numeric_df.corr()

        sns.heatmap(
            corr[
                [
                    "success",
                    "avg_lap_time",
                    "laps_completed",
                    "endurance_score",
                ]
            ],
            annot=True,
            cmap="RdYlGn",
        )

        plt.title("Correlation Matrix")

        # ========================================================= #
        # 2. SPEED VS COLLISIONS
        # ========================================================= #

        plt.subplot(3, 2, 2)

        sns.scatterplot(
            data=df,
            x="avg_lap_time",
            y="collisions_per_lap",
            hue="laps_completed",
            size="endurance_score",
            sizes=(40, 450),
            alpha=0.75,
        )

        plt.title("Speed vs Reliability")

        # ========================================================= #
        # 3. FEATURE IMPORTANCE
        # ========================================================= #

        plt.subplot(3, 2, 3)

        importance_df.head(10).plot(
            kind="barh",
            x="parameter",
            y="importance",
            legend=False,
            ax=plt.gca(),
        )

        plt.title("Most Important Parameters")

        # ========================================================= #
        # 4. LAP CONSISTENCY
        # ========================================================= #

        plt.subplot(3, 2, 4)

        sns.scatterplot(
            data=df,
            x="lap_consistency",
            y="avg_lap_time",
            hue="success",
            size="collisions_per_lap",
            sizes=(40, 400),
        )

        plt.title("Consistency vs Speed")

        # ========================================================= #
        # 5. ENDURANCE SCORE DISTRIBUTION
        # ========================================================= #

        plt.subplot(3, 2, 5)

        sns.histplot(
            data=df,
            x="endurance_score",
            bins=25,
            kde=True,
        )

        plt.title("Endurance Score Distribution")

        # ========================================================= #
        # 6. PARAM SWARM
        # ========================================================= #

        plt.subplot(3, 2, 6)

        sns.stripplot(
            data=df,
            x="max_throttle",
            y="endurance_score",
            hue="safety_margin",
            dodge=True,
        )

        plt.title("Throttle vs Endurance Score")

        plt.tight_layout()

        plt.savefig("race_team_exhaustive_analysis.png")

        # ========================================================= #
        # GOLDEN CONFIG
        # ========================================================= #

        golden = df.sort_values(
            by="endurance_score",
            ascending=False,
        ).iloc[0]

        print("\n" + "=" * 80)
        print("BEST CONFIGURATION FOUND")
        print("=" * 80)

        for k, v in golden.items():

            if isinstance(v, float):
                print(f"{k:30}: {v:.4f}")
            else:
                print(f"{k:30}: {v}")

        print("=" * 80)

        print(
            "\nFILES SAVED:"
            "\n - race_telemetry_partial.csv"
            "\n - race_telemetry_exhaustive.csv"
            "\n - race_team_exhaustive_analysis.png"
            "\n - parameter_importance.csv"
        )


# ============================================================= #
# MAIN
# ============================================================= #


def main():

    rclpy.init()

    dash = RaceTeamDashboard()

    try:

        dash.run_exhaustive_optimization(iterations=1000)

    except KeyboardInterrupt:

        print("\nKeyboardInterrupt detected.")
        print("Generating partial analytics...")

        try:

            dash.generate_final_report()

        except Exception as e:

            print(f"Failed generating report: {e}")

    finally:

        try:

            if len(dash.results) > 0:

                pd.DataFrame(dash.results).to_csv(
                    "race_telemetry_emergency_save.csv",
                    index=False,
                )

                print("Emergency save completed.")

        except Exception as e:

            print(f"Emergency save failed: {e}")
        if dash:
            dash.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
