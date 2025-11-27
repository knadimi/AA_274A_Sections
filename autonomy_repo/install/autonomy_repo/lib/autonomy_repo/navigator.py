#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from scipy.interpolate import splrep, splev

from asl_tb3_lib.navigation import BaseNavigator
from asl_tb3_lib.math_utils import wrap_angle

from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl

from P1_astar import AStar, DetOccupancyGrid2D
from P2_trajectory_tracking import TrajectoryTracker, V_PREV_THRES

from navigation import TrajectoryPlan   # your provided file


class Navigator(BaseNavigator):

    def __init__(self):
        super().__init__("navigator")

        # ----------------------------------
        # Gains for heading controller (HW1)
        # ----------------------------------
        self.kw = 1.5     # angular gain

        # ----------------------------------
        # Gains for trajectory tracking (HW2-Q2)
        # ----------------------------------
        self.tracker = TrajectoryTracker(
            kpx=3.0, kpy=3.0, kdx=2.0, kdy=2.0,
            V_max=0.5, om_max=1.0
        )

    # ============================================================
    #  STEP 2 — HEADING controller (same as HW1 heading controller)
    # ============================================================
    def compute_heading_control(self, state: TurtleBotState,
                                      goal: TurtleBotState) -> TurtleBotControl:

        # heading error
        e_th = wrap_angle(goal.theta - state.theta)

        om = self.kw * e_th
        om = float(np.clip(om, -1.0, 1.0))

        return TurtleBotControl(v=0.0, omega=om)

    # ==========================================================================
    #  STEP 3 — TRAJECTORY TRACKING CONTROLLER (using HW2 differential flatness)
    # ==========================================================================
    def compute_trajectory_tracking_control(self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float,
    ) -> TurtleBotControl:

        # Retrieve desired flat outputs via spline derivative sampling
        x_d  = float(splev(t, plan.path_x_spline, der=0))
        xd_d = float(splev(t, plan.path_x_spline, der=1))
        xdd_d = float(splev(t, plan.path_x_spline, der=2))

        y_d  = float(splev(t, plan.path_y_spline, der=0))
        yd_d = float(splev(t, plan.path_y_spline, der=1))
        ydd_d = float(splev(t, plan.path_y_spline, der=2))

        # Run the HW2 tracker
        V, om = self.tracker.compute_control(
            x=state.x,
            y=state.y,
            th=state.theta,
            t=t
        )

        return TurtleBotControl(v=float(V), omega=float(om))

    # ==========================================================================
    #  STEP 4 — TRAJECTORY PLANNING (A* + cubic spline interpolation)
    # ==========================================================================
    def compute_trajectory_plan(
        self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy,
        resolution: float,
        horizon: float,
    ):

        # -----------------------------------------
        # Convert stochastic occupancy → deterministic grid
        # -----------------------------------------
        ox = occupancy.origin_xy[0]
        oy = occupancy.origin_xy[1]
        width = occupancy.size_xy[0]
        height = occupancy.size_xy[1]

        # convert prob grid → 0/1 free/occupied
        obstacles = []
        occmap = np.array(occupancy.probs).reshape(height, width)
        for i in range(height):
            for j in range(width):
                if occmap[i, j] > 50:  # occupied probability threshold
                    x0 = ox + j * occupancy.resolution
                    y0 = oy + i * occupancy.resolution
                    obstacles.append(((x0, y0),
                                      (x0 + occupancy.resolution,
                                       y0 + occupancy.resolution)))

        det_grid = DetOccupancyGrid2D(width, height, obstacles)

        # -----------------------------------------
        # Run A*
        # -----------------------------------------
        planner = AStar(
            statespace_lo=[ox, oy],
            statespace_hi=[ox + width, oy + height],
            x_init=(state.x, state.y),
            x_goal=(goal.x, goal.y),
            occupancy=det_grid,
            resolution=resolution
        )

        success = planner.solve()
        if not success:
            return None

        path = np.array(planner.path)

        # -----------------------------------------
        # Fit cubic splines (x(t), y(t))
        # -----------------------------------------
        s = np.linspace(0, 1, len(path))
        tck_x = splrep(s, path[:, 0], k=3, s=0)
        tck_y = splrep(s, path[:, 1], k=3, s=0)

        # duration proportional to path length
        duration = float(len(path) * 0.2)

        return TrajectoryPlan(
            path=path,
            path_x_spline=tck_x,
            path_y_spline=tck_y,
            duration=duration
        )


def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
