#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.duration import Duration
from rclpy.node import Node

from scipy.interpolate import splrep, splev

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool

from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from asl_tb3_lib.navigation import BaseNavigator
from asl_tb3_lib.math_utils import wrap_angle, distance_linear, distance_angular
from asl_tb3_lib.grids import StochOccupancyGrid2D


# ============================================================
# ================   EMBEDDED   A* PLANNER   ==================
# ============================================================

class DetOccupancyGrid2D:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        for obs in self.obstacles:
            if obs[0][0] <= x[0] <= obs[1][0] and \
               obs[0][1] <= x[1] <= obs[1][1]:
                return False
        return True


class AStar:

    def __init__(self, lo, hi, x_init, x_goal, occupancy, resolution=0.1):
        self.lo = lo
        self.hi = hi
        self.occ = occupancy
        self.res = resolution

        self.x_init = self.snap(x_init)
        self.x_goal = self.snap(x_goal)

        self.open = {self.x_init}
        self.closed = set()

        self.came_from = {}
        self.g = {self.x_init: 0.0}
        self.f = {self.x_init: self.dist(self.x_init, self.x_goal)}

    def snap(self, x):
        return (self.res * round(x[0] / self.res),
                self.res * round(x[1] / self.res))

    def dist(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def neighbors(self, x):
        dirs = [
            (self.res, 0), (-self.res, 0),
            (0, self.res), (0, -self.res),
            (self.res, self.res), (self.res, -self.res),
            (-self.res, self.res), (-self.res, -self.res)
        ]
        N = []
        for dx, dy in dirs:
            nxt = (x[0] + dx, x[1] + dy)
            nxt = self.snap(nxt)
            if self.lo[0] <= nxt[0] <= self.hi[0] and \
               self.lo[1] <= nxt[1] <= self.hi[1] and \
               self.occ.is_free(nxt):
                N.append(nxt)
        return N

    def best(self):
        return min(self.open, key=lambda x: self.f[x])

    def reconstruct(self):
        path = [self.x_goal]
        cur = self.x_goal
        while cur != self.x_init:
            cur = self.came_from[cur]
            path.append(cur)
        return np.array(list(reversed(path)))

    def solve(self):
        while self.open:
            current = self.best()
            if current == self.x_goal:
                return self.reconstruct()

            self.open.remove(current)
            self.closed.add(current)

            for nxt in self.neighbors(current):
                if nxt in self.closed:
                    continue

                tentative = self.g[current] + self.dist(current, nxt)

                if nxt not in self.open or tentative < self.g.get(nxt, 1e9):
                    self.came_from[nxt] = current
                    self.g[nxt] = tentative
                    self.f[nxt] = tentative + self.dist(nxt, self.x_goal)
                    self.open.add(nxt)

        return None


# ============================================================
# ============   EMBEDDED DIFFERENTIAL FLATNESS TRACKER  ======
# ============================================================

V_PREV_THRES = 1e-4

class TrajectoryTracker:

    def __init__(self, kpx, kpy, kdx, kdy, vmax=0.5, ommax=1.0):
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.vmax = vmax
        self.ommax = ommax

        self.V_prev = V_PREV_THRES
        self.t_prev = 0.0

    def compute(self, x, y, th, t, xd_d, yd_d, xdd_d, ydd_d, x_d, y_d):

        dt = t - self.t_prev

        # estimated current velocity
        xd = self.V_prev * np.cos(th)
        yd = self.V_prev * np.sin(th)

        # virtual control
        u = np.array([
            xdd_d + self.kpx*(x_d - x) + self.kdx*(xd_d - xd),
            ydd_d + self.kpy*(y_d - y) + self.kdy*(yd_d - yd)
        ])

        J = np.array([
            [np.cos(th), -self.V_prev*np.sin(th)],
            [np.sin(th),  self.V_prev*np.cos(th)]
        ])

        a, om = np.linalg.solve(J, u)
        V = self.V_prev + a * dt

        # limits
        V = float(np.clip(V, -self.vmax, self.vmax))
        om = float(np.clip(om, -self.ommax, self.ommax))

        self.V_prev = V
        self.t_prev = t

        return V, om


# ============================================================
# ==================   TRAJECTORY PLAN   =====================
# ============================================================

class TrajectoryPlan:
    def __init__(self, path, tx, ty, duration):
        self.path = path
        self.tx = tx
        self.ty = ty
        self.duration = duration
        
    def desired_state(self, t: float) -> TurtleBotState:
        """
        Returns desired (x, y, theta) at time t using spline derivatives.
        Matches the BaseNavigator API.
        """

        # position
        x = float(splev(t, self.tx, der=0))
        y = float(splev(t, self.ty, der=0))

        # first derivative → orientation
        xd = float(splev(t, self.tx, der=1))
        yd = float(splev(t, self.ty, der=1))

        theta = float(np.arctan2(yd, xd))

        return TurtleBotState(x=x, y=y, theta=theta)


    def smoothed_path(self, dt: float = 0.1):
        """Sample smoothed spline path for RVIZ visualization."""
        ts = np.arange(0., self.duration, dt)
        xs = splev(ts, self.tx, der=0)
        ys = splev(ts, self.ty, der=0)

        return np.vstack((xs, ys)).T


# ============================================================
# ===================   NAVIGATOR NODE   =====================
# ============================================================

class Navigator(BaseNavigator):

    def __init__(self):
        super().__init__("navigator")

        # heading control gain
        self.kw = 2.0

        # embed the tracker
        self.tracker = TrajectoryTracker(
            kpx=3.0, kpy=3.0, kdx=2.0, kdy=2.0,
            vmax=0.5, ommax=1.0
        )

    # ----------------- HEADING CONTROL -----------------------
    def compute_heading_control(self, state, goal):
        e = wrap_angle(goal.theta - state.theta)
        om = float(np.clip(self.kw * e, -1.0, 1.0))
        return TurtleBotControl(v=0.0, omega=om)

    # ---------------- TRAJECTORY TRACKING --------------------
    def compute_trajectory_tracking_control(self, state, plan, t):

        x_d   = float(splev(t, plan.tx, der=0))
        xd_d  = float(splev(t, plan.tx, der=1))
        xdd_d = float(splev(t, plan.tx, der=2))

        y_d   = float(splev(t, plan.ty, der=0))
        yd_d  = float(splev(t, plan.ty, der=1))
        ydd_d = float(splev(t, plan.ty, der=2))

        V, om = self.tracker.compute(
            x=state.x, y=state.y, th=state.theta, t=t,
            xd_d=xd_d, yd_d=yd_d,
            xdd_d=xdd_d, ydd_d=ydd_d,
            x_d=x_d, y_d=y_d
        )

        return TurtleBotControl(v=V, omega=om)

    # ---------------- TRAJECTORY PLANNING ---------------------
    def compute_trajectory_plan(
        self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
      ):

        """
        Compute full trajectory plan:
        1. Convert stochastic occupancy map → deterministic grid
        2. Run A*
        3. Fit cubic splines
        4. Return TrajectoryPlan
        """

        # -------------------------------------------------------
        # Convert occupancy → deterministic grid of obstacles
        # -------------------------------------------------------
        ox, oy = occupancy.origin_xy
        H, W = occupancy.size_xy
        res = occupancy.resolution

        obstacles = []
        grid = np.array(occupancy.probs).reshape(H, W)

        for i in range(H):
            for j in range(W):
                if grid[i, j] > 50:
                    x0 = ox + j * res
                    y0 = oy + i * res
                    obstacles.append(((x0, y0),
                                    (x0 + res, y0 + res)))

        det = DetOccupancyGrid2D(W*res, H*res, obstacles)

        # run A*
        astar = AStar(
            lo=[ox, oy],
            hi=[ox + W*res, oy + H*res],
            x_init=(state.x, state.y),
            x_goal=(goal.x, goal.y),
            occupancy=det,
            resolution=resolution
        )

        path = astar.solve()
        if path is None:
            return None

        s = np.linspace(0, 1, len(path))
        tx = splrep(s, path[:, 0], k=3, s=0)
        ty = splrep(s, path[:, 1], k=3, s=0)

        duration = float(len(path) * 0.2)

        return TrajectoryPlan(path=path, tx=tx, ty=ty, duration=duration)


# ============================================================
# =======================   MAIN   ============================
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
