#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import splev
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from scipy.interpolate import splrep
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from utils import plot_line_segments

import rclpy
from rclpy.node import Node
from asl_tb3_lib.navigation import BaseNavigator
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        if x == self.x_init or x == self.x_goal:
            return True
        if not (self.statespace_lo[0] <= x[0] <= self.statespace_hi[0] and self.statespace_lo[1] <= x[1] <= self.statespace_hi[1]):
            return False
        return self.occupancy.is_free(x)
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        # 8-connected grid: up, down, left, right, and diagonals
        directions = [
            (self.resolution, 0),
            (-self.resolution, 0),
            (0, self.resolution),
            (0, -self.resolution),
            (self.resolution, self.resolution),
            (self.resolution, -self.resolution),
            (-self.resolution, self.resolution),
            (-self.resolution, -self.resolution)
        ]
        for dx, dy in directions:
            neighbor = self.snap_to_grid((x[0] + dx, x[1] + dy))
            if neighbor != x and self.is_free(neighbor):
                neighbors.append(neighbor)
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while self.open_set:
            current = self.find_best_est_cost_through()
            if current == self.x_goal:
                self.path = self.reconstruct_path()
                return True

            self.open_set.remove(current)
            self.closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_set:
                    continue

                tentative_g_score = self.cost_to_arrive[current] + self.distance(current, neighbor)

                if neighbor not in self.open_set:
                    self.open_set.add(neighbor)
                elif tentative_g_score >= self.cost_to_arrive.get(neighbor, float('inf')):
                    continue

                self.came_from[neighbor] = current
                self.cost_to_arrive[neighbor] = tentative_g_score
                self.est_cost_through[neighbor] = tentative_g_score + self.distance(neighbor, self.x_goal)

        return False
        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - self.width * .01 and \
               x[0] <= obs[1][0] + self.width * .01 and \
               x[1] >= obs[0][1] - self.height * .01 and \
               x[1] <= obs[1][1] + self.height * .01:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))
        

class Navigator(BaseNavigator):
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy):  # Add the missing arguments
        super().__init__('navigator') # Initialize the node with name 'navigator'
        self.statespace_lo = statespace_lo
        self.statespace_hi = statespace_hi
        self.x_init = x_init
        self.x_goal = x_goal
        self.occupancy = occupancy

        self.k_p = 1.0  # Example proportional gain
        self.k_i = 0.0  # Example integral gain
        self.k_d = 0.0  # Example derivative gain
        self.previous_error = 0.0
        self.integral = 0.0

        # Trajectory Tracking Controller Parameters
        self.k_x = 1.0  # Example gains)
        self.k_y = 1.0
        self.k_theta = 1.0
        self.V_PREV_THRESH = 0.01  # Threshold for resetting integrator

        # ODE Integration State Variables (initialize them)
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_time = None # Previous timestamp

        self.astar = AStar(statespace_lo, statespace_hi, x_init, x_goal, occupancy)  # Pass the arguments to AStar

        self.get_logger().info("Navigator node started")
        
    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        self.get_logger().info("compute_heading_control called")
        err = wrap_angle(goal.theta - state.theta)
        control = TurtleBotControl()
        control.omega = self.kp * err
        return control

    def compute_trajectory_tracking_control(self, state, trajectory_plan, time):
        """
        Computes trajectory tracking control based on current state, trajectory plan, and current time.
        """
        x, y, theta = state

        # Extract spline parameters and other relevant data from trajectory_plan
        tck_x = trajectory_plan.spline_x  # Spline parameters for x(t)
        tck_y = trajectory_plan.spline_y  # Spline parameters for y(t)
        start_time = trajectory_plan.start_time
        end_time = trajectory_plan.end_time

        # Calculate the time within the trajectory
        traj_time = time - start_time

        # Sample desired states from splines using scipy.interpolate.splev
        x_d = splev(traj_time, tck_x, der=0)
        xd_d = splev(traj_time, tck_x, der=1)
        xdd_d = splev(traj_time, tck_x, der=2)
        y_d = splev(traj_time, tck_y, der=0)
        yd_d = splev(traj_time, tck_y, der=1)
        ydd_d = splev(traj_time, tck_y, der=2)

        # Implement your differential flatness controller here
        # (Adapt from your HW2, Q2 implementation)

        # Calculate errors
        x_error = x_d - x
        y_error = y_d - y
        theta_error = wrap_angle(np.arctan2(yd_d, xd_d) - theta)

        # Integrate the errors (for I term)
        if self.prev_time is not None:
            dt = time - self.prev_time
            self.integral_x += x_error * dt
            self.integral_y += y_error * dt

            # Anti-windup (clamp the integral terms)
            self.integral_x = max(min(self.integral_x, 1.0), -1.0)  # Example limits
            self.integral_y = max(min(self.integral_y, 1.0), -1.0)  # Example limits

        # Reset integrators if velocity is too low (to avoid windup at standstill)
        v = np.sqrt(xd_d**2 + yd_d**2)
        if v < self.V_PREV_THRESH:
            self.integral_x = 0.0
            self.integral_y = 0.0

        # Control law (example - replace with your actual controller)
        v = np.sqrt(xd_d**2 + yd_d**2) + self.k_x * x_error + self.integral_x # Linear velocity
        omega = self.k_theta * theta_error + self.k_y * y_error + self.integral_y # Angular velocity

        # Create and populate the TurtleBotControl message
        control_msg = TurtleBotControl()
        control_msg.linear_velocity = float(v)
        control_msg.angular_velocity = float(omega)

        self.prev_time = time # Update previous time

        return control_msg
    
    def compute_trajectory_plan(self, start_pose, goal_pose):
        """
        Computes a trajectory plan given a start and goal pose.
        """
        # 1. Initialize A* problem
        self.astar.set_params(start_pose, goal_pose, occupancy, resolution, horizon)

        # 2. Solve A* problem
        path, cost = self.astar.a_star()

        # 3. Check if solution exists
        if path is None or len(path) < 4:
            self.get_logger().warn("A* failed to find a valid path.")
            return None

        # 4. Reset Tracking Controller History
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_time = None

        # 5. Compute planned time stamps (constant velocity heuristic)
        distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        total_distance = np.sum(distances)
        # Assuming a constant velocity of 0.2 m/s
        times = np.cumsum(distances / 0.2)
        times = np.insert(times, 0, 0)  # Add a zero at the beginning

        # 6. Compute smooth plan (spline parameters)
        smoothed_path = self.smooth_path(path)  # Replace with your smoothing function

        # Convert path to numpy arrays
        x = np.array([p[0] for p in smoothed_path])
        y = np.array([p[1] for p in smoothed_path])

        # Spline Interpolation
        try:
            tck_x = splrep(np.linspace(0, 1, len(x)), x, s=0.1)  # Spline for x(t)
            tck_y = splrep(np.linspace(0, 1, len(y)), y, s=0.1)  # Spline for y(t)
        except Exception as e:
            self.get_logger().warn(f"Spline fitting failed: {e}")
            return None

        # 7. Construct TrajectoryPlan
        trajectory_plan = TrajectoryPlan()
        trajectory_plan.spline_x = list(tck_x)  # Convert to lists for ROS message
        trajectory_plan.spline_y = list(tck_y)  # Convert to lists for ROS message
        trajectory_plan.start_time = self.get_clock().now().to_msg().nanosec  # Current time as start time
        trajectory_plan.end_time = self.get_clock().now().to_msg().nanosec + float(times[-1])  # End time based on path duration

        return trajectory_plan
    
def main(args=None):
    rclpy.init(args=args)
    # Use the values you provided
    width = 10
    height = 10
    obstacles = [((6,7),(8,8)),((2,2),(4,3)),((2,5),(4,7)),((6,3),(8,5))]
    occupancy = DetOccupancyGrid2D(width, height, obstacles)

    x_init = (1.0, 9.0)  # Important: Make sure these are floats!
    x_goal = (9.0, 1.0)  # Important: Make sure these are floats!
    statespace_lo = (0.0, 0.0)  # Assuming the origin of your grid is (0, 0)
    statespace_hi = (float(width), float(height))  # Upper bounds of your grid

    navigator = Navigator(statespace_lo, statespace_hi, x_init, x_goal, occupancy) # Pass the arguments here

    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()