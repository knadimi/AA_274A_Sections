#!/usr/bin/env python3
import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        super().__init__('heading_controller')
        self.declare_parameter('kp', 2.0)
        #self.declare_parameter('velocity', 1.5)
        
    @property
    def kp(self) -> float:
        return self.get_parameter('kp').value
    
    #@property
    #def velocity(self) -> float:
    #    return self.get_parameter('velocity').value

    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        err = wrap_angle(goal.theta - state.theta)
        control = TurtleBotControl()
        control.omega = self.kp * err
    
        # Calculate distance to goal
        #distance = np.sqrt((goal.x - state.x)**2 + (goal.y - state.y)**2)

        # Set linear velocity
        #control.v = self.velocity # Set linear velocity

        return control

def main(args=None):
    rclpy.init(args=args)
    node = HeadingController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
