#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Bool


class PerceptionController(BaseHeadingController):

    def __init__(self):
        super().__init__('perception_controller')

        # --------------------
        # Task 2.2 — active parameter
        # --------------------
        self.declare_parameter('active', True)

        # --------------------
        # Task 4.1 — detector listener variable
        # --------------------
        self.image_detected = False

        # Subscribe to /detector_bool topic
        self.create_subscription(
            Bool,
            '/detector_bool',
            self.detector_callback,
            10
        )

    # PROPERTY for the active parameter
    @property
    def active(self) -> bool:
        return self.get_parameter('active').value

    # Callback that listens for detection result
    def detector_callback(self, msg: Bool):
        if msg.data:
            self.image_detected = True
            self.get_logger().info("Stop sign detected! Stopping robot.")
       

    # --------------------
    # Task 2.3 + Task 4.2
    # --------------------
    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        control = TurtleBotControl()

        # If controller inactive → stop
        if not self.active:
            control.omega = 0.0
            return control

        # If stop sign detected → stop spinning
        if self.image_detected:
            control.omega = 0.0
        else:
            # Constant angular velocity
            control.omega = 0.2

        return control


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
