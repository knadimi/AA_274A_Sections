#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class ConstantControl(Node):
    def __init__(self):
        super().__init__('constant_control')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.2, self.timer_callback)
        self.subscription = self.create_subscription(Bool, '/kill', self.kill_callback, 10)

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 0.5  # Example linear velocity
        msg.angular.z = 0.1  # Example angular velocity
        self.publisher.publish(msg)
        self.get_logger().info('sending constant control...')

    def kill_callback(self, msg):
        self.get_logger().info(f'Received kill message: {msg.data}')
        if msg.data:
            self.timer.cancel()
            stop_msg = Twist()  # Zero velocity message
            self.publisher.publish(stop_msg)
            self.get_logger().info('Emergency stop!')

def main(args=None):
    rclpy.init(args=args)
    node = ConstantControl()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()