#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import BufferClient
from tf2_ros.transform_listener import TransformListener
from asl_tb3_msgs.msg import TurtleBotState

class StatePublisher(Node):

    def __init__(self):
        super().__init__('state_publisher')

        self.tf_buffer = BufferClient(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state_publisher = self.create_publisher(TurtleBotState, '/turtlebot_state', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        self.world_frame = 'world'  # Or 'map', depending on your setup
        self.base_frame = 'base_link'

    def timer_callback(self):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.base_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=0.1)) # Increased timeout
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform {self.base_frame} to {self.world_frame}: {ex}')
            return  # Exit if transform fails

        state = TurtleBotState()
        state.x = transform.transform.translation.x
        state.y = transform.transform.translation.y
        # Extract yaw from quaternion
        q = transform.transform.rotation
        state.theta = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        self.state_publisher.publish(state)


def main(args=None):
    rclpy.init(args=args)
    node = StatePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()