#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import yaml
import os
from std_srvs.srv import Trigger
import tf_transformations
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class RobotPoseSaver(Node):
    """
    Node for saving robot poses to files
    """
    
    def __init__(self):
        super().__init__('robot_pose_saver')
        
        self.declare_parameter('output_folder', '/home/drims/drims_ws/calibrations/extrinsic_calibration/robot_poses')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'tool0')
        
        self.output_folder = self.get_parameter('output_folder').value
        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        
        # Create folder
        os.makedirs(self.output_folder, exist_ok=True)
        
        # TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Service to save pose
        self.srv_save = self.create_service(Trigger, '~/save_pose', self.save_pose_callback)
        
        self.pose_counter = 0
        self.get_logger().info(f"📁 Saving poses in: {self.output_folder}")
        self.get_logger().info(f"🔧 Frames: {self.base_frame} → {self.tool_frame}")
        
    def save_pose_callback(self, request, response):
        """Saves current robot pose"""
        try:
            # Get transform
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                rclpy.time.Time()
            )
            
            # Prepare data
            pose_data = {
                'timestamp': self.get_clock().now().to_msg().sec,
                'frame_id': f"{self.base_frame}_to_{self.tool_frame}",
                'position': [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ],
                'orientation': [
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ]
            }
            
            # Save file
            self.pose_counter += 1
            filename = f"pose_{self.pose_counter:03d}.yaml"
            filepath = os.path.join(self.output_folder, filename)
            
            with open(filepath, 'w') as f:
                yaml.dump(pose_data, f, default_flow_style=False)
            
            # Also save in simple TXT format
            txt_file = os.path.join(self.output_folder, f"pose_{self.pose_counter:03d}.txt")
            with open(txt_file, 'w') as f:
                f.write(f"{pose_data['position'][0]} {pose_data['position'][1]} {pose_data['position'][2]} ")
                f.write(f"{pose_data['orientation'][0]} {pose_data['orientation'][1]} ")
                f.write(f"{pose_data['orientation'][2]} {pose_data['orientation'][3]}")
            
            self.get_logger().info(f"✅ Pose {self.pose_counter} saved: {filename}")
            
            response.success = True
            response.message = f"Pose {self.pose_counter} saved"
            
        except Exception as e:
            self.get_logger().error(f"❌ Error: {e}")
            response.success = False
            response.message = str(e)
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = RobotPoseSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
