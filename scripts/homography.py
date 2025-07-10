#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class DocHomography:
    def __init__(self):
        rospy.init_node('doc_homography_node', anonymous=True)
        self.bridge = CvBridge()

        self.image_path = rospy.get_param('~image_path', os.path.join(os.path.dirname(__file__), '../resources/test_document.jpg'))
        self.output_path = rospy.get_param('~output_path', 'doc-homography.png')
        self.output_height = rospy.get_param('~output_height', 1000)
        self.aspect_ratio = 297 / 210.0  

        if not os.path.exists(self.image_path):
            rospy.logerr(f"Image not found: {self.image_path}")
            return
            
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            rospy.logerr(f"Failed to load image: {self.image_path}")
            return
            
        rospy.loginfo("Select 4 corners: LT, RT, RB, LB (press ESC when done)")
        self.corners = self.select_corners()
        
        if len(self.corners) == 4:
            self.process_image()
        else:
            rospy.logwarn("Exactly 4 corners required. Exiting.")

    def select_corners(self):
        image = self.original_image.copy()
        corners = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append((x, y))
                cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(image, str(len(corners)), (x+15, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Select Corners", image)
                rospy.loginfo(f"Selected point {len(corners)}: ({x}, {y})")
        
        cv2.namedWindow("Select Corners", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Corners", mouse_callback)
        
        cv2.imshow("Select Corners", image)
        while len(corners) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        
        cv2.destroyAllWindows()
        return np.array(corners, dtype=np.float32)

    def process_image(self):
        height = self.output_height
        width = int(height / self.aspect_ratio)
        
        dst_points = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        H = cv2.getPerspectiveTransform(self.corners, dst_points)
        
        corrected = cv2.warpPerspective(
            self.original_image, H, (width, height)
        )
        
        orig_resized = cv2.resize(self.original_image, (width, height))
        separator = np.ones((height, 32, 3), dtype=np.uint8) * 255
        comparison = np.hstack([orig_resized, separator, corrected])
        
        cv2.imwrite(self.output_path, comparison)
        
        cv2.imshow("Comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        DocHomography()
    except rospy.ROSInterruptException:
        pass