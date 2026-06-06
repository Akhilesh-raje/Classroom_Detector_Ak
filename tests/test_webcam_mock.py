# -*- coding: utf-8 -*-
"""
SmartClass AI — Automated Webcam Mock Test
Tests the live webcam orchestrator without requiring physical hardware.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add backend path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

import run_webcam

class TestWebcamOrchestratorMock(unittest.TestCase):
    def setUp(self):
        # Create a mock frame
        self.mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Prepare mock VideoCapture
        self.mock_cap = MagicMock()
        self.mock_cap.isOpened.return_value = True
        
        # Capture.read() returns (ret, frame)
        self.mock_cap.read.return_value = (True, self.mock_frame)
        
        # Mock VideoCapture property getters
        self.properties = {
            3: 1280.0, # CAP_PROP_FRAME_WIDTH
            4: 720.0,  # CAP_PROP_FRAME_HEIGHT
        }
        self.mock_cap.get.side_effect = lambda propId: self.properties.get(propId, 0.0)

    @patch("run_webcam.scan_cameras")
    @patch("cv2.VideoCapture")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    @patch("run_webcam.save_session_report")
    def test_run_orchestrator_loop_exit_on_q(self, mock_save_report, mock_wait_key, mock_imshow, mock_video_capture, mock_scan):
        """Verify the loop starts, captures frames, maps layout, and exits cleanly on Q key."""
        mock_scan.return_value = [0]
        mock_video_capture.return_value = self.mock_cap
        
        # Return ord('q') to exit the runner immediately
        mock_wait_key.return_value = ord("q")
        
        # Execute the main orchestrator loop
        run_webcam.run()
        
        # Assertions
        mock_video_capture.assert_called()
        self.mock_cap.read.assert_called()
        mock_imshow.assert_called()
        mock_save_report.assert_called_once()
        print("  \033[92m[PASS]\033[0m Exit on Q key and report generation verified.")

    @patch("run_webcam.scan_cameras")
    @patch("cv2.VideoCapture")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    @patch("run_webcam.save_session_report")
    def test_remap_grid_on_g_key(self, mock_save_report, mock_wait_key, mock_imshow, mock_video_capture, mock_scan):
        """Verify the grid is remapped dynamically when G is pressed, then exits on Q."""
        mock_scan.return_value = [0]
        mock_video_capture.return_value = self.mock_cap
        
        # Return ord('g') on the first frame, and ord('q') on the second frame to exit
        mock_wait_key.side_effect = [ord("g"), ord("q")]
        
        with patch("run_webcam.ClassroomMapper.detect_layout") as mock_detect_layout:
            mock_detect_layout.return_value = MagicMock(rows=2, cols=3, cells=[])
            
            run_webcam.run()
            
            # ClassroomMapper detect_layout should be called twice:
            # 1. During initialization (reference frame)
            # 2. When 'g' key is handled
            self.assertEqual(mock_detect_layout.call_count, 2)
            print("  \033[92m[PASS]\033[0m Grid re-mapping on G key verified.")

    @patch("run_webcam.scan_cameras")
    @patch("cv2.VideoCapture")
    def test_resolution_escalation(self, mock_video_capture, mock_scan):
        """Verify camera fallback logic escalates and returns optimal resolutions."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        # Simulate initial read failures or low resolution, fallback resolution settings
        res_values = {
            3: [3840, 1920, 1280], # CAP_PROP_FRAME_WIDTH
            4: [2160, 1080, 720],  # CAP_PROP_FRAME_HEIGHT
        }
        
        # Set get return values
        get_calls = []
        def mock_get(propId):
            if propId == 3: # CAP_PROP_FRAME_WIDTH
                return 1280
            elif propId == 4: # CAP_PROP_FRAME_HEIGHT
                return 720
            return 0
            
        mock_cap.get.side_effect = mock_get
        
        cap, w, h = run_webcam.open_camera_optimized(0)
        
        self.assertEqual(w, 1280)
        self.assertEqual(h, 720)
        mock_cap.set.assert_any_call(3, 3840)
        mock_cap.set.assert_any_call(4, 2160)
        print("  \033[92m[PASS]\033[0m Resolution fallback/escalation logic verified.")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SmartClass AI — Automated Webcam Mock Test Results")
    print("="*60)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWebcamOrchestratorMock)
    unittest.TextTestRunner(verbosity=1).run(suite)
