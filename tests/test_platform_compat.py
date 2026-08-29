from __future__ import annotations

import unittest

import cv2

from motion_cut.main import _default_user_data_dir
from motion_cut.vision.capture import _capture_apis_for_platform


class PlatformPathTests(unittest.TestCase):
    def test_linux_default_data_dir_uses_xdg(self) -> None:
        path = _default_user_data_dir(
            platform_name="linux", env={"XDG_DATA_HOME": "/tmp/xdg"}, home="/home/u"
        )
        self.assertEqual(path, "/tmp/xdg/motion-cut")

    def test_macos_default_data_dir(self) -> None:
        path = _default_user_data_dir(platform_name="darwin", env={}, home="/Users/test")
        self.assertEqual(path, "/Users/test/Library/Application Support/motion-cut")


class CaptureBackendTests(unittest.TestCase):
    def test_macos_capture_prefers_avfoundation(self) -> None:
        apis = _capture_apis_for_platform("darwin")
        self.assertEqual(apis[0], cv2.CAP_AVFOUNDATION)
        self.assertIn(cv2.CAP_ANY, apis)


if __name__ == "__main__":
    unittest.main()