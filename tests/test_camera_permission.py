from __future__ import annotations

import unittest
from unittest import mock

from motion_cut.vision import capture


class _FakeCaptureDevice:
    def __init__(self, status: int, grant_on_request: bool | None = None) -> None:
        self._status = status
        self._grant_on_request = grant_on_request
        self.requested = False

    def authorizationStatusForMediaType_(self, _media_type):
        return self._status

    def requestAccessForMediaType_completionHandler_(self, _media_type, handler):
        self.requested = True
        if self._grant_on_request is not None:
            handler(self._grant_on_request)


class _FakeAVFoundation:
    AVAuthorizationStatusNotDetermined = 0
    AVAuthorizationStatusAuthorized = 3
    AVMediaTypeVideo = "video"

    def __init__(self, capture_device: _FakeCaptureDevice) -> None:
        self.AVCaptureDevice = capture_device


class MacCameraPermissionTests(unittest.TestCase):
    def test_non_macos_skips_permission_check(self) -> None:
        allowed, message = capture._ensure_macos_camera_permission(platform_name="linux")
        self.assertTrue(allowed)
        self.assertEqual(message, "")

    def test_denied_permission_returns_user_actionable_error(self) -> None:
        allowed, message = capture._ensure_macos_camera_permission(
            platform_name="darwin", avfoundation=_FakeAVFoundation(_FakeCaptureDevice(status=2))
        )
        self.assertFalse(allowed)
        self.assertIn("System Settings", message)

    def test_not_determined_requests_access(self) -> None:
        device = _FakeCaptureDevice(status=0, grant_on_request=True)
        allowed, message = capture._ensure_macos_camera_permission(
            platform_name="darwin", avfoundation=_FakeAVFoundation(device)
        )
        self.assertTrue(allowed)
        self.assertEqual(message, "")
        self.assertTrue(device.requested)

    def test_missing_avfoundation_does_not_hard_block(self) -> None:
        with mock.patch("motion_cut.vision.capture.importlib.import_module", side_effect=ModuleNotFoundError):
            allowed, message = capture._ensure_macos_camera_permission(platform_name="darwin")
        self.assertTrue(allowed)
        self.assertEqual(message, "")


if __name__ == "__main__":
    unittest.main()