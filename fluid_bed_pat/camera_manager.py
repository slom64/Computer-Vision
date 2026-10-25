import os
import glob
import time
import sys
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

# Try importing Daheng Imaging SDK (gxipy) from environment or standard GalaxySDK paths
GXIPY_AVAILABLE = False
try:
    import gxipy as gx
    GXIPY_AVAILABLE = True
except ImportError:
    sdk_paths = [
        r"C:\Program Files\DAHENG IMAGING\GalaxySDK\Development\Samples\Python\api",
        r"C:\Program Files (x86)\DAHENG IMAGING\GalaxySDK\Development\Samples\Python\api",
        r"C:\GalaxySDK\Development\Samples\Python\api",
        r"C:\Program Files\DAHENG IMAGING\GalaxySDK\Development\Samples\Python\gxipy",
    ]
    for sp in sdk_paths:
        if os.path.exists(sp) and sp not in sys.path:
            sys.path.append(sp)
    try:
        import gxipy as gx
        GXIPY_AVAILABLE = True
    except ImportError:
        GXIPY_AVAILABLE = False

# Try importing DirectShow device enumerator
DSHOW_ENUM_AVAILABLE = False
try:
    from pygrabber.dshow_graph import FilterGraph
    DSHOW_ENUM_AVAILABLE = True
except ImportError:
    DSHOW_ENUM_AVAILABLE = False


@dataclass
class CameraDeviceInfo:
    device_id: Any
    name: str
    backend: str  # 'gxipy', 'directshow', 'dataset', 'file'
    description: str = ""
    resolution: Tuple[int, int] = (2048, 1536)
    max_fps: float = 125.0


class IndustrialCameraManager:
    """
    Industrial Camera Interface supporting:
    1. Daheng Imaging USB 3.0 Cameras (e.g. MER2-301-125U3M) via native gxipy SDK
    2. DirectShow / UVC Video Devices via OpenCV DirectShow backend
    3. Replay Datasets (e.g. Real-Data BMP sequences) for simulation & offline testing
    4. Video File Replay (MP4, AVI)
    """

    def __init__(self):
        self.current_device_info: Optional[CameraDeviceInfo] = None
        self.is_connected: bool = False
        self.is_streaming: bool = False
        
        # Backend-specific handles
        self._gx_device = None
        self._cv_capture: Optional[cv2.VideoCapture] = None
        self._replay_files: List[str] = []
        self._replay_idx: int = 0
        self._replay_fps: float = 30.0
        self._replay_loop: bool = True
        
        # Threading & frame queue
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_count: int = 0
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        
        # FPS estimation
        self._fps_history: List[float] = []
        self._actual_fps: float = 0.0
        self._last_fps_calc_time = time.time()
        
        # Camera parameters
        self.target_fps: float = 60.0
        self.exposure_time_us: float = 10000.0  # 10ms default
        self.gain_db: float = 0.0

    @staticmethod
    def enumerate_cameras() -> List[CameraDeviceInfo]:
        """
        Auto-detects all accessible camera devices across all backends:
        1. Native Daheng Galaxy USB3 cameras (gxipy)
        2. DirectShow / Windows Cameras (pygrabber)
        3. Real Dataset directories found in workspace (Virtual Feeds)
        """
        devices: List[CameraDeviceInfo] = []

        # 1. Native Daheng Galaxy SDK
        if GXIPY_AVAILABLE:
            try:
                device_manager = gx.DeviceManager()
                dev_num, dev_info_list = device_manager.update_device_list()
                for i, info in enumerate(dev_info_list):
                    sn = info.get("sn", f"SN_{i}")
                    model = info.get("model_name", "Daheng MER2 Camera")
                    user_id = info.get("user_id", "")
                    label = f"Daheng {model} (SN: {sn})" if not user_id else f"{user_id} - {model}"
                    devices.append(CameraDeviceInfo(
                        device_id=i + 1,  # gxipy is 1-indexed for open_device_by_index
                        name=label,
                        backend="gxipy",
                        description=f"Native Galaxy SDK USB3 Vision: {model} [SN:{sn}]",
                        resolution=(2048, 1536),
                        max_fps=125.0
                    ))
            except Exception as e:
                print(f"[IndustrialCameraManager] Notice on gxipy enum: {e}")

        # 2. DirectShow Cameras (via pygrabber or fallback probing)
        if DSHOW_ENUM_AVAILABLE:
            try:
                graph = FilterGraph()
                dshow_names = graph.get_input_devices()
                for idx, name in enumerate(dshow_names):
                    # Check if this name is a Daheng or general camera
                    is_daheng = ("MER2" in name.upper()) or ("DAHENG" in name.upper()) or ("GALAXY" in name.upper())
                    devices.append(CameraDeviceInfo(
                        device_id=idx,
                        name=f"{name} [DirectShow]",
                        backend="directshow",
                        description=f"DirectShow Device Index {idx}" + (" (Daheng Industrial)" if is_daheng else ""),
                        resolution=(2048, 1536) if is_daheng else (1920, 1080),
                        max_fps=125.0 if is_daheng else 30.0
                    ))
            except Exception as e:
                print(f"[IndustrialCameraManager] Notice on DirectShow enum: {e}")
        else:
            # Fallback probe first 3 indices
            for idx in range(3):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    devices.append(CameraDeviceInfo(
                        device_id=idx,
                        name=f"Camera Device {idx} [DirectShow]",
                        backend="directshow",
                        description=f"DirectShow Camera Index {idx}"
                    ))
                    cap.release()

        # 3. Built-in Real-Data Virtual Streams (for fluid bed testing)
        real_data_dir = "Real-Data"
        if os.path.exists(real_data_dir):
            for t_dir in sorted(glob.glob(os.path.join(real_data_dir, "T*"))):
                t_name = os.path.basename(t_dir)
                for b_dir in sorted(glob.glob(os.path.join(t_dir, "B*"))):
                    b_name = os.path.basename(b_dir)
                    bmps = glob.glob(os.path.join(b_dir, "*.bmp"))
                    if bmps:
                        devices.append(CameraDeviceInfo(
                            device_id=b_dir,
                            name=f"Dataset Stream: {t_name}/{b_name} ({len(bmps)} frames)",
                            backend="dataset",
                            description=f"Pre-captured 2048x1536 Fluid Bed Stream ({len(bmps)} BMP frames)",
                            resolution=(2048, 1536),
                            max_fps=60.0
                        ))

        # Always guarantee at least one simulation fallback if nothing else is found
        if not devices:
            devices.append(CameraDeviceInfo(
                device_id="synthetic",
                name="Synthetic Fluid Bed Simulator",
                backend="dataset",
                description="Simulated fluid bed particle flow"
            ))

        return devices

    def connect(self, device_info: CameraDeviceInfo) -> bool:
        """Connects to the specified camera device."""
        self.disconnect()

        self.current_device_info = device_info
        backend = device_info.backend

        try:
            if backend == "gxipy":
                if not GXIPY_AVAILABLE:
                    raise RuntimeError("Daheng Galaxy SDK (gxipy) is not installed or available.")
                dm = gx.DeviceManager()
                self._gx_device = dm.open_device_by_index(device_info.device_id)
                
                # Apply initial camera settings
                try:
                    self._gx_device.ExposureTime.set(self.exposure_time_us)
                    self._gx_device.Gain.set(self.gain_db)
                except Exception as param_err:
                    print(f"[IndustrialCameraManager] Setting initial Daheng params notice: {param_err}")
                    
                self.is_connected = True
                print(f"[IndustrialCameraManager] Successfully connected to Daheng camera: {device_info.name}")
                return True

            elif backend == "directshow":
                idx = int(device_info.device_id)
                self._cv_capture = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not self._cv_capture.isOpened():
                    # Fallback to default backend
                    self._cv_capture = cv2.VideoCapture(idx)
                    
                if not self._cv_capture.isOpened():
                    raise RuntimeError(f"Could not open DirectShow camera index {idx}")
                    
                # Request camera resolution and high frame rate
                target_w, target_h = device_info.resolution
                self._cv_capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
                self._cv_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
                self._cv_capture.set(cv2.CAP_PROP_FPS, device_info.max_fps)
                
                self.is_connected = True
                print(f"[IndustrialCameraManager] Successfully connected to DirectShow camera: {device_info.name}")
                return True

            elif backend == "dataset":
                dataset_path = str(device_info.device_id)
                if os.path.isdir(dataset_path):
                    # Filter out user-labelled duplicates to stream clean frames
                    all_bmps = sorted(glob.glob(os.path.join(dataset_path, "*.bmp")))
                    valid_bmps = [f for f in all_bmps if not f.endswith("_labelled.bmp") and "_labelled" not in os.path.basename(f)]
                    if not valid_bmps:
                        valid_bmps = all_bmps
                    self._replay_files = valid_bmps
                else:
                    self._replay_files = []
                    
                self._replay_idx = 0
                self.is_connected = True
                print(f"[IndustrialCameraManager] Connected to Dataset feed: {device_info.name} ({len(self._replay_files)} frames)")
                return True

            elif backend == "file":
                filepath = str(device_info.device_id)
                self._cv_capture = cv2.VideoCapture(filepath)
                if not self._cv_capture.isOpened():
                    raise RuntimeError(f"Could not open video file: {filepath}")
                self.is_connected = True
                return True

            else:
                raise ValueError(f"Unknown backend type: {backend}")

        except Exception as e:
            print(f"[IndustrialCameraManager] Connection error: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        """Disconnects active camera and releases all resources."""
        self.stop_stream()
        
        if self._gx_device is not None:
            try:
                self._gx_device.close_device()
            except Exception:
                pass
            self._gx_device = None

        if self._cv_capture is not None:
            try:
                self._cv_capture.release()
            except Exception:
                pass
            self._cv_capture = None

        self._replay_files = []
        self._replay_idx = 0
        self.is_connected = False
        self.current_device_info = None

    def start_stream(self) -> bool:
        """Starts background acquisition thread."""
        if not self.is_connected or self.is_streaming:
            return False

        if self.current_device_info.backend == "gxipy" and self._gx_device:
            try:
                self._gx_device.stream_on()
            except Exception as e:
                print(f"[IndustrialCameraManager] stream_on error: {e}")
                return False

        self._stop_event.clear()
        self.is_streaming = True
        self._worker_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._worker_thread.start()
        return True

    def stop_stream(self):
        """Stops background acquisition thread."""
        if not self.is_streaming:
            return

        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        self._worker_thread = None

        if self.current_device_info and self.current_device_info.backend == "gxipy" and self._gx_device:
            try:
                self._gx_device.stream_off()
            except Exception:
                pass

        self.is_streaming = False

    def _capture_worker(self):
        """Dedicated high-speed frame grabbing worker."""
        backend = self.current_device_info.backend if self.current_device_info else ""
        fps_target = self.target_fps if self.target_fps > 0 else 30.0
        frame_interval = 1.0 / fps_target
        last_frame_time = time.time()
        frame_count_for_fps = 0
        fps_timer = time.time()

        while not self._stop_event.is_set():
            t_now = time.time()
            frame: Optional[np.ndarray] = None

            try:
                if backend == "gxipy":
                    raw_img = self._gx_device.data_stream[0].get_image(timeout=1000)
                    if raw_img is not None and raw_img.get_status() == gx.GxStatusList.SUCCESS:
                        frame = raw_img.get_numpy_array()
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                elif backend == "directshow" or backend == "file":
                    if self._cv_capture and self._cv_capture.isOpened():
                        ret, raw = self._cv_capture.read()
                        if ret and raw is not None:
                            frame = raw
                        elif backend == "file":
                            # Loop video
                            self._cv_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

                elif backend == "dataset":
                    if self._replay_files:
                        img_path = self._replay_files[self._replay_idx]
                        raw = cv2.imread(img_path)
                        if raw is not None:
                            frame = raw
                        self._replay_idx += 1
                        if self._replay_idx >= len(self._replay_files):
                            if self._replay_loop:
                                self._replay_idx = 0
                            else:
                                break
                    else:
                        # Synthetic particle noise pattern
                        frame = self._generate_synthetic_fluid_bed_frame()

                if frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_timestamp = t_now
                        self._frame_count += 1
                        
                    frame_count_for_fps += 1
                    if t_now - fps_timer >= 1.0:
                        self._actual_fps = frame_count_for_fps / (t_now - fps_timer)
                        frame_count_for_fps = 0
                        fps_timer = t_now

                # Precise rate limiter to match target FPS
                elapsed = time.time() - t_now
                sleep_time = frame_interval - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"[IndustrialCameraManager] Capture error: {e}")
                time.sleep(0.05)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float, int]:
        """
        Thread-safe retrieval of the freshest grabbed frame.
        Returns: (frame, timestamp, frame_count)
        """
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0, self._frame_count
            return self._latest_frame.copy(), self._latest_timestamp, self._frame_count

    def get_fps(self) -> float:
        """Returns the real measured camera acquisition FPS."""
        return self._actual_fps

    def set_exposure_us(self, exposure_us: float):
        """Sets camera exposure time in microseconds."""
        self.exposure_time_us = float(exposure_us)
        if self.current_device_info and self.current_device_info.backend == "gxipy" and self._gx_device:
            try:
                self._gx_device.ExposureTime.set(self.exposure_time_us)
            except Exception as e:
                print(f"[IndustrialCameraManager] Failed to set Daheng exposure: {e}")
        elif self._cv_capture and self._cv_capture.isOpened():
            try:
                # DirectShow exposure is typically in log2(seconds)
                # Map roughly: 10000us (10ms) -> -6
                exp_val = max(-13, min(-1, int(np.log2(exposure_us / 1000000.0))))
                self._cv_capture.set(cv2.CAP_PROP_EXPOSURE, exp_val)
            except Exception:
                pass

    def set_gain_db(self, gain_db: float):
        """Sets analog gain in dB."""
        self.gain_db = float(gain_db)
        if self.current_device_info and self.current_device_info.backend == "gxipy" and self._gx_device:
            try:
                self._gx_device.Gain.set(self.gain_db)
            except Exception as e:
                print(f"[IndustrialCameraManager] Failed to set Daheng gain: {e}")
        elif self._cv_capture and self._cv_capture.isOpened():
            try:
                self._cv_capture.set(cv2.CAP_PROP_GAIN, self.gain_db)
            except Exception:
                pass

    def set_target_fps(self, fps: float):
        """Adjusts the acquisition rate target."""
        self.target_fps = max(1.0, min(125.0, float(fps)))
        if self._cv_capture and self._cv_capture.isOpened():
            self._cv_capture.set(cv2.CAP_PROP_FPS, self.target_fps)

    def _generate_synthetic_fluid_bed_frame(self) -> np.ndarray:
        """Generates realistic synthetic monochrome fluid bed frame if no dataset exists."""
        img = np.full((1536, 2048, 3), 40, dtype=np.uint8)
        # Chamber wall on right
        cv2.line(img, (1876, 0), (1876, 1536), (80, 80, 80), 2)
        # Random pellets
        num_pellets = np.random.randint(60, 120)
        for _ in range(num_pellets):
            cx = np.random.randint(200, 1850)
            cy = np.random.randint(100, 1400)
            r = np.random.randint(14, 28)
            intensity = np.random.randint(180, 245)
            cv2.circle(img, (cx, cy), r, (intensity, intensity, intensity), -1)
            cv2.circle(img, (cx, cy), r, (120, 120, 120), 1)
        return img
