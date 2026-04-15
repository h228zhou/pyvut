"""High-level API for streaming 6DoF poses from VIVE Ultimate Trackers."""

from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from .tracker_core import ViveTrackerGroup, mac_str, set_tracker_core_debug, stage_print

logger = logging.getLogger(__name__)

POSE_SLOTS = 5
POSE_FIELDS = 11  # [px, py, pz, rw, rx, ry, rz, timestamp_ms, buttons, tracking_status, valid_flag]
MAC_STR_LEN = 64
SN_STR_LEN = 64


class SharedPoseBuffer:
    def __init__(self):
        total_bytes = POSE_SLOTS * POSE_FIELDS * np.dtype(np.float64).itemsize
        self.shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        self.array = np.ndarray((POSE_SLOTS, POSE_FIELDS), dtype=np.float64, buffer=self.shm.buf)
        self.lock = mp.Lock()
        self.mac_buffer = mp.Array('c', MAC_STR_LEN * POSE_SLOTS)
        self.sn_buffer = mp.Array('c', SN_STR_LEN * POSE_SLOTS)
        self.write_timestamps = mp.Array('d', POSE_SLOTS)
        self.sequence_numbers = mp.Array('L', POSE_SLOTS)
        self._owns_shm = True
        self._mac2id_map = dict()

    @classmethod
    def attach(
        cls,
        shm_name: str,
        lock,
        mac_buffer,
        sn_buffer,
        write_timestamps,
        sequence_numbers,
    ) -> "SharedPoseBuffer":
        total_bytes = POSE_SLOTS * POSE_FIELDS * np.dtype(np.float64).itemsize
        instance = cls.__new__(cls)
        instance.shm = shared_memory.SharedMemory(name=shm_name)
        instance.array = np.ndarray((POSE_SLOTS, POSE_FIELDS), dtype=np.float64, buffer=instance.shm.buf)
        instance.lock = lock
        instance.mac_buffer = mac_buffer
        instance.sn_buffer = sn_buffer
        instance.write_timestamps = write_timestamps
        instance.sequence_numbers = sequence_numbers
        instance._owns_shm = False
        instance._mac2id_map = dict()
        return instance

    def close(self):
        self.shm.close()
        if self._owns_shm:
            self.shm.unlink()

    def write_pose(self, tracker_index: int, pose: "TrackerPose") -> None:
        if tracker_index < 0 or tracker_index >= POSE_SLOTS:
            return
        str_mac = pose.mac
        if not str_mac in self._mac2id_map:
            self._mac2id_map[str_mac] = tracker_index
        with self.lock:
            row = self.array[tracker_index]
            row[:3] = pose.position
            row[3:7] = pose.rotation
            row[7] = pose.timestamp_ms
            row[8] = float(pose.buttons)
            row[9] = float(pose.tracking_status)
            row[10] = 1.0
            self._write_identity_locked(tracker_index, pose.mac, pose.sn)
            self.write_timestamps[tracker_index] = time.time()
            self.sequence_numbers[tracker_index] += 1

    def write_identity(self, tracker_index: int, mac: str = "", sn: str = "") -> None:
        if tracker_index < 0 or tracker_index >= POSE_SLOTS:
            return
        with self.lock:
            self._write_identity_locked(tracker_index, mac, sn)

    def _write_identity_locked(self, tracker_index: int, mac: str, sn: str) -> None:
        raw_mac = (mac or "").encode("utf-8")[:MAC_STR_LEN - 1]
        raw_sn = (sn or "").encode("utf-8")[:SN_STR_LEN - 1]
        offset = tracker_index * MAC_STR_LEN
        sn_offset = tracker_index * SN_STR_LEN
        self.mac_buffer[offset:offset + MAC_STR_LEN] = b"\x00" * MAC_STR_LEN
        self.mac_buffer[offset:offset + len(raw_mac)] = raw_mac
        self.sn_buffer[sn_offset:sn_offset + SN_STR_LEN] = b"\x00" * SN_STR_LEN
        self.sn_buffer[sn_offset:sn_offset + len(raw_sn)] = raw_sn

    def read_identity(self, tracker_index: int) -> Dict[str, str]:
        if tracker_index < 0 or tracker_index >= POSE_SLOTS:
            return {"mac": "", "sn": ""}
        offset = tracker_index * MAC_STR_LEN
        sn_offset = tracker_index * SN_STR_LEN
        with self.lock:
            raw_mac = bytes(self.mac_buffer[offset:offset + MAC_STR_LEN])
            raw_sn = bytes(self.sn_buffer[sn_offset:sn_offset + SN_STR_LEN])
        return {
            "mac": raw_mac.split(b"\x00", 1)[0].decode("utf-8", errors="ignore"),
            "sn": raw_sn.split(b"\x00", 1)[0].decode("utf-8", errors="ignore"),
        }

    def read_pose(self, tracker_index: int) -> Optional[Dict]:
        if tracker_index < 0 or tracker_index >= POSE_SLOTS:
            return None
        offset = tracker_index * MAC_STR_LEN
        sn_offset = tracker_index * SN_STR_LEN
        with self.lock:
            row = self.array[tracker_index].copy()
            write_time = self.write_timestamps[tracker_index]
            sequence = int(self.sequence_numbers[tracker_index])
            raw_mac = bytes(self.mac_buffer[offset:offset + MAC_STR_LEN])
            raw_sn = bytes(self.sn_buffer[sn_offset:sn_offset + SN_STR_LEN])
        if row[10] < 0.5:
            return None
        mac = raw_mac.split(b"\x00", 1)[0]
        sn = raw_sn.split(b"\x00", 1)[0]
        return {
            "position": row[:3],
            "rotation": row[3:7],
            "timestamp_ms": int(row[7]),
            "buttons": int(row[8]),
            "tracking_status": int(row[9]),
            "mac": mac.decode("utf-8", errors="ignore"),
            "sn": sn.decode("utf-8", errors="ignore"),
            "write_time": write_time,
            "sequence": sequence,
        }
        
    def read_pose_by_mac(self, tracker_mac: str):
        if not tracker_mac in self._mac2id_map:
            logger.warning(f'Unknown tracker MAC: {tracker_mac}, now we have {self._mac2id_map}')
            return None
        
        tracker_id = self._mac2id_map[tracker_mac]
        assert isinstance(tracker_id, int), f"mac's tracker id {tracker_id} is not int but {type(tracker_id)}"
        return self.read_pose(tracker_id)

# pair_on_startup is true means manual pairing is required
def _tracker_process_main(
    mode: str,
    wifi_info_path: Optional[str],
    debug: bool,
    pair_on_startup: bool,
    shm_name: str,
    lock,
    mac_buffer,
    sn_buffer,
    write_timestamps,
    sequence_numbers,
    stop_event,
    startup_error_queue,
):
    try:
        api = UltimateTrackerAPI(
            mode=mode,
            wifi_info_path=wifi_info_path,
            debug=debug,
            pair_on_startup=pair_on_startup,
        )
        buffer = SharedPoseBuffer.attach(shm_name, lock, mac_buffer, sn_buffer, write_timestamps, sequence_numbers)

        def handle_pose(pose: TrackerPose) -> None:
            buffer.write_pose(pose.tracker_index, pose)

        api.add_pose_callback(handle_pose)
        api.start()
        while not stop_event.is_set():
            tracker_group = api.tracker_group
            for tracker_index, sn in enumerate(tracker_group.tracker_sn):
                if sn:
                    buffer.write_identity(tracker_index, sn=sn)
            stop_event.wait(0.005)
    except Exception as exc:
        startup_error_queue.put(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if 'api' in locals():
            api.stop()
        if 'buffer' in locals():
            buffer.close()



@dataclass
class TrackerPose:
    """Representation of a single 6DoF pose sample coming from a tracker."""

    tracker_index: int
    mac: str
    sn: str
    buttons: int
    tracking_status: int
    timestamp_ms: int
    position: np.ndarray
    rotation: np.ndarray
    acceleration: np.ndarray
    angular_velocity: np.ndarray


PoseCallback = Callable[[TrackerPose], None]


class UltimateTrackerAPI:
    """Convenience wrapper that exposes tracker poses through a simple API."""

    def __init__(
        self,
        mode: str = "DONGLE_USB",
        poll_interval: float = 0.001,
        wifi_info_path: Optional[str] = None,
        debug: bool = False,
        pair_on_startup: bool = False,
    ) -> None:
        set_tracker_core_debug(debug)
        stage_print(
            "A1",
            f"UltimateTrackerAPI init: mode={mode} poll_interval={poll_interval} pair_on_startup={pair_on_startup}",
        )
        self._group = ViveTrackerGroup(
            mode=mode,
            wifi_info_path=wifi_info_path,
            debug=debug,
            pair_on_startup=pair_on_startup,
        )
        self._poll_interval = max(0.0, poll_interval)
        self._pose_callbacks: List[PoseCallback] = []
        self._latest_pose: Dict[int, TrackerPose] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        self._group.add_pose_listener(self._handle_pose_event)
        stage_print("A2", "UltimateTrackerAPI pose listener attached")

    @property
    def tracker_group(self) -> ViveTrackerGroup:
        """Access the underlying ViveTrackerGroup for advanced workflows."""

        return self._group

    def start(self) -> None:
        """Start polling HID devices for pose data in a background thread."""

        if self._thread and self._thread.is_alive():
            return

        stage_print("A3", "Starting UltimateTrackerAPI polling thread")
        self._running.set()
        self._thread = threading.Thread(target=self._loop_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop polling and join the background thread."""

        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> "UltimateTrackerAPI":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def add_pose_callback(self, callback: PoseCallback) -> None:
        """Register a callback that receives TrackerPose objects as they stream in."""

        if callback not in self._pose_callbacks:
            self._pose_callbacks.append(callback)

    def remove_pose_callback(self, callback: PoseCallback) -> None:
        self._pose_callbacks = [cb for cb in self._pose_callbacks if cb != callback]

    def get_latest_pose(self, tracker_index: int) -> Optional[TrackerPose]:
        """Return the most recent pose for the requested tracker index, if available."""

        with self._lock:
            return self._latest_pose.get(tracker_index)

    def iter_latest_poses(self) -> Iterable[TrackerPose]:
        """Iterate over the most recent poses for all trackers that have reported."""

        with self._lock:
            return list(self._latest_pose.values())

    def _loop_forever(self) -> None:
        stage_print("A4", "UltimateTrackerAPI polling thread entered main loop")
        while self._running.is_set():
            self._group.do_loop()
            if self._poll_interval:
                time.sleep(self._poll_interval)

    def _handle_pose_event(self, sample: Dict) -> None:
        pose = TrackerPose(
            tracker_index=sample["tracker_index"],
            mac=mac_str(sample["mac"]),
            sn=sample.get("sn", ""),
            buttons=sample["buttons"],
            tracking_status=sample["tracking_status"],
            timestamp_ms=sample["timestamp_ms"],
            position=sample["position"],
            rotation=sample["rotation"],
            acceleration=sample["acceleration"],
            angular_velocity=sample["angular_velocity"],
        )

        with self._lock:
            self._latest_pose[pose.tracker_index] = pose

        for callback in list(self._pose_callbacks):
            try:
                callback(pose)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Pose callback raised an exception")


class TrackerService:
    """Runs the tracker polling loop inside its own process and shares poses through shared memory."""

    def __init__(
        self,
        mode: str = "DONGLE_USB",
        wifi_info_path: Optional[str] = None,
        debug: bool = False,
        pair_on_startup: bool = False,
    ):
        self._running = False
        self._buffer = SharedPoseBuffer()
        self._stop_event = mp.Event()
        self._startup_error_queue = mp.Queue(maxsize=1)
        self._process = mp.Process(
            target=_tracker_process_main,
            args=(
                mode,
                wifi_info_path,
                debug,
                pair_on_startup,
                self._buffer.shm.name,
                self._buffer.lock,
                self._buffer.mac_buffer,
                self._buffer.sn_buffer,
                self._buffer.write_timestamps,
                self._buffer.sequence_numbers,
                self._stop_event,
                self._startup_error_queue,
            ),
            daemon=True,
        )
        self._process.start()
        self._last_pose_age_ms: Optional[float] = None
        self._last_pose_sequence: Optional[int] = None
        self._running = True
        self._assert_process_started()

    @property
    def trackers(self):
        return None

    @property
    def last_pose_age_ms(self) -> Optional[float]:
        return self._last_pose_age_ms

    @property
    def last_pose_sequence(self) -> Optional[int]:
        return self._last_pose_sequence

    def get_pose(self, tracker_index: int) -> Optional[TrackerPose]:
        data = self._buffer.read_pose(tracker_index)
        if data is None:
            return None
        self._last_pose_age_ms = (time.time() - data["write_time"]) * 1000.0
        self._last_pose_sequence = data["sequence"]
        return TrackerPose(
            tracker_index=tracker_index,
            mac=data["mac"],
            sn=data["sn"],
            buttons=data["buttons"],
            tracking_status=data["tracking_status"],
            timestamp_ms=data["timestamp_ms"],
            position=np.array(data["position"], dtype=float),
            rotation=np.array(data["rotation"], dtype=float),
            acceleration=np.zeros(3),
            angular_velocity=np.zeros(3),
        )

    def get_identity(self, tracker_index: int) -> Dict[str, str]:
        return self._buffer.read_identity(tracker_index)

    def is_alive(self) -> bool:
        return bool(self._process and self._process.is_alive())

    def assert_healthy(self) -> None:
        if self.is_alive():
            return
        startup_error = self._read_startup_error()
        detail = f": {startup_error}" if startup_error else ""
        raise RuntimeError(f"Tracker backend process is not running{detail}")

    def _assert_process_started(self, startup_timeout_s: float = 0.5) -> None:
        deadline = time.time() + max(startup_timeout_s, 0.0)
        while time.time() < deadline and self._process.is_alive():
            startup_error = self._read_startup_error()
            if startup_error:
                self.stop()
                raise RuntimeError(f"Tracker backend failed during startup: {startup_error}")
            time.sleep(0.01)
        if self._process.is_alive():
            return
        startup_error = self._read_startup_error()
        self.stop()
        detail = f": {startup_error}" if startup_error else f" with exit code {self._process.exitcode}"
        raise RuntimeError(f"Tracker backend exited during startup{detail}")

    def _read_startup_error(self) -> Optional[str]:
        try:
            return self._startup_error_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        if not self._running:
            return
        self._stop_event.set()
        if self._process.is_alive():
            self._process.join(timeout=2.0)
        self._buffer.close()
        self._running = False
        self._process = None

    def __enter__(self) -> "TrackerService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
