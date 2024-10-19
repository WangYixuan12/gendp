from typing import Optional, Callable, Dict, Union, Any
import os
import enum
import time
import warnings
import json
import numpy as np
from pyorbbecsdk import Context, DeviceList, \
    OBPropertyID, OBSensorType, OBFormat, OBAlignMode, \
        Pipeline, Config, StreamProfileList, VideoFrame

import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from gendp.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from gendp.shared_memory.shared_ndarray import SharedNDArray
from gendp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from gendp.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from gendp.real_world.video_recorder import VideoRecorder

MIN_DEPTH = 20 # mm
MAX_DEPTH = 10000 # mm


def yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    yuyv = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    return bgr_image


def uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    uyvy = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    return bgr_image


def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image

def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    elif color_format == OBFormat.BGRA:
        image = np.resize(data, (height, width, 4))[..., :3]
        return image
    else:
        raise ValueError(f'color_format {color_format} not supported.')

class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4

class SingleFemto(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1280,960),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=30,
            advanced_mode_config=None,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            verbose=False,
            extrinsics_dir=os.path.join(os.path.dirname(__file__), 'cam_extrinsics'),
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=shape, dtype=np.uint16)
            examples['intrinsics'] = np.empty(shape=(3,3), dtype=np.float32)
            examples['extrinsics'] = np.empty(shape=(4,4), dtype=np.float32)
            os.system(f'mkdir -p {extrinsics_dir}')
            if extrinsics_dir is None:
                self.extrinsics = np.eye(4)
                warnings.warn('extrinsics_dir is None, using identity matrix.')
            else:
                extrinsics_path = os.path.join(extrinsics_dir, f'{serial_number}.npy')
                if not os.path.exists(extrinsics_path):
                    self.extrinsics = np.eye(4)
                    warnings.warn(f'extrinsics_path {extrinsics_path} does not exist, using identity matrix.')
                else:
                    self.extrinsics = np.load(os.path.join(extrinsics_dir, f'{serial_number}.npy'))
        if enable_infrared:
            examples['infrared'] = np.empty(
                shape=shape, dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        # vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        #     shm_manager=shm_manager,
        #     examples=examples if vis_transform is None 
        #         else vis_transform(dict(examples)),
        #     get_max_k=1,
        #     get_time_budget=0.2,
        #     put_desired_frequency=capture_fps
        # )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT.value,
            'option_value': 20000.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        intrinsics_array.get()[:] = 0

        dist_coeff_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(8,),
                dtype=np.float64)
        dist_coeff_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            # realsense uses bgr24 pixel format
            # default thread_type to FRAEM
            # i.e. each frame uses one core
            # instead of all cores working on all frames.
            # this prevents CPU over-subpscription and
            # improves performance significantly
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps, 
                codec='h264',
                input_pix_fmt='bgr24', 
                crf=18,
                thread_type='FRAME',
                thread_count=1)

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None
        self.extrinsics_dir = extrinsics_dir

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        # self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
        self.dist_coeff_array = dist_coeff_array
    
    @staticmethod
    def get_connected_devices_serial():
        context = Context()
        device_list : DeviceList = context.query_devices()
        serials = list()
        for i in range(device_list.get_count()):
            serial = device_list.get_device_serial_number_by_index(i)
            serials.append(serial)
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    # def get_vis(self, out=None):
    #     return self.vis_ring_buffer.get(out=out)
    
    # ========= user API ===========
    def set_color_option(self, option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })
    
    def set_depth_option(self, option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_DEPTH_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })
    
    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, 1.0)
        else:
            # manual exposure
            self.set_color_option(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, 0.0)
            if exposure is not None:
                self.set_color_option(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure)
            if gain is not None:
                self.set_color_option(OBPropertyID.OB_PROP_COLOR_GAIN_INT, gain)
    
    def set_depth_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_depth_option(OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, 1.0)
        else:
            # manual exposure
            self.set_depth_option(OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, 0.0)
            if exposure is not None:
                self.set_color_option(OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, exposure)
            if gain is not None:
                self.set_color_option(OBPropertyID.OB_PROP_DEPTH_GAIN_INT, exposure)
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, 1.0)
        else:
            self.set_color_option(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, 0.0)
            self.set_color_option(OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = ppx
        mat[1,2] = ppy
        return mat

    def get_dist_coeff(self):
        assert self.ready_event.is_set()
        return np.array(self.dist_coeff_array.get()[:])

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale
    
    def start_recording(self, video_path: str, start_time: float=-1):
        assert self.enable_color

        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })
    
    def calibrate_extrinsics(self, visualize=True, board_size=(6,9), squareLength=0.03, markerLength=0.022):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        board = cv2.aruco.CharucoBoard(
            size=board_size,
            squareLength=squareLength,
            markerLength=markerLength,
            dictionary=dictionary,
        )
        while not self.ready_event.is_set():
            time.sleep(0.1)
        intrinsic_matrix = self.get_intrinsics()
        dist_coef = self.get_dist_coeff()
        
        R_gripper2base = []
        t_gripper2base = []
        R_board2cam = []
        t_board2cam = []
        rgbs = []
        depths = []
        point_list = []
        masks = []


        # Calculate the markers
        out = self.ring_buffer.get()
        # points, colors, depth_img, mask = self.camera.get_observations()
        colors = out['color']
        calibration_img = colors.copy()
        cv2.imshow(f"calibration_img_{self.serial_number}", calibration_img)
        cv2.waitKey(1)
        # import pdb
        # pdb.set_trace()
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            image=calibration_img,
            dictionary=dictionary,
            parameters=None,
        )
        if len(corners) == 0:
            warnings.warn('no markers detected')
            return

        # import pdb
        # pdb.set_trace()

        # Calculate the charuco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=calibration_img,
            board=board,
            cameraMatrix=intrinsic_matrix,
            distCoeffs=dist_coef,
        )
        # all_charuco_corners.append(charuco_corners)
        # all_charuco_ids.append(charuco_ids)

        if charuco_corners is None:
            warnings.warn('no charuco corners detected')
            return

        print("number of corners: ", len(charuco_corners))

        visualize = True
        if visualize:
            cv2.aruco.drawDetectedCornersCharuco(
                image=calibration_img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
            )
            cv2.imshow(f"calibration_{self.serial_number}", calibration_img)
            cv2.waitKey(1)

        rvec = None
        tvec = None
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            board,
            intrinsic_matrix,
            dist_coef,
            rvec,
            tvec,
        )
        
        if not retval:
            warnings.warn('pose estimation failed')
            return

        reprojected_points, _ = cv2.projectPoints(
            board.getChessboardCorners()[charuco_ids, :],
            rvec,
            tvec,
            intrinsic_matrix,
            dist_coef,
        )

        # Reshape for easier handling
        reprojected_points = reprojected_points.reshape(-1, 2)
        charuco_corners = charuco_corners.reshape(-1, 2)

        # Calculate the error
        error = np.sqrt(
            np.sum((reprojected_points - charuco_corners) ** 2, axis=1)
        ).mean()

        print("Reprojection Error:", error)


        print("retval: ", retval)
        print("rvec: ", rvec)
        print("tvec: ", tvec)
        if not retval:
            raise ValueError("pose estimation failed")
        R_board2cam=cv2.Rodrigues(rvec)[0]
        t_board2cam=tvec[:, 0]
        print("R_board2cam: ", R_board2cam)
        print("t_board2cam: ", t_board2cam)

        tf = np.eye(4)
        tf[:3, :3] = R_board2cam
        tf[:3, 3] = t_board2cam
        
        tf_world2board = np.eye(4)
        tf_world2board[:3, :3] = np.array([[0, -1, 0],
                                           [-1, 0, 0],
                                           [0, 0, -1]])
        tf = tf @ tf_world2board
        # #make tf as list
        # tf = tf.tolist()
        # print("tf: ", tf)
        
        np.save(os.path.join(self.extrinsics_dir, f'{self.serial_number}.npy'), tf)
    
    # def _init_depth_process(self):
        # self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_HOLEFILTER_BOOL, True)
        # self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_POSTFILTER_BOOL, True)
        # self.device.set_bool_property(OBPropertyID.OB_PROP_DISPARITY_TO_DEPTH_BOOL, True)
     
    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        # cv2.setNumThreads(1)
        w, h = self.resolution
        fps = self.capture_fps
        # Enable the streams from all the intel realsense devices
        context = Context()
        device_list : DeviceList = context.query_devices()
        self.device = device_list.get_device_by_serial_number(self.serial_number)
        fb_config = Config()
        pipeline = Pipeline(self.device)
        color_profile_list : StreamProfileList = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profile_list : StreamProfileList = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if self.enable_color:
            color_profile = color_profile_list.get_video_stream_profile(w, h, OBFormat.BGRA, fps)
            fb_config.enable_stream(color_profile)
        if self.enable_depth:
            depth_profile = depth_profile_list.get_video_stream_profile(640, 576, OBFormat.Y16, fps)
            fb_config.enable_stream(depth_profile)
            fb_config.set_align_mode(OBAlignMode.SW_MODE)
            pipeline.enable_frame_sync()
            # self._init_depth_process()
        
        try:
            pipeline.start(fb_config)

            # get camera param
            camera_param = pipeline.get_camera_param()
            self.intrinsics_array.get()[0] = camera_param.rgb_intrinsic.fx
            self.intrinsics_array.get()[1] = camera_param.rgb_intrinsic.fy
            self.intrinsics_array.get()[2] = camera_param.rgb_intrinsic.cx
            self.intrinsics_array.get()[3] = camera_param.rgb_intrinsic.cy
            self.intrinsics_array.get()[4] = camera_param.rgb_intrinsic.height
            self.intrinsics_array.get()[5] = camera_param.rgb_intrinsic.width
            
            self.dist_coeff_array.get()[:] = np.array([camera_param.rgb_distortion.k1,
                                                       camera_param.rgb_distortion.k2,
                                                       camera_param.rgb_distortion.p1,
                                                       camera_param.rgb_distortion.p2,
                                                       camera_param.rgb_distortion.k3,
                                                       camera_param.rgb_distortion.k4,
                                                       camera_param.rgb_distortion.k5,
                                                       camera_param.rgb_distortion.k6,])
            
            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f'[SingleFemto {self.serial_number}] Main loop started.')

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            for _ in range(30):
                pipeline.wait_for_frames(100)

            while not self.stop_event.is_set():
                wait_start_time = time.time()
                # wait for frames to come in
                frameset = pipeline.wait_for_frames(100)
                if frameset is None or frameset.get_color_frame() is None or (frameset.get_depth_frame() is None and self.enable_depth):
                    print('frameset is None.')
                    continue
                
                # change intrinsics if necessary
                if self.intrinsics_array.get()[-1] == 0 and self.enable_depth:
                    self.intrinsics_array.get()[-1] = frameset.get_depth_frame().get_depth_scale()
                
                receive_time = time.time()
                # align frames to color
                wait_time = time.time() - wait_start_time

                # grab data
                grab_start_time = time.time()
                data = dict()
                data['camera_receive_timestamp'] = receive_time
                # realsense report in ms
                data['camera_capture_timestamp'] = frameset.get_timestamp() / 1000
                if self.enable_color:
                    color_frame = frameset.get_color_frame()
                    color_image = frame_to_bgr_image(color_frame)
                    data['color'] = color_image
                    t = color_frame.get_timestamp() / 1000
                    data['camera_capture_timestamp'] = t
                    # print('device', time.time() - t)
                    # print(color_frame.get_frame_timestamp_domain())
                if self.enable_depth:
                    depth_frame = frameset.get_depth_frame()
                    width = depth_frame.get_width()
                    height = depth_frame.get_height()
                    scale = depth_frame.get_depth_scale()

                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_data = depth_data.reshape((height, width))
                    depth_data = depth_data.astype(np.float32) * scale
                    depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
                    depth_data = depth_data.astype(np.uint16)
                    data['depth'] = depth_data
                    if self.is_ready:
                        data['intrinsics'] = self.get_intrinsics()
                    else:
                        warnings.warn('Intrinsics not ready, using identity matrix temporarily.')
                        data['intrinsics'] = np.eye(3)
                    data['extrinsics'] = self.extrinsics
                if self.enable_infrared:
                    data['infrared'] = np.asarray(
                        frameset.get_infrared_frame().get_data())
                grab_time = time.time() - grab_start_time
                if self.verbose:
                    print(f'[SingleFemto {self.serial_number}] Grab data time {grab_time}')
                
                # apply transform
                transform_start_time = time.time()
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:                
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[receive_time],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        # put_data['timestamp'] = put_start_time + step_idx / self.put_fps
                        put_data['timestamp'] = receive_time
                        # print(step_idx, data['timestamp'])
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)
                transform_time = time.time() - transform_start_time
                if self.verbose:
                    print(f'[SingleFemto {self.serial_number}] Transform time {transform_time}')

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                
                # # put to vis
                # vis_start_time = time.time()
                # vis_data = data
                # if self.vis_transform == self.transform:
                #     vis_data = put_data
                # elif self.vis_transform is not None:
                #     vis_data = self.vis_transform(dict(data))
                # self.vis_ring_buffer.put(vis_data, wait=False)
                # vis_time = time.time() - vis_start_time
                # if self.verbose:
                #     print(f'[SingleFemto {self.serial_number}] Vis time {vis_time}')
                
                # record frame
                rec_start_time = time.time()
                rec_data = data
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))

                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(rec_data['color'], 
                        frame_time=receive_time)
                rec_time = time.time() - rec_start_time
                if self.verbose:
                    print(f'[SingleFemto {self.serial_number}] Record time {rec_time}')

                # fetch command from queue
                cmd_start = time.time()
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    option = OBPropertyID(command['option_enum'])
                    option_name = option.name
                    is_bool = option_name.endswith('BOOL')
                    if cmd == Command.SET_COLOR_OPTION.value or cmd == Command.SET_DEPTH_OPTION.value:
                        if is_bool:
                            value = bool(command['option_value'])
                            self.device.set_bool_property(option, value)
                        else:
                            value = int(command['option_value'])
                            self.device.set_int_property(option, value)
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop()
                        # stop need to flush all in-flight frames to disk, which might take longer than dt.
                        # soft-reset put to drop frames to prevent ring buffer overflow.
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                        # self.ring_buffer.clear()
                cmd_time = time.time() - cmd_start
                if self.verbose:
                    print(f'[SingleFemto {self.serial_number}] Command time {cmd_time}')
                    print(f'[SingleFemto {self.serial_number}] white balance {self.device.get_int_property(OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT)}')
                    print(f'[SingleFemto {self.serial_number}] exposure {self.device.get_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)}')
                    print(f'[SingleFemto {self.serial_number}] gain {self.device.get_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT)}')

                iter_idx += 1
                
                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if frequency < fps // 2:
                    warnings.warn(f'[{self.serial_number}] FPS {frequency} is much smaller than {fps}.')
                    print('debugging info:')
                    print('wait_time:', wait_time)
                    print('grab_time:', grab_time)
                    print('transform_time:', transform_time)
                    # print('vis_time:', vis_time)
                    print('rec_time:', rec_time)
                    print('cmd_time:', cmd_time)
                if self.verbose:
                    print(f'[SingleFemto {self.serial_number}] FPS {frequency}')
        finally:
            self.video_recorder.stop()
            fb_config.disable_all_stream()
            self.ready_event.set()
        
        if self.verbose:
            print(f'[SingleFemto {self.serial_number}] Exiting worker process.')

def get_real_exporure_gain_white_balance():
    series_number = SingleFemto.get_connected_devices_serial()
    with SharedMemoryManager() as shm_manager:
        with SingleFemto(
            shm_manager=shm_manager,
            serial_number=series_number[0],
            enable_color=True,
            enable_depth=True,
            enable_infrared=False,
            put_fps=30,
            record_fps=30,
            verbose=True,
        ) as realsense:
            # realsense.set_exposure()
            # realsense.set_white_balance()
            realsense.set_exposure(168, 50)
            realsense.set_white_balance(2700)

            for i in range(30):
                realsense.get()
                time.sleep(0.1)
            
            cv2.imshow('color', realsense.get()['color'])
            cv2.waitKey(0)

def test_rgbd():
    import open3d as o3d
    from d3fields.utils.draw_utils import depth2fgpcd, np2o3d
    
    series_number = SingleFemto.get_connected_devices_serial()
    with SharedMemoryManager() as shm_manager:
        with SingleFemto(
            shm_manager=shm_manager,
            serial_number=series_number[0],
            enable_color=True,
            enable_depth=True,
            enable_infrared=False,
            put_fps=30,
            record_fps=30,
            verbose=True,
        ) as realsense:
            realsense.set_exposure(168, 50)
            realsense.set_white_balance(2700)

            visualizer = o3d.visualization.Visualizer()
            vis_pcd = o3d.geometry.PointCloud()
            visualizer.create_window()
            iter_idx = 0
            while True:
                msg = realsense.get()
                depth = msg['depth']
                color = msg['color']
                color = color[..., ::-1] / 255.
                cam_params = [msg['intrinsics'][0,0], msg['intrinsics'][1,1], msg['intrinsics'][0,2], msg['intrinsics'][1,2]]
                fgpcd = depth2fgpcd(depth, np.ones_like(depth).astype(bool), cam_params=cam_params, preserve_zero=True)
                curr_pcd = np2o3d(fgpcd, color=color.reshape(-1, 3))
                vis_pcd.points = curr_pcd.points
                vis_pcd.colors = curr_pcd.colors
                if iter_idx == 0:
                    visualizer.add_geometry(vis_pcd)
                visualizer.update_geometry(vis_pcd)
                visualizer.poll_events()
                visualizer.update_renderer()
                if iter_idx == 0:
                    visualizer.run()
                iter_idx += 1
                time.sleep(0.1)

def calibrate_single_extrinsics():
    with SingleFemto(
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        while True:
            # realsense.calibrate_extrinsics(visualize=True, board_size=(6,9), squareLength=0.03, markerLength=0.022)
            realsense.set_exposure(exposure=200, gain=16)
            realsense.calibrate_extrinsics(visualize=True, board_size=(6,5), squareLength=0.04, markerLength=0.03)
            time.sleep(0.1)

if __name__ == '__main__':
    # get_real_exporure_gain_white_balance()
    test_rgbd()

