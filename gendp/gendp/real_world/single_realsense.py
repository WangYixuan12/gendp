from typing import Optional, Callable, Dict
import os
import enum
import time
import warnings
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from gendp.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from gendp.shared_memory.shared_ndarray import SharedNDArray
from gendp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from gendp.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from gendp.real_world.video_recorder import VideoRecorder

class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4

class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1280,720),
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
            'option_enum': rs.option.exposure.value,
            'option_value': 0.0,
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
                shape=(5,),
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
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == 'D400':
                    # only works with D400 series
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
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })
    
    def set_depth_option(self, option: rs.option, value: float):
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
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
    
    def set_depth_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_depth_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_depth_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_depth_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_depth_option(rs.option.gain, gain)
    
    def set_depth_preset(self, preset: str):
        visual_preset = {
            'Custom': 0,
            'Default': 1,
            'Hand': 2,
            'High Accuracy': 3,
            'High Density': 4,
        }
        self.set_depth_option(rs.option.visual_preset, visual_preset[preset])
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

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
    
    def _init_depth_process(self):
        # Initialize the processing steps
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        # self.spatial.set_option(rs.option.holes_fill, 1)
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 1)
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        # Initialize the alignment to make the depth data aligned to the rgb camera coordinate
        self.align = rs.align(rs.stream.color)

    def project_point_to_pixel(self, points):
        # The points here should be in the camera coordinate, n*3
        points = np.array(points)
        pixels = []
        # # Use the realsense projeciton, however, it's slow for the loop; this can give nan for invalid points
        # for i in range(len(points)):
        #     pixels.append(rs.rs2_project_point_to_pixel(self.intrinsic, points))
        # pixels = np.array(pixels)

        # Use the opencv projection
        # The width and height are inversed here
        pixels = cv2.projectPoints(
            points,
            np.zeros(3),
            np.zeros(3),
            self.intrinsic_matrix,
            self.dist_coef,
        )[0][:, 0, :]

        return pixels[:, ::-1]

    def deproject_pixel_to_point(self, pixel_depth):
        # pixel_depth contains [i, j, depth[i, j]]
        points = []
        for i in range(len(pixel_depth)):
            # The width and height are inversed here
            points.append(
                rs.rs2_deproject_pixel_to_point(
                    self.intrinsic,
                    [pixel_depth[i, 1], pixel_depth[i, 0]],
                    pixel_depth[i, 2],
                )
            )
        return np.array(points)

    def _process_depth(self, depth_frame):
        # Depth process
        filtered_depth = self.depth_to_disparity.process(depth_frame)
        filtered_depth = self.spatial.process(filtered_depth)
        filtered_depth = self.temporal.process(filtered_depth)
        filtered_depth = self.disparity_to_depth.process(filtered_depth)
        return filtered_depth
     
    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        # cv2.setNumThreads(1)
        w, h = self.resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color, 
                w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth, 
                w, h, rs.format.z16, fps)
            self._init_depth_process()
        if self.enable_infrared:
            rs_config.enable_stream(rs.stream.infrared,
                w, h, rs.format.y8, fps)
        
        try:
            rs_config.enable_device(self.serial_number)

            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            d = pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)

            # setup advanced mode
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # get
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor = pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale
            
            self.dist_coeff_array.get()[:] = np.array(intr.coeffs)
            
            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f'[SingleRealsense {self.serial_number}] Main loop started.')

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            for _ in range(30):
                pipeline.wait_for_frames()

            while not self.stop_event.is_set():
                wait_start_time = time.time()
                # wait for frames to come in
                try:
                    frameset = pipeline.wait_for_frames(timeout_ms=int(1000 * 1/fps)+ 10)
                except RuntimeError:
                    warnings.warn('RuntimeError in wait_for_frames, skipping frame.')
                    continue
                receive_time = time.time()
                # align frames to color
                frameset = align.process(frameset)
                wait_time = time.time() - wait_start_time

                # grab data
                grab_start_time = time.time()
                data = dict()
                data['camera_receive_timestamp'] = receive_time
                # realsense report in ms
                data['camera_capture_timestamp'] = frameset.get_timestamp() / 1000
                if self.enable_color:
                    color_frame = frameset.get_color_frame()
                    data['color'] = np.asarray(color_frame.get_data())
                    t = color_frame.get_timestamp() / 1000
                    data['camera_capture_timestamp'] = t
                    # print('device', time.time() - t)
                    # print(color_frame.get_frame_timestamp_domain())
                if self.enable_depth:
                    depth_frame = frameset.get_depth_frame()
                    proc_depth_frame = self._process_depth(depth_frame)
                    data['depth'] = np.asarray(proc_depth_frame.get_data())
                    # data['depth'] = np.asarray(
                    #     frameset.get_depth_frame().get_data())
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
                    print(f'[SingleRealsense {self.serial_number}] Grab data time {grab_time}')
                
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
                    print(f'[SingleRealsense {self.serial_number}] Transform time {transform_time}')

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
                #     print(f'[SingleRealsense {self.serial_number}] Vis time {vis_time}')
                
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
                    print(f'[SingleRealsense {self.serial_number}] Record time {rec_time}')

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
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                        # print('auto', sensor.get_option(rs.option.enable_auto_exposure))
                        # print('exposure', sensor.get_option(rs.option.exposure))
                        # print('gain', sensor.get_option(rs.option.gain))
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
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
                    print(f'[SingleRealsense {self.serial_number}] Command time {cmd_time}')

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
                    print(f'[SingleRealsense {self.serial_number}] FPS {frequency}')
        finally:
            self.video_recorder.stop()
            rs_config.disable_all_streams()
            self.ready_event.set()
        
        if self.verbose:
            print(f'[SingleRealsense {self.serial_number}] Exiting worker process.')

def get_real_exporure_gain_white_balance():
    series_number = SingleRealsense.get_connected_devices_serial()
    with SharedMemoryManager() as shm_manager:
        with SingleRealsense(
            shm_manager=shm_manager,
            serial_number=series_number[2],
            enable_color=True,
            enable_depth=True,
            enable_infrared=False,
            put_fps=30,
            record_fps=30,
            verbose=True,
        ) as realsense:
            # realsense.set_exposure()
            # realsense.set_white_balance()
            realsense.set_exposure(200, 64)
            realsense.set_white_balance(2800)

            for i in range(30):
                realsense.get()
                time.sleep(0.1)
            
            cv2.imshow('color', realsense.get()['color'])
            cv2.waitKey(0)

if __name__ == '__main__':
    get_real_exporure_gain_white_balance()

