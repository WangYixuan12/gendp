import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import json
import time
import numpy as np
from tqdm import tqdm
from multiprocessing.managers import SharedMemoryManager
from gild.real_world.multi_realsense import MultiRealsense, SingleRealsense
from gild.real_world.video_recorder import VideoRecorder

def test():
    serial_numbers = SingleRealsense.get_connected_devices_serial()
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    realsense =  MultiRealsense(
            serial_numbers=serial_numbers,
            shm_manager=shm_manager,
            resolution=(1280,720),
            capture_fps=30,
            record_fps=15,
            enable_color=True,
            enable_depth=True,
            verbose=True)
    
    # one thread per camera
    video_recorders = [VideoRecorder.create_h264(
        fps=30,
        codec='h264',
        input_pix_fmt='bgr24',
        thread_type='FRAME'
    ) for _ in range(len(serial_numbers))]
        
    realsense.start()
    realsense.set_exposure(exposure=None, gain=None)

    video_path = '/home/bing4090/yixuan_old_branch/general_dp/temp'
    os.system(f'mkdir -p {video_path}')
    rec_start_time = time.time() + 1
    realsense.restart_put(start_time=rec_start_time)
    print(realsense.is_ready)
    
    for i in range(len(serial_numbers)):
        video_recorders[i].start(file_path=f'{video_path}/{i}.mp4') # , start_time=rec_start_time)

    out = None
    vis_img = None
    with tqdm(total=20) as pbar:
        while True:
            out = realsense.get(out=out)
            for i in range(len(serial_numbers)):
                video_recorders[i].write_frame(out[i]['color'], frame_time=out[i]['timestamp'])

            time.sleep(1/60)
            if time.time() > (rec_start_time + 3.0):
                break
            pbar.update(1/60)

if __name__ == "__main__":
    test()
