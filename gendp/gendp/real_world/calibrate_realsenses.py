import time
import argparse

from gendp.real_world.multi_realsense import MultiRealsense

def calibrate_all(rows, cols, checker_width, marker_width):
    with MultiRealsense(
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        while True:
            realsense.set_exposure(exposure=200, gain=16)
            realsense.calibrate_extrinsics(visualize=True, board_size=(cols,rows), squareLength=checker_width, markerLength=marker_width)
            time.sleep(0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=5)
    parser.add_argument('--cols', type=int, default=6)
    parser.add_argument('--checker_width', type=float, default=0.04)
    parser.add_argument('--marker_width', type=float, default=0.03)
    
    args = parser.parse_args()
    calibrate_all(args.rows, args.cols, args.checker_width, args.marker_width)
