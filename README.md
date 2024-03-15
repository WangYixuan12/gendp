# GILD: Generalizable Imitation Learning with 3D Semantic Fields

## Table of Contents
1. [Install](#install)
2. [Generate Dataset](#generate-dataset)
    1. [Generate from Existing Environments](#generate-from-existing-environments)
    2. [Generate from Customized Environments](#generate-from-customized-environments)
    3. [Generate Large-Scale Data](#generate-large-scale-data)
3. [Download Dataset](#download-dataset)
4. [Visualize Dataset](#visualize-dataset)
    1. [Visualize 2D Observation](#visualize-2d-observation)
    2. [Visualize Aggregated 3D Observation](#visualize-aggregated-3d-observation)
    3. [Visualize 3D Semantic Fields](#visualize-3d-semantic-fields)
5. [Train](#train)
    1. [Train GILD](#train-gild)
    2. [Config Explanation](#config-explanation)
6. [Infer in Simulator](#infer-in-simulator)
7. [Deploy in Real World](#deploy-in-real-world)
    1. [Set Up Robot](#set-up-robot)
    2. [Calibrate Camera and Robot Transformation](#calibrate-camera-and-robot-transformation)
    3. [Collect Demonstration](#collect-demonstration)
    4. [<span style="color:red">Train</span>](#train)
    5. [<span style="color:red">Infer in Real World</span>](#infer-in-real-world)
    6. [<span style="color:red">Adapt to New Task</span>](#adapt-to-new-task)

## TODO:
- [ ] Download checkpoints
- [ ] Test in the real world
- [ ] Go thru the whole README

## :hammer: Install
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
mamba env create -f conda_environment.yaml
pip install -e gild/
cd external
pip install -e sapien_env/
pip install -e robomimic/
pip install -e d3fields_dev/
```

## :floppy_disk: Generate Dataset

### Generate from Existing Environments
We use the [SAPIEN](https://sapien.ucsd.edu/docs/latest/index.html) to build the simulation environments. To create the data of heuristic policy for single episode, use the following command:
```console
python gen_single_episode.py [episode_idx] [dataset_dir] [task_name] --headless --obj_name [OBJ_NAME] --mode [MODE_NAME]
```
Meanings for each argument are visible when running `python gen_data.py --help`.

### Generate from Customized Environments
If you want to create your own environments with different objects, please imitate `sapien_env/sapien_env/sim_env/mug_collect_env.py`. Note that `sim_env/custom_env.py` does NOT contain the robot. To add robots, please imitate `sapien_env/sapien_env/rl_env/mug_collect_env.py` to add robots. To adjust camera views, please change `YX_TABLE_TOP_CAMERAS` within `sapien_env/sapien_env/gui/gui_base.py`.

### Generate Large-Scale Data
We notice that sapien renderer have memory leak for large-scale data generation. To avoid this, we use bash commands to generate large-scale data.
```console
python gen_multi_episodes.py
```
Arguments can be edited within `gen_multi_episodes.py`.

## :inbox_tray: Download Dataset
If you want to download a small dataset to test the whole pipeline, you can run `bash scripts/download_small_data.sh`. For hangning mug and pencil insertion task, you can run the following commands:
```console
bash scripts/download_hang_mug.sh
bash scripts/download_pencil_insertion.sh
```
If the scripts do not work, you could manully download the data from [UIUC Box](https://uofi.box.com/s/n5gahx98s14actc695tn3z0fzl8twcyk) or [Google Drive](https://drive.google.com/drive/folders/1_znHpzBj4c3fulXqt-0UjceRij2SApsH?usp=drive_link) and unzip them.

## :art: Visualize Dataset

### Visualize 2D Observation
To visualize image observations within hdf5 files, use the following command:
```console
python gild/tests/vis_data_2d.py 
```
You could adjust dataset path and observation keys in `gild/tests/vis_data_2d.py`.

### Visualize Aggregated 3D Observation
Similarly, to visualize aggegated 3D observations, use the following command:
```console
python gild/tests/vis_aggr_data_3d.py
```
This will visualize aggregated point clouds from multiple views, robot states, and actions from the dataset. You could adjust dataset path and observation keys in `gild/tests/vis_aggr_data_3d.py`.

### Visualize 3D Semantic Fields
Similarly, to visualize 3D semantic fields, use the following command:
```console
python gild/tests/vis_semantic_fields.py
```
This will visualize 3D semantic fields processed by [D3Fields](https://robopil.github.io/d3fields/), robot states, and actions. You could adjust dataset path and observation keys in `gild/tests/vis_semantic_fields.py`. The explanation of each entries within `shape_meta` can be seen at [Config Explanation](#config-explanation).

## :gear: Train

### Train GILD
To run training, we first set the environment variables.
```console
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export MKL_NUM_THREADS=1
```
Then, we run the following command:
```console
cd [PATH_TO_REPO]/gild
python train.py --config-dir=config/[TASK_NAME] --config-name=distilled_dino_N_4000.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=[PATH_TO_DATA]
```
For example, to train on `small_data` in my local machine, I run the following command:
```console
python train.py --config-dir=config/small_data --config-name=distilled_dino_N_4000.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=/home/yixuan/gild
```
Please wait at least till 2 epoches to make sure that all pipelines are working properly. For `hang_mug_sim` task and `pencil_insertion_sim` task, you could simply replace [TASK_NAME] with `hang_mug_sim` and `pencil_insertion_sim` respectively.

### Config Explanation
There are several critical entries within the config file. Here are some explanations:
```yaml
shape_meta: shape_meta contains the policy input and output information.
    action: output information
        shape: action dimension. In our work, it is 10 = (3 for translation, 6 for 6d rotation*, 1 for gripper)
        key: [optional] key for the action in the dataset. It could be 'eef_action' or 'joint_action'. Default is 'eef_action'.
    obs: input information
        ... # other input modalities if needed
        d3fields: 3D semantic fields
            shape: shape of the 3D semantic fields, i.e. (num_channel, num_points)
            type: type of inputs. It should be 'spatial' for point cloud inputs
            info: information of the 3D semantic fields.
                reference_frame: frame of input semantic fields. It should be 'world' or 'robot'
                distill_dino: whether to add semantic information to the point cloud
                distill_obj: the name for reference features, which are saved in `d3fields_dev/d3fields/sel_feats/[DISTILL_OBJ].npy`.
                view_keys: viewpoint keys for the semantic fields.
                N_gripper: number of points sampled from the gripper.
                boundaries: boundaries for the workspace.
                resize_ratio: our pipeline will resize images by this ratio to save time and memory.
task:
    env_runner: the configuration for the evaluation environment during the training
        max_steps: maximum steps for each episode, which should be adjusted according to the task
        n_test: number of testing environments
        n_test_vis: number of testing environments that will be visualized on wandb
        n_train: number of training environments
        n_train_vis: number of training environments that will be visualized on wandb
        train_obj_ls: list of objects that appear in the training environments
        test_obj_ls: list of objects that appear in the testing environments
training:
    checkpoint_every: the frequency of saving checkpoints
    rollout_every: the frequency of rolling out the policy in the env_runner
```

## :video_game: Infer in Simulator
To run an existing policy in the simulator, use the following command:
```console
cd [PATH_TO_REPO]/gild
python eval.py --checkpoint [PATH_TO_CHECKPOINT] --output_dir [OUTPUT_DIR] --n_test [NUM_TEST] --n_train [NUM_TRAIN] --n_test_vis [NUM_TEST_VIS] --n_train_vis [NUM_TRAIN_VIS] --test_obj_ls [OBJ_NAME_1] --test_obj_ls [OBJ_NAME_2] --data_root [PATH_TO_DATA]
```
To download the existing checkpoints, you could run the following commands.
```console
bash scripts/download_ckpts.sh
```
You can also download them from [UIUC Box](https://uofi.box.com) or [Google Drive](https://drive.google.com) and unzip them if the scipt fails.


## :robot: Deploy in Real World

### Hardware Prerequisites
- [Aloha](https://github.com/tonyzhaozh/aloha)
- \>=1 Realsense Camera

### Set Up Robot
1. If you already have ROS noetic installed, you could run `bash scripts/setup_aloha.sh`.
2. As mentioned in Aloha README, you need to go to ``~/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py``, find function ``publish_positions``. Change ``self.T_sb = mr.FKinSpace(self.robot_des.M, self.robot_des.Slist, self.joint_commands)`` to ``self.T_sb = None``. This prevents the code from calculating FK at every step which delays teleoperation.
3. We also need to update usb rules for the robot. You could run the following commands to update the usb rules.
```console
sudo bash scripts/modify_usb_rules.sh
sudo udevadm control --reload && sudo udevadm trigger
```
4. Remember to reboot the computer after the installation. If you encounter any problems, please refer to the [Aloha](https://github.com/tonyzhaozh/aloha).
5. To test whether the robot installation is successful, you could run the following command:
```console
# for boths sides
roslaunch aloha 4arms_teleop.launch
python gild/gild/real_world/aloha_simple_teleop.py --left --right

# for left side
roslaunch aloha 2arms_left_teleop.launch
python gild/gild/real_world/aloha_simple_teleop.py --left

# for right side
roslaunch aloha 2arms_right_teleop.launch
python gild/gild/real_world/aloha_simple_teleop.py --right
```

### Calibrate Camera and Robot Transformation
We found raw RealSense intrinsics are accurate enough for our pipeline, but you might want to verify it before proceeding.

First, we calibrate the extrinsics between the camera and the world (i.e. calibration board) frame. We use [calib.io](https://calib.io/pages/camera-calibration-pattern-generator) to generate the calibration board. Please use `ChArUco` as `Target Type`. You could select the rest of options according to your preference and printing capability. Then you can click `Save calibration board as PDF` to download and print the calibration board. Then you could run
```console
python gild/gild/real_world/calibrate_realsenses.py --rows [NUM_ROWS] --cols [NUM_COLS] --checker_width [CHECKER_WIDTH] --marker_width [MARKER_WIDTH]
```
This will keep running calibration pipeline in a `while True` loop and save the calibration results in `gild/gild/real_world/cam_extrinsics`. To visualize the calibration results, you could run
```console
python gild/gild/real_world/vis_cam_cali.py --iterative
```
Enabling `--iterative` will visualize each camera's point cloud iteratively and aggregated point cloud at the end. Otherwise, it will only visualize the aggregated point cloud. You are expected to see a well-aligned point cloud of the workspace.

Lastly, we calibrate the transformations between the robot base and the world frame, which is done manually. You could adjust `robot_base_in_world` within `gild/gild/real_world/calibrate_robot.py`, which represents the robots' base pose in the world (i.e. calibration board) frame. You could run
```console
python gild/gild/real_world/calibrate_robot.py
```
This will allow you to control robots and visualize the robot point cloud and the aggregated point cloud from cameras at the same time. You could adjust the robot base pose until the robot point cloud is well-aligned with the aggregated point cloud.

### Collect Demonstration
You could collect demonstrations by running the following command:
```console
python gild/demo_real_aloha.py --output_dir [OUTPUT_DIR] --robot_sides [ROBOT_SIDE] --robot_sides [ROBOT_SIDE] # [ROBOT_SIDE] could be 'left' or 'right'
```

### Train
### Infer in Real World
### Adapt to New Task
- label
- calibration
- bimanual
- modify config

## :pray: Acknowledgement
