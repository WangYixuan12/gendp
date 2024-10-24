# GenDP: 3D Semantic Fields for Category-Level Generalizable Diffusion Policy [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Yk6uDg2So9A3yWALR7mF5dhl_KKR24eh?usp=sharing)

[Website](https://robopil.github.io/GenDP/) | [Paper](https://arxiv.org/abs/2410.17488) | [Colab](https://colab.research.google.com/drive/1Yk6uDg2So9A3yWALR7mF5dhl_KKR24eh?usp=sharing) | [Video](https://youtu.be/6jUGmUaAEOc)

<a target="_blank" href="https://wangyixuan12.github.io/">Yixuan Wang</a><sup>1</sup>,
<a target="_blank" href="https://robopil.github.io/GenDP/">Guang Yin</a><sup>2</sup>,
<a target="_blank" href="https://binghao-huang.github.io/">Binghao Huang</a><sup>1</sup>,
<a target="_blank" href="https://kelestemur.com/">Tarik Kelestemur</a><sup>3</sup>,
<a target="_blank" href="https://www.robo.guru/">Jiuguang Wang</a><sup>3</sup>,
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1</sup>
            
<sup>1</sup>Columbia University,
<sup>2</sup>University of Illinois Urbana-Champaign,
<sup>3</sup>Boston Dynamics AI Institute<br>


https://github.com/WangYixuan12/gendp/assets/32333199/f86c977b-bcce-45cf-b632-95663abf3607


## :bookmark_tabs: Table of Contents
- [Install](#hammer-install)
- [Generate Dataset](#floppy_disk-generate-dataset)
    - [Generate from Existing Environments](#generate-from-existing-environments)
    - [Generate from Customized Environments](#generate-from-customized-environments)
    - [Generate Large-Scale Data](#generate-large-scale-data)
- [Download Dataset](#inbox_tray-download-dataset)
- [Visualize Dataset](#art-visualize-dataset)
    - [Visualize 2D Observation](#visualize-2d-observation)
    - [Visualize Aggregated 3D Observation](#visualize-aggregated-3d-observation)
    - [Visualize 3D Semantic Fields](#visualize-3d-semantic-fields)
- [Train](#gear-train)
    - [Train in Simulation](#train-in-simulation)
    - [Config Explanation](#config-explanation)
- [Infer in Simulation](#video_game-infer-in-simulation)
- [Deploy in Real World](#robot-deploy-in-real-world)
    - [Hardware Prerequisites](#hardware-prerequisites)
    - [Install Environment for Real World](#install-environment-for-real-world)
    - [Set Up Robot](#set-up-robot)
    - [Calibrate Camera and Robot Transformation](#calibrate-camera-and-robot-transformation)
    - [Collect Demonstration](#collect-demonstration)
    - [Train in Real World](#train-in-real-world)
    - [Infer in Real World](#infer-in-real-world)
    - [Adapt to New Task](#adapt-to-new-task)

## :hammer: Install
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
mamba env create -f conda_environment.yaml
conda activate gendp
pip install -e gendp/
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
For example, to generate one episode for the `hang_mug` task with the GUI, you could run the following command:
```console
python gen_single_episode.py 0 data/ hang_mug --obj_name nescafe_mug # random seed is 0; save the data into data/; task name is hang_mug; object name is nescafe_mug
```
Meanings for each argument are visible when running `python gen_single_episode.py --help`.

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
If the scripts do not work, you could manully download the data from [UIUC Box](https://uofi.box.com/s/n5gahx98s14actc695tn3z0fzl8twcyk) or [Google Drive](https://drive.google.com/drive/folders/1_znHpzBj4c3fulXqt-0UjceRij2SApsH?usp=sharing) and unzip them.

## :art: Visualize Dataset

### Visualize 2D Observation
To visualize image observations within hdf5 files, use the following command:
```console
python gendp/tests/vis_data_2d.py 
```
You could adjust dataset path and observation keys in `gendp/tests/vis_data_2d.py`.

### Visualize Aggregated 3D Observation
Similarly, to visualize aggegated 3D observations, use the following command:
```console
python gendp/tests/vis_aggr_data_3d.py
```
This will visualize aggregated point clouds from multiple views, robot states, and actions from the dataset. You could adjust dataset path and observation keys in `gendp/tests/vis_aggr_data_3d.py`.

### Visualize 3D Semantic Fields
Similarly, to visualize 3D semantic fields, use the following command:
```console
python gendp/tests/vis_semantic_fields.py
```
This will visualize 3D semantic fields processed by [D3Fields](https://robopil.github.io/d3fields/), robot states, and actions. You could adjust dataset path and observation keys in `gendp/tests/vis_semantic_fields.py`. The explanation of each entries within `shape_meta` can be seen at [Config Explanation](#config-explanation).

## :gear: Train

### Train in Simulation
To run training, we first set the environment variables.
```console
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export MKL_NUM_THREADS=1
```
Then, we run the following command:
```console
cd [PATH_TO_REPO]/gendp
python train.py --config-dir=config/[TASK_NAME] --config-name=distilled_dino_N_4000.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=[PATH_TO_DATA]
```
For example, to train on `small_data` in my local machine, I run the following command:
```console
python train.py --config-dir=config/small_data --config-name=distilled_dino_N_4000.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=/home/yixuan/gendp
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
Also, the configuration might be repetitive in the config file. Please sync them manually.

## :video_game: Infer in Simulation
To run an existing policy in the simulator, use the following command:
```console
cd [PATH_TO_REPO]/gendp
python eval.py --checkpoint [PATH_TO_CHECKPOINT] --output_dir [OUTPUT_DIR] --n_test [NUM_TEST] --n_train [NUM_TRAIN] --n_test_vis [NUM_TEST_VIS] --n_train_vis [NUM_TRAIN_VIS] --test_obj_ls [OBJ_NAME_1] --test_obj_ls [OBJ_NAME_2] --data_root [PATH_TO_DATA]
```
For example, we can run
```console
python eval.py --checkpoint /home/yixuan/gendp/checkpoints/small_data/distilled_dino_N_4000/ckpt_00000000.pt --output_dir /home/yixuan/gendp/eval_results/small_data --n_test 10 --n_train 10 --n_test_vis 5 --n_train_vis 5 --test_obj_ls nescafe_mug --data_root /home/yixuan/gendp
```
To download the existing checkpoints, you could run the following commands.
```console
bash scripts/download_ckpts.sh
```
You can also download them from [UIUC Box](https://uofi.box.com/s/3hjv6obgxcn67abm7li2sa98npzry4d5) or [Google Drive](https://drive.google.com/drive/folders/1JRwLUXBUewRYNdY-dCp54CLAUiAdwrL1?usp=sharing) and unzip them if the scipt fails.


## :robot: Deploy in Real World

### Hardware Prerequisites
- [Aloha](https://github.com/tonyzhaozh/aloha)
- \>=1 Realsense Camera

### Install Environment for Real World
```console
mamba env create -f conda_environment_real.yaml
pip install -e gendp/
pip install -e d3fields_dev/
```

### Set Up Robot
1. If you already have ROS noetic installed, you could run `bash scripts/setup_aloha.sh` **outside** of conda environments. Remember to put `source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh` into `~/.bashrc` after installation.
2. As mentioned in Aloha README, you need to go to ``~/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py``, find function ``publish_positions``. Change ``self.T_sb = mr.FKinSpace(self.robot_des.M, self.robot_des.Slist, self.joint_commands)`` to ``self.T_sb = None``. This prevents the code from calculating FK at every step which delays teleoperation.
3. We also need to update usb rules for the robot. You could run the following commands to update the usb rules. You might need to change the serial numbers to your own.
```console
sudo bash scripts/modify_usb_rules.sh
sudo udevadm control --reload && sudo udevadm trigger
```
4. Remember to reboot the computer after the installation. If you encounter any problems, please refer to the [Aloha](https://github.com/tonyzhaozh/aloha).
5. To test whether the robot installation is successful, you could run the following command:
```console
# boths sides
roslaunch aloha 4arms_teleop.launch
python gendp/gendp/real_world/aloha_simple_teleop.py --left --right

# left side
roslaunch aloha 2arms_left_teleop.launch
python gendp/gendp/real_world/aloha_simple_teleop.py --left

# right side
roslaunch aloha 2arms_right_teleop.launch
python gendp/gendp/real_world/aloha_simple_teleop.py --right
```

### Calibrate Camera and Robot Transformation
We found raw RealSense intrinsics are accurate enough for our pipeline, but you might want to verify it before proceeding.

First, we calibrate the extrinsics between the camera and the world (i.e. calibration board) frame. We use [calib.io](https://calib.io/pages/camera-calibration-pattern-generator) to generate the calibration board. Please use `ChArUco` as `Target Type`. You could select the rest of options according to your preference and printing capability. Then you can click `Save calibration board as PDF` to download and print the calibration board. Then you could run
```console
python gendp/gendp/real_world/calibrate_realsenses.py --rows [NUM_ROWS] --cols [NUM_COLS] --checker_width [CHECKER_WIDTH] --marker_width [MARKER_WIDTH]
```
This will keep running calibration pipeline in a `while True` loop and save the calibration results in `gendp/gendp/real_world/cam_extrinsics`. To visualize the calibration results, you could run
```console
python gendp/gendp/real_world/vis_cam_cali.py --iterative
```
Enabling `--iterative` will visualize each camera's point cloud iteratively and aggregated point cloud at the end. Otherwise, it will only visualize the aggregated point cloud. You are expected to see a well-aligned point cloud of the workspace.

Lastly, we calibrate the transformations between the robot base and the world frame, which is done manually. You could adjust `robot_base_in_world` within `gendp/gendp/real_world/calibrate_robot.py`, which represents the robots' base pose in the world (i.e. calibration board) frame. You could run
```console
python gendp/gendp/real_world/calibrate_robot.py
```
This will allow you to control robots and visualize the robot point cloud and the aggregated point cloud from cameras at the same time. You could adjust the robot base pose until the robot point cloud is well-aligned with the aggregated point cloud.

### Collect Demonstration
You could collect demonstrations by running the following command:
```console
python gendp/demo_real_aloha.py --output_dir [OUTPUT_DIR] --robot_sides [ROBOT_SIDE] --robot_sides [ROBOT_SIDE] # [ROBOT_SIDE] could be 'left' or 'right'
```
Press "C" to start recording. Use SpaceMouse to move the robot. Press "S" to stop recording. 

### Train in Real World
The traning is similar to the training in the simulator. Here are two examples:
```console
bash scripts/download_real_data.sh # download the real data
python train.py --config-dir=config/knife_real --config-name=distilled_dino_N_1000.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=/home/yixuan/gendp # train the model for pick_up_knife task
python train.py --config-dir=config/pen_real --config-name=distilled_dino_N_1000.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=/home/yixuan/gendp # train the model for pick_up_pen task
```

### Infer in Real World
Given a checkpoint, you could run the following command to infer in the real world (absolute path is recommended):
```console
python gendp/eval_real_aloha.py -i [PATH_TO_CKPT_FILE] -o [OUTPUT_DIR] -r [ROBOT_SIDE] --vis_d3fields [true OR false]
```
Press "C" to start evaluation (handing control over to the policy). Press "S" to stop the current episode.

### Adapt to New Task
To adapt our framework to new tasks, you could follow the following steps:
1. You can select reference DINO features by running `python d3fields_dev/d3fields/scripts/sel_features.py`. This will provide an interactive interface to select the reference features given four arbitrary images. Click left mouse button to select the reference features and 'N' to next image. Click `Q` to quit and save the selected features.
2. For the new task, you may need to update several important configuration entries.
```console
shape_meta:
    action:
        shape: 10 if using single robot and 20 for bimanual manipulation
    obs:
        d3fields:
            shape: change the first number (number of channel). It is 3 if only using raw point cloud. It is 3 + number of reference features if using DINOv2 features.
            info:
                distill_dino: whether to add semantic information to the point cloud
                distill_obj: the name for reference features, which are saved in `d3fields_dev/d3fields/sel_feats/[DISTILL_OBJ].npy`.
                bounding_box: the bounding box for the workspace
task_name: name for tasks, which will be used in wandb and logging files
dataset_name: the name for the training dataset, which will be used to infer dataset_dir (e.g. ${data_root}/data/real_aloha_demo/${dataset_name} or  ${data_root}/data/sapien_demo/${dataset_name})
```

## :pray: Acknowledgement

This repository is built upon the following repositories. Thanks for their great work!
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [robomimic](https://github.com/ARISE-Initiative/robomimic)
- [D3Fields](https://github.com/WangYixuan12/d3fields)
