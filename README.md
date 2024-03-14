# general_dp
General Diffusion Policies - Yixuan Wang's Internship Project

# Table of Contents
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
    2. [<span style="color:red">Config Explanation</span>](#config-explanation)
6. [<span style="color:red">Infer in Simulator</span>](#infer-in-simulator)
7. [<span style="color:red">Deploy in Real World</span>](#deploy-in-real-world)
    1. [<span style="color:red">Set Up Robot</span>](#set-up-robot)
    2. [<span style="color:red">Collect Demonstration</span>](#collect-demonstration)
    3. [<span style="color:red">Train</span>](#train)
    4. [<span style="color:red">Infer in Real World</span>](#infer-in-real-world)
    3. [<span style="color:red">Adapt to New Task</span>](#adapt-to-new-task)

## TODO
- [ ] Train
- [ ] Sim inference
- [ ] Real deployment
- [ ] Clean up robomimic database

## Install
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
mamba env create -f conda_environment.yaml
pip install -e gild/
cd external
pip install -e sapien_env/
pip install -e robomimic/
pip install -e d3fields_dev/
```

## Generate Dataset

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

## Download Dataset
If you want to download a small dataset to test the whole pipeline, you can run `bash scripts/download_small_data.sh`. For hangning mug and pencil insertion task, you can run the following commands:
```console
bash scripts/download_hang_mug.sh
bash scripts/download_pencil_insertion.sh
```
If the scripts do not work, you could manully download the data from [UIUC Box](https://uofi.box.com/s/n5gahx98s14actc695tn3z0fzl8twcyk) or [Google Drive](https://drive.google.com/drive/folders/1_znHpzBj4c3fulXqt-0UjceRij2SApsH?usp=drive_link) and unzip them.

## Visualize Dataset

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

## Train

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
Please wait at least till 2 epoches to make sure that all pipelines are working properly.

### Config Explanation
- n_test / n_train
- _every
- max_steps

## Infer in Simulator

## Deploy in Real World
### Set Up Robot
### Collect Demonstration
### Train
### Infer in Real World
### Adapt to New Task
- label
- calibration
- bimanual
- modify config
```console
curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
chmod +x xsarm_amd64_install.sh
./xsarm_amd64_install.sh -d noetic
​source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh
cd ~/interbotix_ws/src
​git clone git@github.com:tonyzhaozh/aloha.git
​cd ~/interbotix_ws
catkin_make
```
1. Go to ``~/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py``, find function ``publish_positions``. Change ``self.T_sb = mr.FKinSpace(self.robot_des.M, self.robot_des.Slist, self.joint_commands)`` to ``self.T_sb = None``. This prevents the code from calculating FK at every step which delays teleoperation.
2. Remember to reboot the computer after the installation.


```console
sudo vim /etc/udev/rules.d/99-interbotix-udev.rules
```

Add following lines:

```
# puppet robot left 
SUBSYSTEM=="tty", ATTRS{serial}=="FT66WCAW", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_puppet_left"

# puppet robot right 
SUBSYSTEM=="tty", ATTRS{serial}=="FT66WB35", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_puppet_right"

# master robot left
SUBSYSTEM=="tty", ATTRS{serial}=="FT6Z5Q1I", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_master_left"

# master robot right
SUBSYSTEM=="tty", ATTRS{serial}=="FT6Z5MYV", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_master_right"
```
​
Reload usb dev:
​`sudo udevadm control --reload && sudo udevadm trigger`
