# general_dp
General Diffusion Policies - Yixuan Wang's Internship Project

# Table of Contents
1. [Install](#install)
2. [Generate Dataset](#generate-dataset)
    1. [Generate from Existing Environments](#generate-from-existing-environments)
    2. [Generate from Customized Environments](#generate-from-customized-environments)
3. [Download Data](#download-data)
4. [Train](#train)
    1. [Visualize Semantic Fields](#visualize-semantic-fields)
    2. [Train GILD](#train-gild)
    3. [Config Explanation](#config-explanation)
    4. [Adapt to New Task](#adapt-to-new-task)
5. [Infer in Simulator](#infer-in-simulator)
6. [Infer in Real World](#infer-in-real-world)

## TODO
- [ ] Training
- [ ] Sim inference
- [ ] Data visualization
- [ ] Select your DINOv2 features
- [ ] Real inference
- [ ] Clean up robomimic database

## Install
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
mamba env create -f conda_environment.yaml
pip install -e gild/
pip install -e robomimic/
pip install -e d3fields_dev/
```

## Generate Dataset

### Generate from Existing Environments
We use the [SAPIEN](https://sapien.ucsd.edu/docs/latest/index.html) to build the simulation environments. To create the data of heuristic policy for single episode, use the following command:
```console
python gen_data.py [episode_idx] [dataset_dir] [task_name] --headless --obj_name [OBJ_NAME] --mode []
```
Meanings for each argument are visible when running `python gen_data.py --help`.

### Generate from Customized Environments
If you want to create your own environments with different objects, please imitate `sapien_env/sapien_env/sim_env/mug_collect_env.py`. Note that `sim_env/custom_env.py` does NOT contain the robot. To add robots, please imitate `sapien_env/sapien_env/rl_env/mug_collect_env.py` to add robots. To adjust camera views, please change `YX_TABLE_TOP_CAMERAS` within `sapien_env/sapien_env/gui/gui_base.py`.

## Download Data
If you want to download a small dataset to test the whole pipeline, you can run `./scripts/download_small_data.sh`. For hangning mug and pencil insertion task, you can run the following commands:
```console
./scripts/download_hang_mug.sh
./scripts/download_pencil_insertion.sh
```
If the scripts do not work, you could manully download the data from [UIUC Box](https://uofi.box.com/s/n5gahx98s14actc695tn3z0fzl8twcyk) or [Google Drive](https://drive.google.com/drive/folders/1_znHpzBj4c3fulXqt-0UjceRij2SApsH?usp=drive_link) and unzip them.

## Train

### Visualize Semantic Fields
### Train GILD
To run training, use the following command:
```console
# some env variables for the training to run
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export MKL_NUM_THREADS=1
cd [PATH_TO_REPO]/gild
python train.py --config-dir=config --config-name=sapien_pick_place_can_d3fields_test.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=[PATH_TO_DATA]
```
Please wait at least till 2 epoches to make sure that all pipelines are working properly.

### Config Explanation
### Adapt to New Task

## Infer in Simulator

## Infer in Real World
### Install Interbotix
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


### Edit USB rules
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
