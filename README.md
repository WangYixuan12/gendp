# general_dp
General Diffusion Policies - Yixuan Wang's Internship Project

## TODO
- [x] Installation
- [ ] Sim data generation
- [ ] Training
- [ ] Sim inference
- [ ] Data visualization
- [ ] Real inference

## Installation
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
mamba env create -f conda_environment.yaml
pip install -e gild/
pip install -e robomimic/
pip install -e d3fields_dev/
```

## Create dataset


## Download data
```console
mkdir -p data/sapien_env/teleop_data/pick_place_soda
cd data/sapien_env/teleop_data/pick_place_soda
gdown 1v264rhqXWqqJfYWcMHiD54OSfvvDH7ak
unzip small_rand_cola_demo_1.zip -d .
rm small_rand_cola_demo_1.zip
```

## Training
To run training, use the following command:
```console
# some env variables for the training to run
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export MKL_NUM_THREADS=1
cd [PATH_TO_REPO]/general_dp
python train.py --config-dir=config --config-name=sapien_pick_place_can_d3fields_test.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=[PATH_TO_REPO]
```
Please wait at least till 2 epoches to make sure that all pipelines are working properly.

## Real Inference
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
