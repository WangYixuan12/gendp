curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
chmod +x xsarm_amd64_install.sh
./xsarm_amd64_install.sh -d noetic
source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh
cd ~/interbotix_ws/src
git clone git@github.com:WangYixuan12/aloha.git # Yixuan's fork
cd ~/interbotix_ws
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
