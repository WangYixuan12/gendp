curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
chmod +x xsarm_amd64_install.sh
./xsarm_amd64_install.sh -d noetic
rm xsarm_amd64_install.sh
source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh
cd ~/interbotix_ws/src
git clone https://github.com/WangYixuan12/aloha.git # Yixuan's fork
cd ~/interbotix_ws
catkin_make
source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh
