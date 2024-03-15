# puppet robot left 
sudo echo 'SUBSYSTEM=="tty", ATTRS{serial}=="FT66WCAW", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_puppet_left"' >> /etc/udev/rules.d/99-interbotix-udev.rules

# puppet robot right 
sudo echo 'SUBSYSTEM=="tty", ATTRS{serial}=="FT66WB35", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_puppet_right"' >> /etc/udev/rules.d/99-interbotix-udev.rules

# master robot left
sudo echo 'SUBSYSTEM=="tty", ATTRS{serial}=="FT6Z5Q1I", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_master_left"' >> /etc/udev/rules.d/99-interbotix-udev.rules

# master robot right
sudo echo 'SUBSYSTEM=="tty", ATTRS{serial}=="FT6Z5MYV", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_master_right"' >> /etc/udev/rules.d/99-interbotix-udev.rules
