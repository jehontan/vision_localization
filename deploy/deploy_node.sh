wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
chmod +x install_pivariety_pkgs.sh
./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps
./install_pivariety_pkgs.sh -p imx519_kernel_driver

sudo apt install -y python2
sudo apt install -y raspberrypi-kernel-headers
sudo apt install -y gstreamer1.0-tools
mkdir /home/pi/$(uname -r)

# download kernel source code
cd ~
sudo apt install git bc bison flex libssl-dev
sudo wget https://raw.githubusercontent.com/RPi-Distro/rpi-source/master/rpi-source -O /usr/local/bin/rpi-source && sudo chmod +x /usr/local/bin/rpi-source && /usr/local/bin/rpi-source -q --tag-update
rpi-source -d $(uname -r)

# compile driver
cd ~
git clone --branch v0.12.5 https://github.com/umlaeute/v4l2loopback.git
cd v4l2loopback
make clean && make
make && sudo make install
sudo depmod -a

# use v4l2loopback
sudo modprobe v4l2loopback video_nr=3
gst-launch-1.0 libcamerasrc ! 'video/x-raw,width=1920,height=1080' ! videoconvert ! tee ! v4l2sink device=/dev/video3

# install
sudo apt install python3-opencv
git clone https://github.com/jehontan/vision_localization.git
echo 'export PATH=$PATH:${HOME}/.local/bin' >> ~/.bashrc 

# autofocus
libcamera-still -t 0 --keypress