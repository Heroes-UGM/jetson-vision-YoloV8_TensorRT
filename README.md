# jetson-vision-YoloV8_TensorRT
Inference: https://github.com/triple-Mu/YOLOv8-TensorRT

Alur convert model:  PyTorch(trained yolov8 model) -> ONNX -> TensorRT Engine

## Tahapan install dependencies dari bare image
### Aktifin CUDA
Masukin kode dibawah ke ~/.bashrc
```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$ 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Cek di terminal
```
nvcc --version
```
Kalau muncul berarti udah oke
### Install PIP
```
sudo apt install python3-pip
```
### Install TensorRT untuk binding ke python
file wheel tensorrt 8.2.1.8 ada di repo, didownload aja terus install pake
```
sudo -H pip3 install tensorrt-8.2.1.8-cp38-none-linux_aarch64.whl
```
### Install GCC
https://linuxize.com/post/how-to-install-gcc-on-ubuntu-20-04/#google_vignette
```
sudo apt install gcc-8 g++-8 gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 --slave /usr/bin/g++ g++ /usr/bin/g++-8 --slave /usr/bin/gcov gcov /usr/bin/gcov-8
sudo update-alternatives --config gcc
```
Pastiin gcc yang kepake versi 8
### Install Protobuf
Versi protobuf harus <= 3.20.1
```
pip3 install protobuf==3.20.*
```
### Install ONNX
```
pip3 install onnx==1.11.0
```
untuk onnxruntime versi 1.11.0, download di (https://elinux.org/Jetson_Zoo#ONNX_Runtime) (wheel installation)
### Install ultralytics
```
pip3 install ultralytics --no-dependencies
```
### Install Jetson Inference
```
git clone https://github.com/dusty-nv/jetson-inference
cd jetson-inference
git submodule update --init
mkdir build
cd build
cmake ..
make -j8
sudo make install
sudo ldconfig
```
### Install PyTorch dan TorchVision
https://qengineering.eu/install-pytorch-on-jetson-nano.html
### Install cmake
versi cmake yang dipake 3.22.4
```
wget https://github.com/Kitware/CMake/releases/download/v3.22.4/cmake-3.22.4-linux-aarch64.tar.gz -q --show-progress 
tar -zxvf cmake-3.22.4-linux-aarch64.tar.gz 
cd cmake-3.22.4-linux-aarch64/
sudo cp -rf bin/ doc/ share/ /usr/local/
sudo cp -rf man/* /usr/local/man
sync
cmake --version 
```
### Install onnxsim
```
pip3 install onnxsim
```

## Instal YOLOv8-TensorRT
GitHub YOLOv8-TensorRT: https://github.com/triple-Mu/YOLOv8-TensorRT

Versi ONNX yang dipake: 1.11.0

Versi ONNXRuntime-gpu yang dipake: 1.11.0 (https://elinux.org/Jetson_Zoo#ONNX_Runtime)

Versi CUDA: 10.2.300

Versi TensorRT: 8.2.1.8

## Cara Convert Model
Bikin dulu model yang dah di train (pytorch model .pt)

Kemudian convert .pt ke .onnx pake colab
https://colab.research.google.com/drive/1TkLnD-2UbtfW2o76C_wV4A64Mx1V6J1D?usp=sharing

Setelah model dikonvert jadi .onnx, masukin ke jetson buat konvert .onnx ke .engine:
```
/usr/src/tensorrt/bin/trtexec --onnx=<NAMA MODEL>.onnx --saveEngine=<NAMA MODEL>.engine --fp16
```

selesai deh, buat ngecek modelnya bisa apa ga:
```
/usr/src/tensorrt/bin/trtexec --loadEngine=<NAMA MODEL>.engine
```

## Cara ngetest inference pake webcam

### Dengan library jetson-inference (paling efisien)
https://github.com/dusty-nv/jetson-inference
Ada problem buat display ke OpenGL, tambahin kode ini ke ./bashrc
```
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
Display dari lib ini gabisa dipake kalo gacolok hdmi.
```
python3 inf-jeutils.py
```

### Dengan library nanocamera
```
python3 inf-nanocamera.py
```
### Dengan library opencv
```
python3 inf-opencv.py
```
