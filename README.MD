[comment]: <> (# RGBD GS-ICP SLAM)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> RGBD GS-ICP SLAM (中文注释版)
  </h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://arxiv.org/pdf/2403.12550.pdf">Paper</a> 
  | <a href="https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM">Original Github Page</a>
  | <a href="https://blog.csdn.net/gwplovekimi/article/details/137587778?spm=1001.2014.3001.5501">CSDN 配置教程</a>
  </h3>
  <div align="center"></div>

<div align=center>

![github (1)](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM/assets/34827206/5722e8f4-165d-4093-8064-a7ed5d9ea008)

</div>

<br>


## Environments
Install requirements
```bash
conda create -n gsicpslam python==3.9
conda activate gsicpslam
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# 安装pcl
pip install pcl
conda config --add channels conda-forge
conda install -c sirokujira python-pcl  
```
Also, PCL is needed for fast-gicp submodule.

Install submodules

```bash
 cd submodules/
 
git clone https://github.com/Lab-of-AI-and-Robotics/diff-gaussian-rasterization.git
 
git clone https://github.com/Lab-of-AI-and-Robotics/fast_gicp.git
 
git clone https://github.com/camenduru/simple-knn.git


# 然后执行下面代码
conda activate gsicpslam
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

cd submodules/fast_gicp
mkdir build
cd build
cmake ..
make
cd ..
python setup.py install --user
```


## Datasets

- Replica
  - Download
    ```bash
    bash download_replica.sh
    ```
  - Configure
  
    Please modify the directory structure to ours.

    The original structure
    ```bash
    Replica
        - room0
            - results (contain rgbd images)
                - frame000000.jpg
                - depth000000.jpg
                ...
            - traj.txt
        ...
    ```
    Our structure
    ```bash
    Replica
        - room0
            - images (contain rgb images)
                - frame000000.jpg
                ...
            - depth_images (contain depth images)
                - depth000000.jpg
                ...
            - traj.txt
        ...
    ```    

- TUM-RGBD
  - Download
    ```bash
        bash download_tum.sh
    ```

## Run
- Limited to 30 FPS
    ```bash
        # Replica
        bash replica.sh
        # TUM
        bash tum.sh
    ```

- Unlimited tracking speed
    ```bash
        # Replica
        bash replica_unlimit.sh
        # TUM
        bash tum_unlimit.sh
    ```

## Installing SIBR Viewer
```bash
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
```

## Real-time demo
### Using rerun.io viewer

Rerun viewer shows the means of trackable Gaussians, and rendered image from reconstructed 3dgs map.

![GIFMaker_me](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM/assets/34827206/b4715071-2e4a-4d17-b7a2-612bbd32dbd0)

```bash
python -W ignore gs_icp_slam.py --rerun_viewer
```


### Using SIBR viewer
```bash
python -W ignore gs_icp_slam.py --dataset_path dataset/Replica/office0 --verbose

# In other terminal
cd SIBR_viewers
./install/bin/SIBR_remoteGaussian_app --rendering-size 1280 720
```

## Docker
Please see the README.md in the docker_files folder.
