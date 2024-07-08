#!/usr/bin/env python3
import os
import torch.multiprocessing as mp
import torch.multiprocessing
import sys
import cv2
import open3d as o3d
import time
import bisect
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import numpy as np
import torch
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

sys.path.append(os.path.dirname(__file__))
from argparse import ArgumentParser
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.graphics_utils import focal2fov
from scene.shared_objs import SharedCam, SharedGaussians, SharedPoints, SharedTargetPoints
from gaussian_renderer import render, network_gui
from mp_Tracker import Tracker
from mp_Mapper import Mapper

torch.multiprocessing.set_sharing_strategy('file_system')

"""
设置管道信息
"""
class Pipe():
    # 初始化传入三个参数：convert_SHs_python：将SHs转换为点云，compute_cov3D_python：计算3D协方差矩阵，debug：是否开启debug模式
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug

"""
GS ICP主要的高斯溅射类，继承自SLAMParameters
"""
class GS_ICP_SLAM(SLAMParameters):
    def __init__(self, args): # 初始化函数
        super().__init__()
        self.dataset_path = args.dataset_path # 数据集路径
        self.config = args.config # 相机参数路径
        self.output_path = args.output_path # 输出路径
        os.makedirs(self.output_path, exist_ok=True)# 创建输出路径
        self.verbose = args.verbose # 是否开启verbose模式
        self.keyframe_th = float(args.keyframe_th) # 关键帧阈值
        self.knn_max_distance = float(args.knn_maxd) # knn最大距离
        self.overlapped_th = float(args.overlapped_th) # 重叠阈值
        self.max_correspondence_distance = float(args.max_correspondence_distance)# 最大对应距离
        self.trackable_opacity_th = float(args.trackable_opacity_th)# 可跟踪的透明度阈值
        self.overlapped_th2 = float(args.overlapped_th2)# 重叠阈值2
        self.downsample_rate = int(args.downsample_rate) # 下采样率
        self.test = args.test # 测试
        self.save_results = args.save_results # 是否保存结果
        
        camera_parameters_file = open(self.config)# 打开相机参数文件
        camera_parameters_ = camera_parameters_file.readlines() # 读取相机参数文件
        self.camera_parameters = camera_parameters_[2].split() # 获取相机参数
        self.W = int(self.camera_parameters[0]) # 图像宽度
        self.H = int(self.camera_parameters[1]) # 图像高度
        # self.fx = float(self.camera_parameters[2]) # fx
        # self.fy = float(self.camera_parameters[3]) # fy
        # self.cx = float(self.camera_parameters[4]) # cx
        # self.cy = float(self.camera_parameters[5]) # cy
        self.depth_scale = float(self.camera_parameters[6]) # 深度缩放
        self.depth_trunc = float(self.camera_parameters[7]) # 深度截断
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate) # 下采样索引，x_pre，y_pre
        
        try:
            mp.set_start_method('spawn', force=True) # 设置启动方法, 强制使用spawn
        except RuntimeError:
            pass
        
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path) # 轨迹管理器,将路径下的轨迹文件读取出来，一个1*3的轨迹矩阵
        
        # Get size of final poses
        num_final_poses = len(self.trajmanager.gt_poses)# 获取轨迹的长度
        
        # Shared objects
        self.shared_cam = SharedCam(FoVx=focal2fov(self.fx, self.W), FoVy=focal2fov(self.fy, self.H),
                                    W = self.W, H=self.H)# 共享相机参数
        self.shared_new_points = SharedPoints(200000)# 共享新的点
        self.shared_new_gaussians = SharedGaussians(200000)# 共享新的高斯
        self.shared_target_gaussians = SharedTargetPoints(10000000)# 共享目标高斯
        self.end_of_dataset = torch.zeros((1)).int()# 数据集结束
        self.is_tracking_keyframe_shared = torch.zeros((1)).int()# 跟踪关键帧共享
        self.is_mapping_keyframe_shared = torch.zeros((1)).int()# 映射关键帧共享
        self.target_gaussians_ready = torch.zeros((1)).int()# 目标高斯准备
        self.new_points_ready = torch.zeros((1)).int()# 新的点准备
        self.final_pose = torch.zeros((num_final_poses,4,4)).float()# 最终姿态
        self.demo = torch.zeros((1)).int()# 演示
        self.is_mapping_process_started = torch.zeros((1)).int()# 映射进程开始
        
        self.shared_new_points.share_memory()
        self.shared_new_gaussians.share_memory()
        self.shared_target_gaussians.share_memory()
        self.end_of_dataset.share_memory_()
        self.is_tracking_keyframe_shared.share_memory_()
        self.is_mapping_keyframe_shared.share_memory_()
        self.target_gaussians_ready.share_memory_()
        self.new_points_ready.share_memory_()
        self.final_pose.share_memory_()
        self.demo.share_memory_()
        self.is_mapping_process_started.share_memory_()
        
        self.demo[0] = args.demo
        self.mapper = Mapper(self)#建图参数
        self.tracker = Tracker(self)#跟踪参数

        self.pointcloud_list = []
        self.pose_list = []
        self.process_synced_data  = []
        self.time_tolerance =0.1



    def process_pointcloud(self, points_timestamp, points_np):
         self.pointcloud_list.append((points_timestamp, points_np))
         self.sync_data()
        
    def process_pose(self, pose_timestamp, pose):
        self.pose_list.append((pose_timestamp, pose))
        self.sync_data()

    def sync_data(self):
        while self.pose_list and self.pointcloud_list:
            pose_timestamp, pose = self.pose_list[0]
            points_timestamp, points_np = self.pointcloud_list[0]
            
            if abs(pose_timestamp - points_timestamp) <= self.time_tolerance:
                self.pose_list.pop(0)
                self.pointcloud_list.pop(0)
                self.process_synced_data.append((pose_timestamp, pose, points_np))
            elif pose_timestamp < points_timestamp:
                self.pose_list.pop(0)
            else:
                self.pointcloud_list.pop(0)

    """
    跟踪线程执行
    """
    def tracking(self, process_synced_data):
        self.tracker.run(process_synced_data)
    
    """
    建图线程执行
    """
    def mapping(self, rank):
        self.mapper.run()

    def run(self):# 运行
        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(self.process_synced_data))# 跟踪
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=()) # 建图
            p.start()
            processes.append(p)# 添加进程
        for p in processes:
            p.join()

    def convert_point_np_to_points_and_colors(self, points_np):
        points = torch.from_numpy(points_np[:, :3])
        colors = torch.from_numpy(points_np[:, 3:])
        z_values = torch.from_numpy(points_np[:, 2])
        filter = torch.where((z_values!=0)&(z_values<=self.depth_trunc)) # 过滤条件
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()



def pose_callback(pose_msg):
    global gs_icp_slam
    # Convert PoseStamped to numpy array
    pose = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
    pose_timestamp = pose_msg.header.stamp.to_sec()
    gs_icp_slam.process_pose(pose_timestamp, pose)
    rospy.loginfo("Received new pose information.")

def pointcloud_callback(cloud_msg):
    global gs_icp_slam
    # Convert PointCloud2 to array
    points = pc2.read_points_list(cloud_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
    points_np = np.array(points, dtype=np.float32)
    points_timestamp = cloud_msg.header.stamp.to_sec()
    # Process pointcloud using the slam_system
    # You may need to modify the process method to accept point cloud data
    gs_icp_slam.process_pointcloud(points_timestamp,points_np)
    rospy.loginfo("Processed pointcloud data.")

if __name__ == '__main__':
    parser = ArgumentParser(description="dataset_path / output_path / verbose")
    parser.add_argument("--dataset_path", help="dataset path", default="dataset/Replica/room0")
    parser.add_argument("--config", help="caminfo", default="configs/Replica/caminfo.txt")
    parser.add_argument("--output_path", help="output path", default="output/room0")
    parser.add_argument("--keyframe_th", default=0.7)
    parser.add_argument("--knn_maxd", default=99999.0)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--demo", action='store_true', default=False)
    parser.add_argument("--overlapped_th", default=5e-4)
    parser.add_argument("--max_correspondence_distance", default=0.02)
    parser.add_argument("--trackable_opacity_th", default=0.05)
    parser.add_argument("--overlapped_th2", default=5e-5)
    parser.add_argument("--downsample_rate", default=10)
    parser.add_argument("--test", default=None)
    parser.add_argument("--save_results", action='store_true', default=None)
    args = parser.parse_args()

    gs_icp_slam = GS_ICP_SLAM(args)


    rospy.init_node('gs_icp_slam_node', anonymous=True)

    # Subscribers
    pose_subscriber = rospy.Subscriber('/pose_topic', PoseStamped, pose_callback)
    pointcloud_subscriber = rospy.Subscriber('/pointcloud_topic', PointCloud2, pointcloud_callback)

    rospy.spin()
