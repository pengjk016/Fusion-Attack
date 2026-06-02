import math
import numpy as np
import zmq  # 添加 ZMQ 用于跨进程发坐标

from collections import namedtuple
from panda3d.core import Vec3, Point2, Point3  # 添加 Point2, Point3 用于坐标换算
from multiprocessing.connection import Connection

from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation

from openpilot.common.realtime import Ratekeeper

from openpilot.tools.sim.lib.common import vec3
from openpilot.tools.sim.lib.camerad import W, H
from cereal import messaging
from metadrive.envs.base_env import BaseEnv

C3_POSITION = Vec3(0.0, 0, 1.22)
C3_HPR = Vec3(0, 0, 0)

metadrive_state = namedtuple("metadrive_state", ["velocity", "position", "bearing", "steering_angle"])


# 前车 3D 转 2D 像素框计算函数
def get_front_vehicle_pixel_data(env, camera_name="rgb_road", width=W, height=H):
  traffic_manager = env.engine.traffic_manager
  if not traffic_manager.vehicles:
    return None

  ego_vehicle = env.vehicle
  ego_pos = ego_vehicle.position
  ego_heading = ego_vehicle.heading_theta

  min_dist = float('inf')
  front_vehicle = None
  heading_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])

  for v in traffic_manager.vehicles:
    if v.id == ego_vehicle.id:
      continue
    rel_pos = v.position - ego_pos
    if np.dot(rel_pos, heading_vec) > 0:
      dist = np.linalg.norm(rel_pos)
      if dist < min_dist:
        min_dist = dist
        front_vehicle = v

  if front_vehicle is None:
    return None

  camera = env.engine.sensors[camera_name]
  cam_node = camera.get_cam()
  lens = cam_node.node().getLens()

  l, w, h = front_vehicle.LENGTH, front_vehicle.WIDTH, front_vehicle.HEIGHT

  # [核心修正] 移除人为的 1.15 倍增高，还原真实的车辆顶部物理高度，解决补丁悬空问题
  h_top = h
  corners_local = [
    Point3(w / 2, l / 2, h_top), Point3(-w / 2, l / 2, h_top),
    Point3(w / 2, -l / 2, h_top), Point3(-w / 2, -l / 2, h_top),
    Point3(w / 2, l / 2, 0), Point3(-w / 2, l / 2, 0),
    Point3(w / 2, -l / 2, 0), Point3(-w / 2, -l / 2, 0)
  ]

  u_min, u_max = float('inf'), -float('inf')
  v_min, v_max = float('inf'), -float('inf')

  for corner in corners_local:
    pt_cam = cam_node.getRelativePoint(front_vehicle.origin, corner)
    p_corner = Point2()
    if lens.project(pt_cam, p_corner):
      u = (p_corner[0] + 1) / 2 * width
      v = (1 - p_corner[1]) / 2 * height
      u_min, u_max = min(u_min, u), max(u_max, u)
      v_min, v_max = min(v_min, v), max(v_max, v)

  u_min, u_max = max(0, int(u_min)), min(width, int(u_max))
  v_min, v_max = max(0, int(v_min)), min(height, int(v_max))

  if u_min >= u_max or v_min >= v_max:
    return None

  return (u_min, v_min, u_max, v_max)


# ==========================================


def apply_metadrive_patches():
  # By default, metadrive won't try to use cuda images unless it's used as a sensor for vehicles, so patch that in
  def add_image_sensor_patched(self, name: str, cls, args):
    if self.global_config["image_on_cuda"]:  # and name == self.global_config["vehicle_config"]["image_source"]:
      sensor = cls(*args, self, cuda=True)
    else:
      sensor = cls(*args, self, cuda=False)
    assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
    self.sensors[name] = sensor

  EngineCore.add_image_sensor = add_image_sensor_patched

  # we aren't going to use the built-in observation stack, so disable it to save time
  def observe_patched(self, *args, **kwargs):
    return self.state

  ImageObservation.observe = observe_patched

  # disable destination, we want to loop forever
  def arrive_destination_patch(self, *args, **kwargs):
    return False

  MetaDriveEnv._is_arrive_destination = arrive_destination_patch


def metadrive_process(dual_camera: bool, config: dict, camera_array, wide_camera_array, image_lock,
                      controls_recv: Connection, state_send: Connection, exit_event):
  apply_metadrive_patches()

  road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
  if dual_camera:
    assert wide_camera_array is not None
    wide_road_image = np.frombuffer(wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

  env = MetaDriveEnv(config)

  def reset():
    env.reset()
    env.vehicle.config["max_speed_km_h"] = 1000

  reset()

  def get_cam_as_rgb(cam):
    cam = env.engine.sensors[cam]
    cam.get_cam().reparentTo(env.vehicle.origin)
    cam.get_cam().setPos(C3_POSITION)
    cam.get_cam().setHpr(C3_HPR)
    img = cam.perceive(clip=False)
    if type(img) != np.ndarray:
      img = img.get()  # convert cupy array to numpy
    return img

  rk = Ratekeeper(100, None)

  steer_ratio = 8
  vc = [0, 0]
  lidar = env.engine.get_sensor("lidar")

  pm = messaging.PubMaster(['surroundingInfo'])

  # ==========================================
  # 初始化 ZMQ 广播端
  # ==========================================
  zmq_context = zmq.Context()
  zmq_socket = zmq_context.socket(zmq.PUB)
  zmq_socket.bind("tcp://127.0.0.1:5555")  # 在本地 5555 端口广播
  # ==========================================

  while not exit_event.is_set():
    # === 状态信息 ===
    state = metadrive_state(
      velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
      position=env.vehicle.position,
      bearing=float(math.degrees(env.vehicle.heading_theta)),
      steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING
    )

    lidar_obs, detected_objects = lidar.perceive(
      env.vehicle,
      env.engine.physics_world.dynamic_world,
      num_lasers=lidar.Lidar_point_cloud_obs_dim,
      distance=255.0
    )

    surrounding_info = lidar.get_surrounding_vehicles_info(
      ego_vehicle=env.vehicle,
      detected_objects=detected_objects,
      perceive_distance=255.0,
      num_others=1,
      add_others_navi=False
    )

    state_send.send((state, surrounding_info))

    # === 控制逻辑 ===
    if controls_recv.poll(0):
      while controls_recv.poll(0):
        steer_angle, gas, should_reset = controls_recv.recv()

      steer_metadrive = steer_angle * 1 / (env.vehicle.MAX_STEERING * steer_ratio)
      steer_metadrive = np.clip(steer_metadrive, -1, 1)

      vc = [steer_metadrive, gas]

      if should_reset:
        reset()

    # === 环境 step ===
    if rk.frame % 5 == 0:
      obs, _, terminated, _, info = env.step(vc)

      if terminated:
        reset()

      if dual_camera:
        wide_road_image[...] = get_cam_as_rgb("rgb_wide")
      road_image[...] = get_cam_as_rgb("rgb_road")

      # ==========================================
      # 计算前车 2D 框并立刻通过 ZMQ 广播
      # ==========================================
      bbox = get_front_vehicle_pixel_data(env, "rgb_road", W, H)
      if bbox is not None:
        # 发送格式: "u_min,v_min,u_max,v_max"
        zmq_socket.send_string(f"{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}")
      else:
        zmq_socket.send_string("None")
      # ==========================================

      image_lock.release()

    rk.keep_time()
