import math
import numpy as np

from collections import namedtuple
from panda3d.core import Vec3
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
C3_HPR = Vec3(0, 0,0)

metadrive_state = namedtuple("metadrive_state", ["velocity", "position", "bearing", "steering_angle"])

def apply_metadrive_patches():
  # By default, metadrive won't try to use cuda images unless it's used as a sensor for vehicles, so patch that in
  def add_image_sensor_patched(self, name: str, cls, args):
    if self.global_config["image_on_cuda"]:# and name == self.global_config["vehicle_config"]["image_source"]:
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
  # env2 = BaseEnv(dict(log_level=50))

  def reset():
    env.reset()
    env.vehicle.config["max_speed_km_h"] = 1000

  reset()

  def get_cam_as_rgb(cam):
    # print(env.engine.sensors.items())
    cam = env.engine.sensors[cam]
    cam.get_cam().reparentTo(env.vehicle.origin)
    cam.get_cam().setPos(C3_POSITION)
    cam.get_cam().setHpr(C3_HPR)
    img = cam.perceive(clip=False)
    if type(img) != np.ndarray:
      img = img.get() # convert cupy array to numpy
    return img

  rk = Ratekeeper(100, None)

  steer_ratio = 8
  vc = [0,0]
  lidar = env.engine.get_sensor("lidar")

  pm = messaging.PubMaster(['surroundingInfo'])
  # lidar = env2.engine.get_sensor("lidar"),和  # env2 = BaseEnv(dict(log_level=50))一起
  while not exit_event.is_set():
    # === 状态信息 ===
    state = metadrive_state(
      velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
      position=env.vehicle.position,
      bearing=float(math.degrees(env.vehicle.heading_theta)),
      steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING
    )
    #print(f'自车速度(m/s,相对于世界坐标),距离(相对于世界坐标系）:{env.vehicle.velocity,env.vehicle.position}')#打印的是秒速
    # print(f'env.vehicles:{env.vehicles}')
    # # === 周围车辆信息 (利用 Lidar) ===
    # # 参数：车，物理世界，激光线数，探测距离
    lidar_obs, detected_objects = lidar.perceive(
      env.vehicle,
      env.engine.physics_world.dynamic_world,
      num_lasers=lidar.Lidar_point_cloud_obs_dim,
      distance=255.0
    )

    # 得到周围 num_others 辆车的信息（比如 5 辆），不需要导航信息
    surrounding_info = lidar.get_surrounding_vehicles_info(
      ego_vehicle=env.vehicle,
      detected_objects=detected_objects,
      perceive_distance=255.0,
      num_others=1,#周围num_others个车辆的信息
      add_others_navi=False
    )
    # print(f'surrounding_info:{surrounding_info}')
    # 你可以选择把车辆状态和周围车辆信息一起发送，因为radar是在simulated_car里面实现，所以以后可能用上发送代码
    state_send.send((state, surrounding_info))
    # state_send.send(state)


    # pm.send('surroundingInfo', surrounding_info)


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
      image_lock.release()

    rk.keep_time()


#
# def metadrive_process(dual_camera: bool, config: dict, camera_array, wide_camera_array, image_lock,
#                       controls_recv: Connection, state_send: Connection, exit_event):
#   apply_metadrive_patches()
#
#   road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
#   if dual_camera:
#     assert wide_camera_array is not None
#     wide_road_image = np.frombuffer(wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
#
#   env = MetaDriveEnv(config)
#
#   def reset():
#     env.reset()
#     env.vehicle.config["max_speed_km_h"] = 1000
#
#   reset()
#
#   def get_cam_as_rgb(cam):
#     cam = env.engine.sensors[cam]
#     print(f'sensors have:{cam}')
#     cam.get_cam().reparentTo(env.vehicle.origin)
#     cam.get_cam().setPos(C3_POSITION)
#     cam.get_cam().setHpr(C3_HPR)
#     img = cam.perceive(clip=False)
#     if type(img) != np.ndarray:
#       img = img.get() # convert cupy array to numpy
#     return img
#
#   rk = Ratekeeper(100, None)
#
#   steer_ratio = 8
#   vc = [0,0]
#
#   while not exit_event.is_set():
#     state = metadrive_state(
#       velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
#       position=env.vehicle.position,
#       bearing=float(math.degrees(env.vehicle.heading_theta)),
#       steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING
#     )
#
#     state_send.send(state)
#
#     if controls_recv.poll(0):
#       while controls_recv.poll(0):
#         steer_angle, gas, should_reset = controls_recv.recv()
#
#       steer_metadrive = steer_angle * 1 / (env.vehicle.MAX_STEERING * steer_ratio)
#       steer_metadrive = np.clip(steer_metadrive, -1, 1)
#
#       vc = [steer_metadrive, gas]
#
#       if should_reset:
#         reset()
#
#     if rk.frame % 5 == 0:
#       obs, _, terminated, _, info = env.step(vc)
#
#       if terminated:
#         reset()
#
#       if dual_camera:
#         wide_road_image[...] = get_cam_as_rgb("rgb_wide")
#       road_image[...] = get_cam_as_rgb("rgb_road")
#       image_lock.release()
#
#     rk.keep_time()
