
# 仿真参数配置

DT = 0.2                        # 控制周期
TARGET_ALTITUDE = 10.0          # 默认飞行高度
ALTITUDE_MARGIN = 2.0           # 避障允许上下偏移
FLIGHT_DURATION = 60            # 总飞行时间 (s)
CIRCLE_RADIUS = 30              # 围绕原点飞行半径
CIRCLE_SPEED = 2                # 绕圈速度 (m/s)
MAX_SPEED = 5.0                 # 最大速度
MAX_Z_SPEED = 1.0               # 最大垂直速度
DESIRED_DISTANCE = 5.0          # Boids 聚合参考距离
MIN_DISTANCE = 3.0              # Boids 分离最小距离
AVOID_DISTANCE = 6.0            # 避障检测距离
MAX_LEADER_SPEED = 2.5          # Leader 最大速度
# Boids 权重
K_TARGET = 1.0                  # 目标吸引力
K_COHESION = 0.5                # 聚合权重
K_SEPARATION = 1.0              # 分离权重
K_AVOID = 5.0                   # 避障权重
FORMATION_DISTANCE = 3.0        # 队形保持距离
FORMATION_WIDTH = 2.0           # 队形左右偏移
K_FORMATION = 1.5               # 队形保持权重
KP_POSITION = 0.6               # follower 位置跟随增益
KP_ALTITUDE = 0.8               # 高度回归增益
TARGET_RADIUS = 10.0            # 绕圈半径（米）
KP_RADIUS = 0.6                 # 拉回圆的强度