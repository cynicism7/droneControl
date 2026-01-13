import time
import math
import numpy as np
from swarm.drone import Drone
from swarm.boids import leader_velocity, follower_velocity
from config.swarm_config import *

def main():
    leader = Drone("Drone1", [0, 0, -TARGET_ALTITUDE])
    left   = Drone("Drone2", [3,  2, -TARGET_ALTITUDE])
    right  = Drone("Drone3", [3, -2, -TARGET_ALTITUDE])

    drones = [leader, left, right]

    for d in drones:
        d.takeoff(TARGET_ALTITUDE)
    time.sleep(2)

    arc_length = 0.0

    while arc_length < 2 * math.pi * CIRCLE_RADIUS:
        # leader 速度统一从 boids 中获取
        v_leader = leader_velocity(leader)
        leader.move(v_leader)

        # follower 跟随
        left.move(follower_velocity(leader, left, +FORMATION_WIDTH))
        right.move(follower_velocity(leader, right, -FORMATION_WIDTH))

        # 路程积分（关键）
        arc_length += np.linalg.norm(v_leader) * DT

        time.sleep(DT)

    for d in drones:
        d.go_home()
        d.client.landAsync(vehicle_name=d.name).join()
if __name__ == "__main__":
    main()