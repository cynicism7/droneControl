import time
import math
from swarm.drone import Drone
from swarm.boids import leader_velocity, follower_velocity
from config.swarm_config import *

def main():
    leader = Drone("Drone1", [0, 0, -TARGET_ALTITUDE])
    left  = Drone("Drone2", [3, 2, -TARGET_ALTITUDE])
    right = Drone("Drone3", [3, -2, -TARGET_ALTITUDE])

    drones = [leader, left, right]

    for d in drones:
        d.takeoff(TARGET_ALTITUDE)
    time.sleep(2)

    total_angle = 0.0
    last_angle = None

    while total_angle < 2 * math.pi:
        pos = leader.get_position()
        angle = math.atan2(pos[1], pos[0])

        if last_angle is not None:
            dtheta = angle - last_angle
            if dtheta > math.pi:
                dtheta -= 2 * math.pi
            if dtheta < -math.pi:
                dtheta += 2 * math.pi
            total_angle += abs(dtheta)

        last_angle = angle

        v_leader = leader_velocity(leader)
        leader.move(v_leader)

        left.move(follower_velocity(leader, left, +FORMATION_WIDTH))
        right.move(follower_velocity(leader, right, -FORMATION_WIDTH))

        time.sleep(DT)

    # 返回并降落
    for d in drones:
        d.go_home()
        d.client.landAsync(vehicle_name=d.name).join()

if __name__ == "__main__":
    main()
