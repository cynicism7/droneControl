import time
import numpy as np
from swarm.drone import Drone
from swarm.boids import compute_swarm_center, compute_avoidance_vector, compute_boids_velocity, compute_circular_velocity
from config.swarm_config import TARGET_ALTITUDE, FLIGHT_DURATION, DT

def main():
    start_positions = [
        [0, 0, -TARGET_ALTITUDE],
        [5, 0, -TARGET_ALTITUDE],
        [-5, 0, -TARGET_ALTITUDE]
    ]
    drones = [Drone(f"Drone{i+1}", pos) for i, pos in enumerate(start_positions)]

    print("无人机起飞...")
    for d in drones:
        d.takeoff(TARGET_ALTITUDE)
    time.sleep(2)

    print("开始绕原点飞行...")
    t_total = 0
    while t_total < FLIGHT_DURATION:
        swarm_center = compute_swarm_center(drones)
        for d in drones:
            target_velocity = compute_circular_velocity(d.get_position())
            v_avoid = compute_avoidance_vector(drones)
            v = compute_boids_velocity(d, drones, target_velocity, v_avoid)
            d.move(v, dt=DT)
        t_total += DT
        time.sleep(DT)

    print("飞行结束，返回起点...")
    for d in drones:
        d.go_home()
    print("任务完成！")

if __name__ == "__main__":
    main()
