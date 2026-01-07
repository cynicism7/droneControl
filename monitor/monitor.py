import airsim
import numpy as np
import cv2

MONITOR_SIZE = 1024

def get_monitor_view(client):
    response = client.simGetImage(
        camera_name="MonitorCam",
        image_type=airsim.ImageType.Scene,
        vehicle_name=""
    )

    if response is None:
        return None

    img = np.frombuffer(response, dtype=np.uint8)
    img = img.reshape(MONITOR_SIZE, MONITOR_SIZE, 3)
    return img


def show_monitor(client):
    img = get_monitor_view(client)
    if img is not None:
        cv2.imshow("Global Monitor (40m Radius)", img)
        cv2.waitKey(1)
