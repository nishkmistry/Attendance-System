"""
Enumerates available video capture device indices on this machine.

Run this after starting OBS Virtual Camera (as part of the drone screen-mirroring
bridge) to find which index Windows assigned it, then set that index as
DRONE_STREAM_URL (env var or via the "Drone Stream Settings" modal in the web UI).
"""
import cv2


def main():
    print("Scanning camera device indices 0-9...")
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Index {i}: OK - frame shape {frame.shape}")
        else:
            print(f"Index {i}: opened but no frame read")
        cap.release()


if __name__ == '__main__':
    main()
