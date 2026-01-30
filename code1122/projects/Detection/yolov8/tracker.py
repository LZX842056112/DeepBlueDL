# -*- coding: utf-8 -*-
from ultralytics import YOLO
from ultralytics.engine.results import Results


def t0():
    # Load an official or custom model
    # model = YOLO('./configs/yolov8n.pt')  # Load an official Detect model
    model = YOLO('./configs/yolov8n-seg.pt')  # Load an official Segment model
    # model = YOLO('./configs/yolov8n-pose.pt')  # Load an official Pose model

    # Perform tracking with the model     # Tracking with default tracker
    results = model.track(
        source="./datasets/test.mp4",
        show=True,
        # tracker=r"./configs/bytetrack.yaml",
        tracker=r"./configs/botsort.yaml",
        persist=True
    )
    print(type(results))


def t1():
    # Load an official or custom model
    model = YOLO('./configs/yolov8n.pt')  # Load an official Detect model
    # model = YOLO('./configs/yolov8n-seg.pt')  # Load an official Segment model
    # model = YOLO('./configs/yolov8n-pose.pt')  # Load an official Pose model

    # Perform tracking with the model     # Tracking with default tracker
    results = model.track(
        source="./datasets/test.mp4",
        # tracker=r"./configs/bytetrack.yaml",
        tracker=r"./configs/botsort.yaml",
        stream=True
    )
    for result in results:
        r: Results = result
        print(type(result))
        print(r.boxes.id)


if __name__ == '__main__':
    t0()
