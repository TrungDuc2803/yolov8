import cv2
import argparse
import threading
import queue

from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def video_capture_thread(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoundingBoxAnnotator(
        thickness=2
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2
    )

    frame_queue = queue.Queue()
    threading.Thread(target=video_capture_thread, args=(cap, frame_queue), daemon=True).start()

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            results = model(frame, agnostic_nms=True)[0]

            detections = results.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
            confidences = results.boxes.conf.cpu().numpy()  # Extract confidences
            class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Extract class IDs

            labels = [
                f"{model.names[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(class_ids, confidences)
            ]

            annotated_frame = box_annotator(frame, detections, labels=labels)  # Annotate bounding boxes

            detections_for_zone = [
                sv.Detection(bbox=detection, confidence=confidence, class_id=class_id)
                for detection, confidence, class_id in zip(detections, confidences, class_ids)
            ]

            zone.trigger(detections=detections_for_zone)  # Trigger zone
            annotated_frame = zone_annotator.annotate(scene=annotated_frame)  # Annotate zone

            cv2.imshow("yolov8", annotated_frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()
