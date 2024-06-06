import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Tải mô hình đã được huấn luyện trước
weights_path = "yolov8n.pt"

# Load mô hình YOLOv8
net = cv2.dnn.readNet(weights_path)

# Đường dẫn đến video đầu vào
video_path = "D:/Desktop/yolov8/yolov8_DATN/people.mp4"

# Mở video bằng OpenCV
cap = cv2.VideoCapture(video_path)

# Kiểm tra xem video có được mở thành công không
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Khởi tạo DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Chuyển đổi khung hình sang blob
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Đưa blob vào mạng để dự đoán
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    detections = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = cv2.dnn.NMSBoxes([detection[:4]], scores, score_threshold=0.5, nms_threshold=0.4)
            if len(class_id) > 0:
                for i in class_id.flatten():
                    center_x, center_y, width, height, confidence = detection[:5]
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    detections.append((x, y, int(width), int(height), float(confidence), int(i)))

    # Cập nhật tracker với các phát hiện hiện tại
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for obj in tracked_objects:
        if not obj.is_confirmed():
            continue

        track_id = obj.track_id
        bbox = obj.to_tlbr()  # Trả về hộp bounding dưới dạng (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        class_name = obj.det_class  # Lớp đối tượng

        # Vẽ hộp phát hiện và ID của đối tượng lên khung hình
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    # Hiển thị khung hình đã xử lý
    cv2.imshow('YOLOv8n Detection with Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
