from ultralytics import YOLO
import cv2

# Tải mô hình đã được huấn luyện trước
model = YOLO("yolov8n.pt")  # Sử dụng mô hình YOLOv8 large đã được huấn luyện trước

# Đường dẫn đến video đầu vào
video_path = "D:/Desktop/yolov8/yolov8_DATN/people.mp4"

# Mở video bằng OpenCV
cap = cv2.VideoCapture(0)

# Kiểm tra xem video có được mở thành công không
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Xử lý từng khung hình của video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Thực hiện dự đoán trên khung hình hiện tại
    results = model.predict(source=frame, stream=True, save=False, imgsz=320, conf=0.25)

    # Trích xuất thông tin từ kết quả dự đoán
    for result in results:
        # Lấy các hộp giới hạn (Bounding boxes)
        boxes = result.boxes  # Đây là dạng BoundingBox trong YOLOv8

        for box in boxes:
            # Trích xuất tọa độ hộp giới hạn
            x1, y1, x2, y2 = box.xyxy[0]  # Tọa độ của hộp phát hiện (top-left và bottom-right)
            # Trích xuất điểm số tự tin và lớp đối tượng
            conf = box.conf[0].item()  # Điểm số tự tin của hộp phát hiện
            cls = box.cls[0].item()  # Lớp đối tượng của hộp phát hiện

            # Lấy tên lớp từ model.names
            class_name = model.names[int(cls)]

            # In thông tin phát hiện
            print(f"Frame {frame_count}: {class_name} ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), Conf: {conf:.2f}")

            # Vẽ hộp phát hiện lên khung hình
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    # Hiển thị khung hình đã xử lý
    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
