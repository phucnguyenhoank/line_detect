import cv2
import numpy as np
import tensorflow as tf
import socket
import time

# Tải mô hình .tflite
interpreter = tf.lite.Interpreter(model_path="steering_model_clean.tflite")
interpreter.allocate_tensors()

# Lấy thông tin input và output của mô hình
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cấu hình UDP để gửi lệnh tới ESP32q
ESP32_IP = "192.168.86.29"  # Thay bằng IP thực tế của ESP32
ESP32_PORT = 5000
BASE_SPEED = 200
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Throttle gửi lệnh
SEND_INTERVAL = 0.5  # Gửi lệnh mỗi 0.5 giây
last_send_time = 0

# Kiểm tra trên luồng video
stream_url = "http://192.168.86.130:81/stream"
cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc luồng video.")
        break

    # Chuẩn bị frame cho dự đoán
    frame_resized = cv2.resize(frame, (224, 224))
    frame_processed = tf.keras.applications.mobilenet_v2.preprocess_input(frame_resized)
    input_data = np.expand_dims(frame_processed, axis=0).astype(np.float32)

    # Đặt dữ liệu đầu vào và chạy dự đoán
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Lấy kết quả dự đoán
    output_data = interpreter.get_tensor(output_details[0]["index"])
    prediction = np.argmax(output_data[0])
    result = ["right", "forward", "left"][prediction]
    print(f"Dự đoán: {result}")

    # Gửi lệnh tới ESP32 nếu đã đủ thời gian
    current_time = time.time()
    if current_time - last_send_time >= SEND_INTERVAL:
        command = f"{result},{BASE_SPEED},{BASE_SPEED}"
        try:
            sock.sendto(command.encode(), (ESP32_IP, ESP32_PORT))
            print(f"Đã gửi lệnh: {command}")
            last_send_time = current_time
        except Exception as e:
            print(f"Lỗi khi gửi lệnh: {e}")

    # Hiển thị frame (tùy chọn)
    cv2.imshow("Video Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Nhấn 'q' để thoát
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
sock.close()