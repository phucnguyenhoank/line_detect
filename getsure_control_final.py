import cv2
import mediapipe as mp
import socket
from collections import deque
from statistics import mode
import time

# Thông tin ESP32 (thay đổi nếu cần)
ESP32_IP = "192.168.197.29"
ESP32_PORT = 5000
STACK_SIZE = 10  # Kích thước stack
ACTION_MAP = {1: 'forward', 2: 'left', 3: 'right', 4: 'reverse', 0: 'stop'}
SEND_INTERVAL = 0.5  # Gửi lệnh mỗi 500ms

class GestureController:
    def __init__(self):
        # Khởi tạo Mediapipe Hands
        self.mp_hands = mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)  # Mở webcam
        self.detect_stack = deque(maxlen=STACK_SIZE)  # Stack lưu detect
        self.current_action = None  # Hành động hiện tại
        self.last_send_time = 0  # Thời gian gửi lệnh cuối

    def send_udp_command(self, action):
        """Gửi lệnh UDP tới ESP32"""
        if action == 'stop':
            print("[🛑] Dừng xe (không gửi lệnh)")
            return
        cmd = f"{action},200,200\n"
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(cmd.encode('utf-8'), (ESP32_IP, ESP32_PORT))
        print(f"[📤] Gửi lệnh: {cmd.strip()}")

    def count_fingers(self, hand_landmarks, img_shape):
        """Đếm số ngón tay"""
        myHand = [[int(lm.x * img_shape[1]), int(lm.y * img_shape[0])] for lm in hand_landmarks.landmark]
        count = 0
        # Đếm ngón trỏ, giữa, nhẫn, út
        for idx in [8, 12, 16, 20]:
            if myHand[idx][1] < myHand[idx - 2][1]:
                count += 1
        # Đếm ngón cái
        if (myHand[4][0] < myHand[2][0] and myHand[5][0] <= myHand[13][0]) or \
           (myHand[4][0] > myHand[2][0] and myHand[5][0] >= myHand[13][0]):
            count += 1
        return count

    def run(self):
        """Chạy vòng lặp chính"""
        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                break

            # Xử lý ảnh
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.mp_hands.process(img_rgb)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            total_fingers = 0
            # Phát hiện tay và đếm ngón
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)
                    total_fingers += self.count_fingers(hand, img.shape)

            # Thêm số ngón tay vào stack
            self.detect_stack.append(total_fingers)

            # Khi stack đủ kích thước, tính mode để cập nhật hành động
            if len(self.detect_stack) == STACK_SIZE:
                try:
                    finger_mode = mode(self.detect_stack)
                    if finger_mode in ACTION_MAP:
                        self.current_action = ACTION_MAP[finger_mode]
                except:
                    pass  # Bỏ qua nếu không có mode duy nhất

            # Gửi lệnh liên tục nếu không phải 'stop'
            current_time = time.time()
            if self.current_action and self.current_action != 'stop' and (current_time - self.last_send_time >= SEND_INTERVAL):
                self.send_udp_command(self.current_action)
                self.last_send_time = current_time

            # Hiển thị thông tin
            cv2.putText(img, f"Fingers: {total_fingers}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            if self.current_action:
                cv2.putText(img, f"Action: {self.current_action}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("🖐️ Điều khiển cử chỉ tay", img)

            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("[🛑] Đã dừng chương trình.")

def main():
    controller = GestureController()
    controller.run()

if __name__ == "__main__":
    main()