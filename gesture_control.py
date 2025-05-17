import cv2
import mediapipe as mp
import socket
import time
from scapy.all import ARP, Ether, srp

# ESP32 info
TARGET_MAC = 'CC:DB:A7:99:A2:94'  # MAC ESP32 cần tìm
ESP32_PORT = 5000
TEST_MODE = True  # <-- Bật test mode

def get_local_subnet():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    net = local_ip.rsplit('.', 1)[0]
    return f"{net}.0/24"

def find_esp32_ip(subnet):
    pkt = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=subnet)
    ans, _ = srp(pkt, timeout=2, verbose=False)
    for _, r in ans:
        if r[Ether].src.lower() == TARGET_MAC.lower():
            return r[ARP].psrc
    return None

def send_udp_command(ip, action, speedA='150', speedB='150', test_mode=TEST_MODE):
    try:
        cmd = f"{action},{speedA},{speedB}\n"
        if test_mode:
            print(f"[🧪] (Test mode) Gửi lệnh: {cmd.strip()} đến {ip}:{ESP32_PORT}")
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(cmd.encode('utf-8'), (ip, ESP32_PORT))
        return cmd
    except Exception as e:
        print(f"[!] Lỗi gửi UDP: {e}")
        return None

def main():
    if TEST_MODE:
        esp32_ip = "192.168.197.29"  # IP giả lập để test
        print(f"[🧪] Test mode: sử dụng IP giả lập {esp32_ip}")
    else:
        print("[🔎] Tìm IP ESP32 trong mạng LAN...")
        subnet = get_local_subnet()
        esp32_ip = "192.168.197.29" # find_esp32_ip(subnet)
        if esp32_ip is None:
            print("❌ Không tìm thấy ESP32 trên mạng. Vui lòng kiểm tra kết nối.")
            return
        else:
            print(f"🎯 Đã tìm thấy ESP32 tại {esp32_ip}")

    mp_drawing_util = mp.solutions.drawing_utils
    mp_drawing_style = mp.solutions.drawing_styles
    mp_hand = mp.solutions.hands
    hands = mp_hand.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    fingersId = [8, 12, 16, 20]
    cap = cv2.VideoCapture(0)

    last_sent = None
    locked = False
    unlock_at = None

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        count = 0  # Tổng số ngón giơ của tất cả các bàn tay

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_drawing_util.draw_landmarks(
                    img_rgb,
                    hand,
                    mp_hand.HAND_CONNECTIONS,
                    mp_drawing_style.get_default_hand_landmarks_style(),
                    mp_drawing_style.get_default_hand_connections_style()
                )

                myHand = []
                for id, lm in enumerate(hand.landmark):
                    h, w, _ = img_rgb.shape
                    myHand.append([int(lm.x * w), int(lm.y * h)])

                fingers_count = 0
                # Đếm ngón trỏ, giữa, nhẫn, út
                for lm_index in fingersId:
                    if myHand[lm_index][1] < myHand[lm_index - 2][1]:
                        fingers_count += 1

                # Đếm ngón cái
                if myHand[4][0] < myHand[2][0] and myHand[5][0] <= myHand[13][0]:
                    fingers_count += 1
                elif myHand[4][0] > myHand[2][0] and myHand[5][0] >= myHand[13][0]:
                    fingers_count += 1

                count += fingers_count

        current_time = time.time()

        if locked and unlock_at and current_time >= unlock_at:
            print("[✅] Đã reset, có thể gửi lệnh mới.")
            locked = False
            last_sent = None
            unlock_at = None

        if locked and count == 10 and unlock_at is None:
            unlock_at = current_time + 3
            print("[🔄] Phát hiện 10 ngón tay - sẽ mở khóa sau 3s...")

        if not locked:
            action_map = {
                1: 'forward',
                2: 'reverse',
                3: 'left',
                4: 'right',
                5: 'stop',
            }

            if count in action_map:
                action = action_map[count]
                if last_sent != action:
                    sent = send_udp_command(esp32_ip, action, test_mode=TEST_MODE)
                    last_sent = action
                    locked = True
                    unlock_at = None
                    print(f"[📤] Gửi lệnh: {sent.strip()}")
                    cv2.putText(img_rgb, f"Sent: {sent.strip()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(img_rgb, f"Fingers: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("🖐️ Điều khiển cử chỉ tay", img_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[🛑] Đã dừng chương trình.")

if __name__ == "__main__":
    main()
