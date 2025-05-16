import streamlit as st
import cv2
import mediapipe as mp
import socket
from scapy.all import ARP, Ether, srp

st.set_page_config(page_title="Điều khiển xe bằng cử chỉ tay", page_icon="🖐️")
st.title("✋ Điều khiển xe bằng cử chỉ tay")

# ESP32 info
TARGET_MAC = 'CC:DB:A7:99:A2:94'.lower()
ESP32_PORT = 5000

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
        if r[Ether].src.lower() == TARGET_MAC:
            return r[ARP].psrc
    return None

ESP32_IP = "192.168.86.29" # find_esp32_ip(get_local_subnet())
if ESP32_IP:
    st.success(f"🎯 Đã tìm thấy ESP32 tại {ESP32_IP}")
else:
    st.error("❌ Không tìm thấy ESP32 trên mạng. Vui lòng kiểm tra kết nối.")

# Mediapipe
mp_drawing_util = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

start_button = st.button("Bắt đầu")
stop_button = st.button("Dừng")

if start_button and not stop_button and ESP32_IP:
    cap = cv2.VideoCapture(0)
    fingersId = [8, 12, 16, 20]
    image_placeholder = st.empty()

    last_sent = None

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        count = 0

        if result.multi_hand_landmarks:
            myHand = []

            for idx, hand in enumerate(result.multi_hand_landmarks):
                mp_drawing_util.draw_landmarks(
                    img_rgb,
                    hand,
                    mp_hand.HAND_CONNECTIONS,
                    mp_drawing_style.get_default_hand_landmarks_style(),
                    mp_drawing_style.get_default_hand_connections_style()
                )

                for id, lm in enumerate(hand.landmark):
                    h, w, _ = img_rgb.shape
                    myHand.append([int(lm.x * w), int(lm.y * h)])

                for lm_index in fingersId:
                    if myHand[lm_index][1] < myHand[lm_index - 2][1]:
                        count += 1

                # Ngón cái
                if myHand[4][0] < myHand[2][0] and myHand[5][0] <= myHand[13][0]:
                    count += 1
                elif myHand[4][0] > myHand[2][0] and myHand[5][0] >= myHand[13][0]:
                    count += 1

        # Gửi lệnh qua UDP
        def send_udp_command(action, speedA='150', speedB='150'):
            try:
                cmd = f"{action},{speedA},{speedB}\n"
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(cmd.encode('utf-8'), (ESP32_IP, ESP32_PORT))
                return cmd
            except Exception as e:
                st.error(f"Lỗi gửi UDP: {e}")
                return None

        action_map = {
            1: 'forward',
            2: 'reverse',
            3: 'left',
            4: 'right',
            5: 'stop'
        }

        if count in action_map:
            action = action_map[count]
            if last_sent != action:
                sent = send_udp_command(action)
                last_sent = action
                cv2.putText(img_rgb, f"Sent: {sent}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(img_rgb, f"Fingers: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        image_placeholder.image(img_rgb, channels="BGR", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()

st.button("Re-run")
