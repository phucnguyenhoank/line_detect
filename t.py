import cv2
import numpy as np

def find_tape_position(frame):
    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray Image", gray)  # Hiển thị ảnh grayscale để kiểm tra
    cv2.waitKey(0)

    # Lấy chiều cao và chiều rộng của ảnh
    H, W = gray.shape

    # Tính số hàng tương ứng với 10% vùng dưới cùng
    region_height = int(H * 0.1)
    # Lấy 10% vùng dưới cùng
    bottom_region = gray[H - region_height:H, :]

    # Áp dụng threshold để tách pixel đen (giả định pixel đen có giá trị cường độ thấp)
    thresh = (bottom_region < 50).astype(np.uint8)  # Pixel đen < 50, nền sáng > 50

    # Tìm tất cả các pixel đen trong vùng 10% dưới cùng
    black_pixels_y, black_pixels_x = np.where(thresh == 1)  # Lấy tọa độ x, y của pixel đen

    if len(black_pixels_x) == 0:
        return 0, None  # Trả về 0 và None nếu không tìm thấy pixel đen

    # Tính tọa độ x trung bình của các pixel đen
    x_mean = np.mean(black_pixels_x)

    # Tính giá trị đầu ra: khoảng cách từ trung tâm ảnh (W/2)
    output = x_mean - W / 2

    return output, x_mean  # Trả về output và x_mean để vẽ



# Ví dụ sử dụng với video stream
if __name__ == "__main__":
    read_image = True
    if read_image:
        # Đọc ảnh từ file
        image_path = r"images\20250510191736 (4).jpg"  # Đường dẫn đến ảnh
        frame = cv2.imread(image_path)
        if frame is None:
            print("Lỗi: Không thể đọc ảnh từ đường dẫn cung cấp.")
            exit()
        # Tính vị trí dải băng keo
        result = find_tape_position(frame)
        if isinstance(result, tuple):
            position, x_mean = result
            if x_mean is not None:
                # Vẽ điểm trung bình của dải băng keo (màu đỏ) ở hàng dưới cùng
                cv2.circle(frame, (int(x_mean), frame.shape[0]-1), 5, (0, 0, 255), -1)
        else:
            position = result
        # In vị trí dải băng keo
        print(f"Vị trí dải băng keo: {position}")
    else:
        # Đọc từ video stream
        # Thay đổi URL theo địa chỉ IP của camera của bạn

        stream_url = "http://192.168.46.130:81/stream"
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            print("Lỗi: Không thể mở stream từ URL cung cấp.")
            exit()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Lỗi: Không thể đọc frame từ stream.")
                    break

                # Tính vị trí dải băng keo
                result = find_tape_position(frame)
                if isinstance(result, tuple):
                    position, x_mean = result
                    if x_mean is not None:
                        # Vẽ điểm trung bình của dải băng keo (màu đỏ) ở hàng dưới cùng
                        cv2.circle(frame, (int(x_mean), frame.shape[0]-1), 5, (0, 0, 255), -1)
                else:
                    position = result

                # In vị trí dải băng keo
                print(f"Vị trí dải băng keo: {position}")

                # Hiển thị frame
                cv2.imshow("Camera Stream", frame)

                # Nhấn 'q' để thoát
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Lỗi: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()