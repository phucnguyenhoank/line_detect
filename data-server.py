from flask import Flask, request
import os
import time

app = Flask(__name__)

# Đảm bảo thư mục dataset tồn tại
if not os.path.exists('dataset'):
    os.makedirs('dataset')

@app.route('/upload', methods=['POST'])
def upload():
    label = request.form['label']
    image = request.files['image']
    # Lưu ảnh với tên gồm nhãn và timestamp
    filename = f"{label}_{int(time.time())}.jpg"
    image.save(os.path.join('dataset', filename))
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)