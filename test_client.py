import sys
import requests

def upload_image(file_path, capacity=50):
    url = 'http://127.0.0.1:5000/upload'
    with open(file_path, 'rb') as f:
        files = {'image': (file_path, f, 'image/jpeg')}
        data = {'capacity': str(capacity)}
        resp = requests.post(url, files=files, data=data, timeout=60)
    return resp


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python test_client.py <image_file> [capacity]')
        sys.exit(1)
    image = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    r = upload_image(image, cap)
    print('Status:', r.status_code)
    try:
        print(r.json())
    except Exception as e:
        print('Response text:', r.text)
