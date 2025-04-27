import requests
import json

def send_image(image_path):
    url = 'http://localhost:5000/predict'
    
    with open(image_path, 'rb') as img:
        files = {'file': img}
        response = requests.post(url, files=files)
        
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': f'Request failed with status code {response.status_code}'}

if __name__ == '__main__':
    # Example usage
    image_path = 'test.jpg'  # Replace with your image path
    result = send_image(image_path)
    print(json.dumps(result, indent=2)) 