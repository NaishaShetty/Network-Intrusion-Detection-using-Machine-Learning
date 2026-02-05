
import urllib.request
import urllib.parse
import json
import mimetypes

url = "http://127.0.0.1:8000/api/predict?model_name=decision_tree"
file_path = "sample_network_traffic.csv"
output_file = "api_test_result.txt"

def post_multipart(url, file_path):
    boundary = '---BOUNDARY---'
    lines = []
    
    # File part
    with open(file_path, "rb") as f:
        file_content = f.read()
        
    lines.append(f'--{boundary}'.encode())
    lines.append(f'Content-Disposition: form-data; name="file"; filename="{file_path}"'.encode())
    lines.append(f'Content-Type: text/csv'.encode())
    lines.append(b'')
    lines.append(file_content)
    lines.append(f'--{boundary}--'.encode())
    lines.append(b'')
    
    body = b'\r\n'.join(lines)
    
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    
    try:
        with urllib.request.urlopen(req) as response:
            return f"Status: {response.status}\nResponse: {response.read().decode()}"
    except urllib.error.HTTPError as e:
        return f"Status: {e.code}\nError: {e.read().decode()}"
    except Exception as e:
        return f"Exception: {str(e)}"

# Run
result = post_multipart(url, file_path)
with open(output_file, "w") as f:
    f.write(result)
    
print("Test complete. Check", output_file)
