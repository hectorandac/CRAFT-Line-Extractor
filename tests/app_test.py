import pytest
from io import BytesIO
from main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_process_image_no_file(client):
    """Test the process_image endpoint without providing an image"""
    response = client.post('/process_image')
    assert response.status_code == 400
    assert response.json['error'] == 'No image provided'

def test_process_image_invalid_json(client):
    """Test the process_image endpoint with invalid JSON"""
    data = {'options': 'invalid_json'}
    response = client.post('/process_image', data=data)
    assert response.status_code == 400
    assert response.json['error'] == 'Invalid JSON in options'

def test_process_image_with_file(client):
    """Test the process_image endpoint with a valid image"""
    data = {
        'image': (BytesIO(b'my image data'), 'test.png'),
        'options': '{"return_image": true, "return_coords": true}'
    }
    response = client.post('/process_image', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert 'image_url' in response.json
    assert 'lines' in response.json
