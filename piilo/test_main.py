import logging

from fastapi.testclient import TestClient

logger = logging.getLogger('api')
logging.basicConfig(level=logging.INFO)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
    
def test_email():
    response = client.post("/anonymize",
                           json={"raw_text": "My name is joe@aol.com"},
                           )
    assert response.status_code == 200
    assert response.json() == {
        "anonymized_text": "My name is janedoe@aol.com"
    }

def test_name():
    response = client.post("/anonymize",
                           json={"raw_text": "My name is Nora Wang"},
                           )
    assert response.status_code == 200
    logger.info(response.json())
    
if __name__ == "__main__":
    from pathlib import Path
    import sys
    # Add piilo to sys.path so that we can import from piilo
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from main import app
    client = TestClient(app)
    
    test_read_main()
    test_email()
    test_name()