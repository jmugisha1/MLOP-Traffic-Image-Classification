# locustfile.py
from locust import HttpUser, task, between

class ModelUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        with open(r'C:/Users/Administrator/Desktop/test/test122.jpg', 'rb') as img:
            files = {'image': img}
            self.client.post("/api/predict/", files=files)