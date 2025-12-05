import csv
from datetime import datetime, timedelta
import random
import os

def generate_mock_data():
    file_path = 'mock_data.csv'
    if os.path.exists(file_path):
        return  # Skip if file already exists
        
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'timestamp', 'value', 'sensor_type', 'location'])
        
        for i in range(1, 1001):
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 10080))
            sensor_type = random.choice(['temperature', 'humidity', 'pressure'])
            
            if sensor_type == 'temperature':
                value = random.uniform(10, 40)
            elif sensor_type == 'humidity':
                value = random.uniform(0, 100)
            else:  # pressure
                value = random.uniform(900, 1100)
                
            location = f"Building {random.randint(1, 20)}-Floor {random.randint(1, 5)}"
            
            writer.writerow([
                i,
                timestamp.isoformat(),
                round(value, 2),
                sensor_type,
                location
            ])

if __name__ == '__main__':
    generate_mock_data()