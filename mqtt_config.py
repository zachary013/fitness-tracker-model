# MQTT Configuration
MQTT_CONFIG = {
    'broker': 'localhost',  # MQTT broker address
    'port': 1883,           # Default MQTT port
    'client_id': 'pushup_detector',  # Unique client ID
    'topics': {
        'pushups': 'fitness/pushups',  # Topic for push-up data
        'cloud': 'fitness/cloud'       # Topic for cloud data (if needed)
    },
    'qos': 2,  # Quality of Service level (0, 1, or 2)
    'keepalive': 60  # Keepalive time in seconds
}

# Blynk Configuration
BLYNK_CONFIG = {
    'token': '3uBmJ6MJQInML8hKFY1jgwlM9I99D0iC',
    'pins': {
        'pushup_count': 0,  # V0
        'position': 1,      # V1
        'form_quality': 2   # V2
    }
}

