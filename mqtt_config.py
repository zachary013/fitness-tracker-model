# mqtt_config.py

# MQTT Configuration
MQTT_CONFIG = {
    'broker': 'localhost',
    'port': 1883,
    'client_id': 'exercise_detector',
    'topics': {
        'exercise_state': 'fitness/exercise_state',  # Changed from just 'exercise_state'
        'pushups': 'fitness/pushups'
    },
    'qos': 1,
    'keepalive': 60
}

# # Blynk Configuration
# BLYNK_CONFIG = {
#     'token': '3uBmJ6MJQInML8hKFY1jgwlM9I99D0iC',
#     'pins': {
#         'pushup_count': 0,  # V0
#         'position': 1,      # V1
#         'form_quality': 2   # V2
#     }
# }