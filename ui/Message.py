message = {
    1: {"Camera is OFF"},
    2: {"Turning ON camera. Please wait..."},
    3: {"Please turn ON the camera first"},
    4: {"Camera not detected in this channel. Please change the camera channel."},
    5: {"Gesture is not selected"},
    6: {"Please select any of gesture listed"},
    7: {'You are too close... Please step backward'},
    8: {'You are too far... Please step forward'},
    9: {'Capturing images...'},
    10: {' HAND is not detected'}
}

class Msg:
    def get_message(self, index):
        message_value = message.get(index)
        return next(iter(message_value), "Index not found") if message_value else "Index not found"