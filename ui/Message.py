message = {
    1: {"Camera is OFF"},
    2: {"Turning ON camera. Please wait..."},
    3: {"Please turn ON the camera first"},
    4: {"Camera not detected in this channel. Please change the camera channel."},
    5: {"Gesture is not selected"},
    6: {"Please select any of gesture listed"}
}

class Msg:
    def get_message(self, index):
        message_value = message.get(index)
        return next(iter(message_value), "Index not found") if message_value else "Index not found"