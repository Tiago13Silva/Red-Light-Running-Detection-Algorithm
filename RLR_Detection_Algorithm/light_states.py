from enum import Enum

class LightStates(Enum):
    """
    Enum class to represent the states of a traffic light, using BGR color values.
    Each state corresponds to a specific color or the light being off.
    """
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    GREEN = (0, 255, 0)
    OFF = (0, 0, 0)