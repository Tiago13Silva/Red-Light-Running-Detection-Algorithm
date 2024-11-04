from enum import Enum

class LightStates(Enum):
    """
    
    """
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    GREEN = (0, 255, 0)
    OFF = (0, 0, 0)
    # OTHER = (0, 0, 0)