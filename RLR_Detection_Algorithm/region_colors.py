from enum import Enum

class RegionColors(Enum):
    """
    Enum class to represent specific regions pairs colors with their corresponding BGR values.
    These colors are used for marking different regions pairs in images.
    """
    BLUE = (255, 0, 0)          # Blue
    MAGENTA = (255, 0, 255)     # Magenta
    CYAN = (255, 255, 0)        # Cyan
    ORANGE = (0, 165, 255)      # Orange
    PURPLE = (128, 0, 128)      # Purple
    TEAL = (128, 128, 0)        # Teal