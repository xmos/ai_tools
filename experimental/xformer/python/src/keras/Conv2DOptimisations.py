from enum import unique, Enum

# Types of Conv2D optimisations
@unique
class Conv2DOptimisation(Enum):
    DEFAULT = 0
    PADDED_INDIRECT = 1
    VALID_INDIRECT = 2
    VALID_DIRECT = 3
