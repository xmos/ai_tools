from enum import unique, Enum

# Strictness Levels
@unique
class Strictness(Enum):
    # Throw XCoreUnoptimisedError if not optimised
    ERROR = 0

    # Print warnings to stderr
    WARNING = 1
