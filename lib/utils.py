from typing import Callable


def equals(x) -> Callable[[int], bool]:
    return lambda y: y == x

def not_equals(x) -> Callable[[int], bool]:
    return lambda y: y != x