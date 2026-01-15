_logging = False


def log(*args, **kwargs):
    if _logging:
        print(*args, **kwargs)


def start():
    global _logging
    _logging = True


def end():
    global _logging
    _logging = False

__all__ = ['log', 'start', 'end']