import logging
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)

def logging_decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get method name
        method_name = func.__name__
        
        # Build args string (skip 'self' for instance methods)
        if args:
            self = args[0]
            real_args = args[1:]
        else:
            real_args = ()
        args_str = ", ".join(repr(a) for a in real_args)
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        full_args = args_str + (", " + kwargs_str if kwargs_str else "")
        
        logger.debug(f">> Entering {method_name}({full_args})")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"<< Exiting {method_name}() → {result!r}")
            return result
        except Exception as exc:
            logger.error(f"!! Exception in {method_name}(): {exc}")
            raise
    return wrapper