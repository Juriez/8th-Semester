# # services.py
# class GreetingService:
#     def greet(self, name: str) -> str:
#         if not name:
#             raise ValueError("Name cannot be empty")
#         return f"Hello, {name}!"

#     def add(self, a: int, b: int, c: int = 0) -> int:
#         return a + b + c

#     def divide(self, x: float, y: float) -> float:
#         return x / y


# services.py  ← Decorator applied directly to methods
from aspects import logging_decorator

class GreetingService:
    @logging_decorator
    def greet(self, name: str) -> str:
        if not name:
            raise ValueError("Name cannot be empty")
        return f"Hello, {name}!"

    @logging_decorator
    def add(self, a: int, b: int, c: int = 0) -> int:
        return a + b + c

    @logging_decorator
    def divide(self, x: float, y: float) -> float:
        return x / y