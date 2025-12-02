# main.py  ← No aspectlib imports or weave calls
import logging
from services import GreetingService

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)

if __name__ == "__main__":
    service = GreetingService()

    print("=== AOP LOGGING IS NOW WORKING ===\n")
    print("Result:", service.greet("Mahir"))
    print("Sum   :", service.add(190, 200, c=30))

    print("\n--- Testing exceptions ---")
    try:
        service.greet("")
    except ValueError:
        pass
    try:
        service.divide(10, 0)
    except ZeroDivisionError:
        pass