# Advanced Python Concepts

## Generators

Generators are functions that yield values one at a time, allowing lazy evaluation:

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Usage
fib = fibonacci()
for _ in range(10):
    print(next(fib))  # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
```

Generators are memory-efficient because they produce values on-the-fly rather than storing them all in memory. This makes them ideal for processing large datasets.

## Async/Await

Python supports asynchronous programming with `async` and `await`:

```python
import asyncio

async def fetch_data(url: str) -> str:
    # Simulate an API call
    await asyncio.sleep(1)
    return f"Data from {url}"

async def main():
    # Run multiple requests concurrently
    results = await asyncio.gather(
        fetch_data("https://api1.com"),
        fetch_data("https://api2.com"),
        fetch_data("https://api3.com"),
    )
    print(results)

asyncio.run(main())
```

## Type Hints

Python 3.5+ supports type hints for better code documentation and IDE support:

```python
from typing import Optional

def process_data(
    items: list[str],
    max_count: int = 100,
    prefix: Optional[str] = None,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items[:max_count]:
        key = f"{prefix}_{item}" if prefix else item
        result[key] = len(item)
    return result
```

## Dataclasses

Dataclasses reduce boilerplate for data-holding classes:

```python
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    label: str = "origin"
    tags: list[str] = field(default_factory=list)

    @property
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Point(3.0, 4.0, "A")
print(p.distance_from_origin)  # 5.0
```

## The Global Interpreter Lock (GIL)

The GIL is a mutex in CPython that allows only one thread to execute Python bytecode at a time. This means:

- CPU-bound tasks do NOT benefit from threading in Python
- Use `multiprocessing` for CPU-bound parallelism
- Use `threading` or `asyncio` for I/O-bound tasks (network, file I/O)
- The GIL does NOT affect multi-process programs

```python
from concurrent.futures import ProcessPoolExecutor

def cpu_intensive(n: int) -> int:
    return sum(i * i for i in range(n))

# Use processes for CPU-bound work
with ProcessPoolExecutor() as executor:
    results = list(executor.map(cpu_intensive, [10**6] * 4))
```

## Metaclasses

Metaclasses are classes that define how other classes are created:

```python
class SingletonMeta(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "connected"

# Both variables point to the same instance
db1 = Database()
db2 = Database()
assert db1 is db2  # True
```

## Descriptors

Descriptors are objects that define how attribute access is handled:

```python
class Validated:
    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, obj, value):
        if not self.min_value <= value <= self.max_value:
            raise ValueError(
                f"{self.name} must be between {self.min_value} and {self.max_value}"
            )
        obj.__dict__[self.name] = value

    def __get__(self, obj, objtype=None):
        return obj.__dict__.get(self.name)

class Temperature:
    celsius = Validated(-273.15, 1000)
    fahrenheit = Validated(-459.67, 1832)
```
