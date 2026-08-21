# Python Basics

## What is Python?

Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with the use of significant indentation. It supports multiple programming paradigms, including structured, object-oriented, and functional programming.

## Variables and Data Types

Python has several built-in data types:

- **Integers** (`int`): Whole numbers like `42`, `-7`, `0`
- **Floats** (`float`): Decimal numbers like `3.14`, `-0.001`
- **Strings** (`str`): Text enclosed in quotes like `"hello"`, `'world'`
- **Booleans** (`bool`): `True` or `False`
- **Lists** (`list`): Ordered, mutable collections like `[1, 2, 3]`
- **Tuples** (`tuple`): Ordered, immutable collections like `(1, 2, 3)`
- **Dictionaries** (`dict`): Key-value pairs like `{"name": "Alice", "age": 30}`
- **Sets** (`set`): Unordered collections of unique elements like `{1, 2, 3}`

Variables in Python are dynamically typed, meaning you don't need to declare their type:

```python
name = "Alice"      # str
age = 30             # int
height = 5.6         # float
is_student = True    # bool
```

## Functions

Functions are defined using the `def` keyword:

```python
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"

result = greet("Alice")  # "Hello, Alice!"
```

## Decorators

A decorator is a function that takes another function and extends its behavior without explicitly modifying it. Decorators are applied using the `@` syntax:

```python
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
    return "done"
```

Decorators are commonly used for:
- Logging function calls
- Measuring execution time
- Access control and authentication
- Caching (memoization) with `@functools.lru_cache`
- Input validation

## Classes and Object-Oriented Programming

Python supports object-oriented programming with classes:

```python
class Dog:
    def __init__(self, name: str, breed: str):
        self.name = name
        self.breed = breed

    def bark(self) -> str:
        return f"{self.name} says Woof!"

    def __repr__(self) -> str:
        return f"Dog(name={self.name!r}, breed={self.breed!r})"

my_dog = Dog("Buddy", "Golden Retriever")
print(my_dog.bark())  # "Buddy says Woof!"
```

## Error Handling

Python uses try/except blocks for error handling:

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
finally:
    print("This always runs")
```

## List Comprehensions

List comprehensions provide a concise way to create lists:

```python
# Traditional loop
squares = []
for x in range(10):
    squares.append(x ** 2)

# List comprehension (equivalent)
squares = [x ** 2 for x in range(10)]

# With filtering
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
```

## Context Managers

Context managers handle resource setup and cleanup using the `with` statement:

```python
# File handling
with open("data.txt", "r") as f:
    content = f.read()
# File is automatically closed after the block

# Custom context manager
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    yield
    print(f"Elapsed: {time.time() - start:.2f}s")

with timer():
    # ... do something slow ...
    pass
```

## Virtual Environments

Virtual environments isolate project dependencies:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (macOS/Linux)
source .venv/bin/activate

# Install packages
pip install requests numpy

# Freeze dependencies
pip freeze > requirements.txt
```
