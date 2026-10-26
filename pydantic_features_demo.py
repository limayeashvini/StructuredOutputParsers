"""
pydantic_features_demo.py (Pydantic v2+)
-----------------------------------------
Concept: Demonstrate key Pydantic features — modern syntax (v2.x)

This version removes all deprecation warnings and uses new-style validation.
"""

from typing import Optional, List, Annotated
from pydantic import BaseModel, Field, ValidationError


def show_validation_errors(e: ValidationError):
    """Print a ValidationError as a short, readable list (expected in demos)."""
    print(f"✅ Expected: Pydantic rejected the invalid data ({e.error_count()} errors)")
    for err in e.errors():
        field = ".".join(str(loc) for loc in err["loc"])
        print(f"   ❌ {field}: {err['msg']} (got {err['input']!r})")
    print()


# ==========================================
# 1️⃣ Basic Example — Type Checking & Validation
# ==========================================
class User(BaseModel):
    id: int
    name: str
    age: int

print("1️⃣ BASIC MODEL VALIDATION")
user = User(id="123", name="Alice", age=25)  # id coerced to int
print(user)
print(user.model_dump(), "\n")


# ==========================================
# 2️⃣ Default Values
# ==========================================
class Product(BaseModel):
    name: str
    price: float = 9.99
    in_stock: bool = True

print("2️⃣ DEFAULT VALUES")
product = Product(name="Notebook")
print(product.model_dump(), "\n")


# ==========================================
# 3️⃣ Optional Fields
# ==========================================
class Employee(BaseModel):
    id: int
    name: str
    department: Optional[str] = None
    manager_id: Optional[int] = None

print("3️⃣ OPTIONAL FIELDS")
employee = Employee(id=101, name="John Doe")
print(employee.model_dump(), "\n")


# ==========================================
# 4️⃣ Type Coercion (Automatic Type Conversion)
# ==========================================
class Item(BaseModel):
    name: str
    quantity: int
    price: float

print("4️⃣ TYPE COERCION")
item = Item(name="Pencil", quantity="5", price="1.25")
print(item.model_dump(), "\n")


# ==========================================
# 5️⃣ Built-in Validation (Error Handling)
# ==========================================
print("5️⃣ BUILT-IN VALIDATION EXAMPLE")
try:
    invalid_user = User(id="abc", name=123, age="NaN")
except ValidationError as e:
    show_validation_errors(e)


# ==========================================
# 6️⃣ Field() — Defaults, Constraints, Descriptions, Regex
# ==========================================
# Pydantic v2: Use Annotated + Field for constraints
UsernameType = Annotated[
    str,
    Field(
        min_length=3,
        max_length=12,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Username must be 3–12 chars (letters, numbers, underscores).",
    ),
]

PasswordType = Annotated[
    str,
    Field(min_length=6, description="Minimum 6 characters required."),
]

EmailType = Annotated[
    str,
    Field(
        pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
        description="Must be a valid email address.",
    ),
]


class Account(BaseModel):
    username: UsernameType
    password: PasswordType
    email: EmailType
    is_active: bool = Field(default=True, description="Whether the account is active.")
    roles: List[str] = Field(default_factory=lambda: ["user"], description="User roles list.")

print("6️⃣ FIELD() USAGE — CONSTRAINTS, DEFAULTS, DESCRIPTIONS, REGEX")
try:
    acc = Account(username="user_1", password="secret123", email="user@example.com")
    print(acc.model_dump(), "\n")
except ValidationError as e:
    print(e)

print("🧩 Try invalid email example:")
try:
    bad_acc = Account(username="user$", password="123", email="bad-email")
except ValidationError as e:
    show_validation_errors(e)
