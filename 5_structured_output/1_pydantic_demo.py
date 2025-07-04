from pydantic import BaseModel, EmailStr


class Student(BaseModel):
    name: str
    age: int
    email: EmailStr


# Option 1: Create a dictionary with proper keys
new_student = {"name": "Ravi", "age": 36, "email": "abc@mail.com"}  # Note: using 36 as int, not string
student = Student(**new_student)  # Use ** to unpack dictionary as keyword arguments

# Option 2: Pass arguments directly
# student = Student(name="Ravi", age=36)

print(student)
