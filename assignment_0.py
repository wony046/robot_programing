class Calculator:

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    def power(self, a, b):
        return a ** b

cal = Calculator()


while True:
    while True:
        try:
            a = float(input("Enter the first number: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    while True:
        try:
            b = float(input("Enter the second number: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    while True:
        cal_type = input("Enter the operation (add, subtract, multiply, divide, power): ")
        if cal_type in ["add", "subtract", "multiply", "divide", "power"]:
            break
        else:
            print("Invalid operation. Please enter a valid operation.")

    try:
        if cal_type == "add":
            print("Result:", cal.add(a, b))
        elif cal_type == "subtract":
            print("Result:", cal.subtract(a, b))
        elif cal_type == "multiply":
            print("Result:", cal.multiply(a, b))
        elif cal_type == "divide":
            print("Result:", cal.divide(a, b))
        elif cal_type == "power":
            print("Result:", cal.power(a, b))
        else:
            print("Invalid operation.")

    except (ValueError, ZeroDivisionError, OverflowError) as e:
        print(e)
        

    if input("Do you want to perform another calculation? (yes/no): ").lower() != "yes":
        break
    else:
        print("Starting a new calculation...")

