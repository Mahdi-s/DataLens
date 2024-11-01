import pandas as pd
import random
import numpy as np

# Seed for reproducibility
random.seed(0)

# Generate 50 rows of random data
data = {
    "Order_ID": [f"ORD{str(i).zfill(4)}" for i in range(1, 51)],
    "Customer_ID": [f"CUST{random.randint(100, 999)}" for _ in range(50)],
    "Product_ID": [f"PROD{random.randint(1, 50)}" for _ in range(50)],
    "Category": random.choices(["Electronics", "Apparel", "Home Decor", "Beauty", "Toys"], k=50),
    "Quantity": [random.randint(1, 10) for _ in range(50)],
    "Unit_Price": [round(random.uniform(5, 500), 2) for _ in range(50)],
    "Total_Price": lambda qty, price: [round(qty[i] * price[i], 2) for i in range(len(qty))],
    "Order_Date": pd.to_datetime(
        [f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for _ in range(50)]
    ),
    "Shipping_Cost": [round(random.uniform(0, 20), 2) for _ in range(50)],
    "Discount": [round(random.uniform(0, 0.3), 2) for _ in range(50)],
    "Payment_Method": random.choices(["Credit Card", "PayPal", "Debit Card", "Net Banking"], k=50),
    "Shipping_Status": random.choices(["Pending", "Shipped", "Delivered", "Returned"], k=50),
    "Customer_Rating": [random.choice([1, 2, 3, 4, 5]) for _ in range(50)],
}

# Calculate Total_Price as Quantity * Unit_Price
data["Total_Price"] = data["Total_Price"](data["Quantity"], data["Unit_Price"])

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = 'data/ecommerce_sales_data_2.csv'
df.to_csv(csv_path, index=False)

csv_path
