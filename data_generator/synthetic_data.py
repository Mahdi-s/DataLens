import pandas as pd
import random
import numpy as np

# Seed for reproducibility
random.seed(0)

# # Generate 50 rows of random data
# data = {
#     "Order_ID": [f"ORD{str(i).zfill(4)}" for i in range(1, 51)],
#     "Customer_ID": [f"CUST{random.randint(100, 999)}" for _ in range(50)],
#     "Product_ID": [f"PROD{random.randint(1, 50)}" for _ in range(50)],
#     "Category": random.choices(["Electronics", "Apparel", "Home Decor", "Beauty", "Toys"], k=50),
#     "Quantity": [random.randint(1, 10) for _ in range(50)],
#     "Unit_Price": [round(random.uniform(5, 500), 2) for _ in range(50)],
#     "Total_Price": lambda qty, price: [round(qty[i] * price[i], 2) for i in range(len(qty))],
#     "Order_Date": pd.to_datetime(
#         [f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for _ in range(50)]
#     ),
#     "Shipping_Cost": [round(random.uniform(0, 20), 2) for _ in range(50)],
#     "Discount": [round(random.uniform(0, 0.3), 2) for _ in range(50)],
#     "Payment_Method": random.choices(["Credit Card", "PayPal", "Debit Card", "Net Banking"], k=50),
#     "Shipping_Status": random.choices(["Pending", "Shipped", "Delivered", "Returned"], k=50),
#     "Customer_Rating": [random.choice([1, 2, 3, 4, 5]) for _ in range(50)],
# }
num_rows = 1000
data = {
    "Order_ID": [f"ORD{str(i).zfill(6)}" for i in range(1, num_rows + 1)],
    "Customer_ID": [f"CUST{random.randint(100, 9999)}" for _ in range(num_rows)],
    "Product_ID": [f"PROD{random.randint(1, 500)}" for _ in range(num_rows)],
    "Category": random.choices(["Electronics", "Apparel", "Home Decor", "Beauty", "Toys", 
                              "Books", "Sports", "Food", "Furniture", "Automotive"], k=num_rows),
    "Quantity": [random.randint(1, 20) for _ in range(num_rows)],
    "Unit_Price": [round(random.uniform(5, 2000), 2) for _ in range(num_rows)],
    "Total_Price": lambda qty, price: [round(qty[i] * price[i], 2) for i in range(len(qty))],
    "Order_Date": pd.to_datetime(
        [f"{random.randint(2022,2024)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}" 
         for _ in range(num_rows)]
    ),
    "Shipping_Cost": [round(random.uniform(0, 50), 2) for _ in range(num_rows)],
    "Discount": [round(random.uniform(0, 0.5), 2) for _ in range(num_rows)],
    "Payment_Method": random.choices(["Credit Card", "PayPal", "Debit Card", "Net Banking", 
                                    "Crypto", "Bank Transfer", "COD"], k=num_rows),
    "Shipping_Status": random.choices(["Pending", "Shipped", "Delivered", "Returned", 
                                     "Cancelled", "Processing"], k=num_rows),
    "Customer_Rating": [random.choice([1, 2, 3, 4, 5, None]) for _ in range(num_rows)],
}

# Calculate Total_Price as Quantity * Unit_Price
data["Total_Price"] = data["Total_Price"](data["Quantity"], data["Unit_Price"])

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = 'data/ecommerce_sales_data_3.csv'
df.to_csv(csv_path, index=False)

csv_path
