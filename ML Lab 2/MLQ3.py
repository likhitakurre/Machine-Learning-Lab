import pandas as pd
import statistics
import matplotlib.pyplot as plt

# Load the Excel file
file = "Lab Session Data.xlsx"
df = pd.read_excel(file, sheet_name="IRCTC Stock Price")

# ---- Clean & Prepare Data ----
# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract weekday and month
df['Weekday'] = df['Date'].dt.day_name()
df['Month'] = df['Date'].dt.month_name()

# 1️⃣ Mean and Variance of Column D (Close Price)
price_data = df.iloc[:, 3]  # Column D is the 4th column (0-indexed)
price_mean = statistics.mean(price_data)
price_var = statistics.variance(price_data)

print("Mean of Close Price (Population):", price_mean)
print("Variance of Close Price:", price_var)

# 2️⃣ Sample Mean for Wednesdays
wednesday_prices = df[df['Weekday'] == 'Wednesday'].iloc[:, 3]
wednesday_mean = statistics.mean(wednesday_prices)

print("Mean of Close Price on Wednesdays:", wednesday_mean)
print("Observation: Wednesday mean is", "higher" if wednesday_mean > price_mean else "lower", "than population mean.")

# 3️⃣ Sample Mean for April
april_prices = df[df['Month'] == 'April'].iloc[:, 3]
april_mean = statistics.mean(april_prices)

print("Mean of Close Price in April:", april_mean)
print("Observation: April mean is", "higher" if april_mean > price_mean else "lower", "than population mean.")

# 4️⃣ Probability of Loss (Chg% < 0)
chg_data = df.iloc[:, 8]  # Column I is the 9th column (0-indexed)
loss_prob = (chg_data < 0).sum() / len(chg_data)

print("Probability of making a loss:", round(loss_prob, 4))

# 5️⃣ Probability of Profit on Wednesday
wed_data = df[df['Weekday'] == 'Wednesday']
profit_wed = wed_data[wed_data.iloc[:, 8] > 0]
profit_wed_prob = len(profit_wed) / len(wed_data)

print("Probability of profit on Wednesday:", round(profit_wed_prob, 4))

# 6️⃣ Conditional Probability P(Profit | Wednesday)
print("Conditional Probability of Profit given Wednesday:", round(profit_wed_prob, 4))

# 7️⃣ Scatter Plot: Chg% vs Day of Week
plt.figure(figsize=(10, 6))
plt.scatter(df['Weekday'], df.iloc[:, 8], alpha=0.6, color='blue')
plt.title("Chg% vs Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Change (%)")
plt.grid(True)
plt.show()
