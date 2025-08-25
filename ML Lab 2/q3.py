'''
A3. Please refer to the data present in “IRCTC Stock Price” data sheet of the above excel file. Do the 
following after loading the data to your programming platform. 
• Calculate the mean and variance of the Price data present in column D.  
(Suggestion: if you use Python, you may use statistics.mean() & 
statistics.variance() methods). 
• Select the price data for all Wednesdays and calculate the sample mean. Compare the mean 
with the population mean and note your observations. 
• Select the price data for the month of Apr and calculate the sample mean. Compare the 
mean with the population mean and note your observations. 
• From the Chg% (available in column I) find the probability of making a loss over the stock. 
(Suggestion: use lambda function to find negative values) 
• Calculate the probability of making a profit on Wednesday. 
• Calculate the conditional probability of making profit, given that today is Wednesday. 
• Make a scatter plot of Chg% data against the day of the week
'''
import pandas as pd
import matplotlib.pyplot as plt
import statistics

def load_stock_data(file_path, sheet_name="IRCTC Stock Price"):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def calculate_mean_variance(data):
    prices = data['Price'].tolist()
    mean_value = statistics.mean(prices)
    variance_value = statistics.variance(prices)
    return mean_value, variance_value

def calculate_wednesday_mean(data):
    wed_prices = data[data['Day'] == 'Wed']['Price'].tolist()
    if wed_prices:
        return statistics.mean(wed_prices)
    return None

def calculate_april_mean(data):
    apr_prices = data[data['Month'] == 'Apr']['Price'].tolist()
    if apr_prices:
        return statistics.mean(apr_prices)
    return None

def calculate_loss_probability(data):
    loss_probability = len(data[data["Chg%"].apply(lambda x: x < 0)]) / len(data)
    return loss_probability

def calculate_profit_probability_wednesday(data):
    wednesday_data = data[data["Day"] == "Wed"]
    if len(wednesday_data) == 0:
        return None
    profit_wed = wednesday_data[wednesday_data["Chg%"] > 0]
    return len(profit_wed) / len(wednesday_data)

def plot_scatter_day_vs_change(data):
    plt.scatter(data['Day'], data['Chg%'], alpha=0.7)
    plt.title("Chg% vs Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Chg%")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = r"C:\Users\iamre\OneDrive\Desktop\ML\Lab Session Data.xlsx"
    
    stock_data = load_stock_data(file_path)
    
    mean_price, variance_price = calculate_mean_variance(stock_data)
    mean_wed = calculate_wednesday_mean(stock_data)
    mean_apr = calculate_april_mean(stock_data)
    loss_prob = calculate_loss_probability(stock_data)
    profit_prob_wed = calculate_profit_probability_wednesday(stock_data)
    
    print(f"Mean: {mean_price}")
    print(f"Variance: {variance_price}")
    
    if mean_wed:
        print(f"Mean on Wednesdays: {mean_wed}")
        if mean_wed > mean_price:
            print("Observation: Wednesday's average price is higher than the population average.")
        elif mean_wed < mean_price:
            print("Observation: Wednesday's average price is lower than the population average.")
        else:
            print("Observation: Wednesday's average price equals the population average.")
    
    if mean_apr:
        print(f"Mean in April: {mean_apr}")
        if mean_apr > mean_price:
            print("Observation: April's average price is higher than the population average.")
        elif mean_apr < mean_price:
            print("Observation: April's average price is lower than the population average.")
        else:
            print("Observation: April's average price equals the population average.")
    
    print(f"Probability of Loss: {loss_prob:.2f}")
    if profit_prob_wed is not None:
        print(f"Probability of Profit on Wednesday: {profit_prob_wed:.2f}")
        print(f"Conditional Probability of Profit given it's Wednesday: {profit_prob_wed:.2f}")
    
    plot_scatter_day_vs_change(stock_data)
