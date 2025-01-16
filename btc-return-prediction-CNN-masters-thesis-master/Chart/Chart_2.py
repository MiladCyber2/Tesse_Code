import pandas as pd
import matplotlib.pyplot as plt

# خواندن فایل CSV
D = pd.read_csv('C:/Users/Syber/Desktop/PayanNameh/code/btc-return-prediction-CNN-masters-thesis-master/Data/Bitfinex_BTCUSD_d fixed.csv')

# تبدیل ستون تاریخ به فرمت datetime
D['date'] = pd.to_datetime(D['date'])

# تنظیم اندازه نمودار
plt.figure(figsize=(12, 6))

# رسم ستون‌های open و close به صورت bar
plt.bar(D['date'], D['open'], color='blue', width=0.5, label='Open')
plt.bar(D['date'], D['close'], color='green', width=0.3, label='Close')

# رسم ستون‌های high و low به صورت خطی
plt.plot(D['date'], D['high'], color='red', label='High')
plt.plot(D['date'], D['low'], color='orange', linestyle='--', label='Low')

# تنظیم برچسب‌های محور x و y
plt.xlabel('Date')
plt.ylabel('Price')

# تنظیم عنوان نمودار
plt.title('Bitcoin Price Over Time')

# نمایش راهنما
plt.legend()

# نمایش نمودار
plt.show()
