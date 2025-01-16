import pandas as pd
import matplotlib.pyplot as plt

# خواندن فایل CSV
D = pd.read_csv('C:/Users/Syber/Desktop/PayanNameh/code/btc-return-prediction-CNN-masters-thesis-master/Data/Bitfinex_BTCUSD_d fixed.csv')

# تبدیل ستون تاریخ به فرمت datetime
D['date'] = pd.to_datetime(D['date'])

# تنظیم اندازه نمودار
plt.figure(figsize=(12, 6))

# رسم ستون‌های open و close با یک رنگ
plt.plot(D['date'], D['open'], color='blue', label='Open')
plt.plot(D['date'], D['close'], color='Orange', label='Close')

# رسم ستون‌های high و low با رنگ دیگر
plt.plot(D['date'], D['high'], color='green', linestyle='--', label='High')
plt.plot(D['date'], D['low'], color='red', linestyle='--', label='Low')

# تنظیم برچسب‌های محور x و y
plt.xlabel('Date')
plt.ylabel('Price')

# تنظیم عنوان نمودار
plt.title('Bitcoin Price Over Time')

# نمایش راهنما
plt.legend()

# نمایش نمودار
plt.show()
