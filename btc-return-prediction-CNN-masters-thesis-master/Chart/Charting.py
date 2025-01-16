import pandas as pd
import matplotlib.pyplot as plt

# خواندن فایل CSV
D = pd.read_csv('C:/Users/Syber/Desktop/PayanNameh/code/btc-return-prediction-CNN-masters-thesis-master/Data/Bitfinex_BTCUSD_d fixed.csv')

# تبدیل ستون تاریخ به فرمت datetime
D['date'] = pd.to_datetime(D['date'])

# تنظیم اندازه نمودار
plt.figure(figsize=(12, 6))

# رسم ستون‌های high و low به صورت خطی
plt.plot(D['date'], D['high'], color='Blue', label='High')
plt.plot(D['date'], D['low'], color='orange', linestyle='--', label='Low')

# رسم ستون‌های open و close به صورت bar با رنگ‌های مختلف
for idx, row in D.iterrows():
    color = 'green' if row['close'] > row['open'] else 'red'
    plt.bar(row['date'], row['close'] - row['open'], bottom=row['open'], color=color, width=0.5)

# تنظیم برچسب‌های محور x و y
plt.xlabel('Date')
plt.ylabel('Price($)')

# تنظیم عنوان نمودار
plt.title('Bitcoin Price Over Time')

# نمایش راهنما
plt.legend()

# نمایش نمودار
plt.show()
