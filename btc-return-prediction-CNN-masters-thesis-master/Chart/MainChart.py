import pandas as pd
import mplfinance as mpf

# خواندن فایل CSV
D = pd.read_csv('C:/Users/Syber/Desktop/PayanNameh/code/btc-return-prediction-CNN-masters-thesis-master/Data/Bitfinex_BTCUSD_d fixed.csv')

# تبدیل ستون تاریخ به فرمت datetime
D['date'] = pd.to_datetime(D['date'])

# تنظیم ستون تاریخ به عنوان اندیس
D.set_index('date', inplace=True)

# اطمینان از وجود ستون volume
if 'Volume.USD' in D.columns:
    D.rename(columns={'Volume.USD': 'volume'}, inplace=True)

# انتخاب ستون‌های مورد نیاز
data = D[['open', 'high', 'low', 'close', 'volume']]

# رسم نمودار شمع ژاپنی با استفاده از mplfinance
mpf.plot(data, type='candle', style='charles', title='Bitcoin Price Over Time', ylabel='Price', volume=True, mav=(3,6,9), show_nontrading=True)

# نمایش نمودار
mpf.show()
