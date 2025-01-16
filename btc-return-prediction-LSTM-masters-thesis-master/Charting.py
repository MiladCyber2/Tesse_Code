import pandas as pd
import matplotlib.pyplot as plt


# خواندن فایل CSV
D = pd.read_csv('C:/Users/Syber/Desktop/PayanNameh/code/btc-return-prediction-LSTM-masters-thesis-master/Data/Blockchain/bitcoinity_data_btc_dailytransactionsonnetwork.csv')

# رسم نمودار خطی ساده
plt.plot(D['Time'], D['Value'] , c='Orange')
plt.xlabel(' X')
plt.ylabel(' Y')
plt.title('Title')
plt.legend(['sin(x)'])
plt.show()

# یا برای نمودار میله‌ای
#plt.bar(df['Time'], df['Value'])
#plt.show()

# برای نمودار پراکندگی
#plt.scatter(df['Time'], df['Value'])
#plt.show()
