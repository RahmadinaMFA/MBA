import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit

data = pd.read_csv("bread_basket.csv")
print("DataFrame shape :",data.shape)
print(data)

#format data waktu
data['date_time'] = pd.to_datetime(data['date_time'], format="%d-%m-%Y %H:%M")

data["date_time"].dtype
data["month"] = data['date_time'].dt.month
data["day"] = data['date_time'].dt.weekday
data["hour"] = data['date_time'].dt.hour

#menampilkan 10 item paling laris
plt.figure(figsize=(13,5))
sns.set_palette("muted")

sns.barplot(x = data["Item"].value_counts()[:10].index,
            y=data["Item"].value_counts()[:10].values)
plt.xlabel(""); plt.ylabel("")
plt.xticks(size = 13, rotation = 45)
plt.title('10 produk yang paling laris', size = 17)
plt.show()

#transaksi tiap bulan
data_perbulan = data.groupby('month')['Transaction'].count()
data_perbulan = pd.concat([data_perbulan.iloc[4:], data_perbulan.iloc[:4]])

plt.figure(figsize= (8,5))
sns.barplot(
    x= ["October", "November", "December", "January", "February", "March", "April"],
    y= data_perbulan.values, color="#D5AAD3")
plt.xticks(size = 12, rotation = -30)
plt.title("jumlah transaksi tiap bulan dari oktober - april", size = 16)
plt.show()

#transaksi perhari
data_perday= data.groupby('day')['Transaction'].count()

plt.figure(figsize= (8,5))
sns.barplot(
    x= ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    y= data_perday.values, color="#BFFCC6")
plt.xticks(size = 12, rotation = -30)
plt.title("total transaksi perhari", size = 16)
plt.show()

#pembelian tiap jam
data_perhour = data.groupby('hour')['Transaction'].count()

plt.figure(figsize= (12,6))
sns.barplot(
    x= data_perhour.index,
    y= data_perhour.values, color= "#85E3FF")
plt.xlabel('Hour', size = 15)
plt.title("total transaksi perjam", size = 17)
plt.xticks(size = 13)
plt.show()

### data preparation
data["Item"] = data["Item"].apply(lambda item: item.lower())
data["Item"] = data["Item"].apply(lambda item: item.strip())

print(data.head(10))

from  mlxtend.frequent_patterns import  association_rules, apriori

item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
print(item_count.head(10))

item_count_privot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
print("Ukuran Dataset :", item_count_privot.shape)
print(item_count_privot.head())

item_count_privot = item_count_privot.astype("int32")
print(item_count_privot.head())

def encode(x):
    if x <=0:
        return 0
    elif x >= 1:
        return 1

item_count_privot =item_count_privot.applymap(encode)
print(item_count_privot.head())

print("Ukuran Dataset : ", item_count_privot.shape)
print("Jumlah Transaksi :", item_count_privot.shape[0])
print("jumlah items :", item_count_privot.shape[1])

support = 0.01
frequent_items_sorted  = apriori(item_count_privot, min_support= support, use_colnames=True)
print(frequent_items_sorted.head(10))

metric = "lift"
min_threshold = 1

rules = association_rules(frequent_items_sorted, metric='confidence', min_threshold=0.5)[["antecedents","consequents", "support", "confidence", "lift"]]
rules.sort_values('confidence', ascending=False, inplace=True)
print(rules.head(15))