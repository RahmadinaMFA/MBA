import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

# load dataset
df = pd.read_csv("bread_basket.csv")
df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")
df["month"] = df['date_time'].dt.month
df["day"] = df['date_time'].dt.weekday

# Mengganti angka bulan dengan nama bulan
nama_bulan = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
df["month"] = df["month"].replace(list(range(1, 12 + 1)), nama_bulan)

# Mengganti angka hari dengan nama hari
nama_hari = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["day"] = df["day"].replace(list(range(7)), nama_hari)

st.title("Market Basket Analysis Menggunakan Algoritma Apriori")

def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data["period_day"].str.contains(period_day)) &
        (data["weekday_weekend"].str.contains(weekday_weekend)) &
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Result!"

def user_input_features():
    item = st.selectbox("Item", ['Bread ', 'Scandinavian', 'Hot choclolate', 'Jam', 'Cookie'])
    period_day = st.selectbox('Period Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    weekday_weekend = st.selectbox('Weekday / Weekend', ['Weekend', 'Weekday'])
    month = st.select_slider("Month", ["Jan", "Feb","Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = st.select_slider('Day',["Mon", "Tue", "Wed", "Thu", "Fri","Sat", "Sun"], value="Sat")

    return period_day, weekday_weekend, month, day, item


period_day, weekday_weekend, month, day, item = user_input_features()

data = get_data(period_day.lower(), weekday_weekend.lower(), month, day)

def encode(x):
    if x <= 0:
        return  0
    elif x >= 1:
        return 1

if type(data) != type("No Result"):
    item_count = data.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Count')
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    if item_count_pivot is not None:
        support = 0.01
        frequent_items_sorted = apriori(item_count_pivot, min_support=support, use_colnames=True)
    else:
        st.write("Data tidak tersedia atau format data tidak sesuai.")
else:
    st.write("Data tidak tersedia atau format data tidak sesuai.")

metric = "lift"
min_threshold = 1

rules = association_rules(frequent_items_sorted, metric=metric, min_threshold=min_threshold)[["antecedents","consequents", "support", "confidence", "lift"]]
rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)


if not isinstance(data, str) or data != "No Result!":
    st.markdown("Hasil Rekomendasi :")
    recommended_item = return_item_df(item)
    if recommended_item and len(recommended_item) > 0:
        if len(recommended_item) > 1:
            st.success(f"Jika Konsumen Membeli **{item}**, maka membeli ** {recommended_item[1]}** secara bersamaan")
        elif len(recommended_item) == 1:
            st.write(f"Hanya ada satu rekomendasi: {recommended_item[0]}")
    else:
        st.write("Tidak ada rekomendasi yang ditemukan atau hasil rekomendasi kosong.")
