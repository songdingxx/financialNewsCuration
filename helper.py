import datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from texttable import Texttable

def benchmark(func, args):
    
    start_time = datetime.datetime.now()
    
    if type(args) is tuple: res = func(*args)
    else : res = func(args)

    run_time = (datetime.datetime.now() - start_time).total_seconds()

    print("Run time Cost is : {}ms".format(round(run_time * 1000, 2)))

    return res

def generate_sp500_symbol_set():
    sp500_table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500 = sp500_table[0]['Symbol']
    res = set()
    for s in sp500: res.add(s.lower())
    return res

def get_intra_by_date(symbol, api_token, date_str):
    url = "https://cloud.iexapis.com/stable/stock/{symbol}/intraday-prices?token={api_token}&exactDate={date_str}".format(**locals())
    print(url)
    res = requests.get(url)
    data = res.json()
    return data

def get_general_info_by_date(symbol, api_token, date_str):
    url = "https://cloud.iexapis.com/stable/stock/{symbol}/chart/date/{date_str}?chartByDay=true&token={api_token}".format(**locals())
    print(url)
    res = requests.get(url)
    data = res.json()
    return data

def check_stock_his_by_keyword(stock_his_dict, symbol, date, keyword):
    return (symbol in stock_his_dict.keys() 
            and date in stock_his_dict[symbol].keys() 
            and keyword in stock_his_dict[symbol][date].keys())

def create_datapoint_in_stock_his(stock_his_dict, symbol, date):
    if symbol not in stock_his_dict.keys(): stock_his_dict[symbol] = {}
    if date not in stock_his_dict[symbol].keys(): stock_his_dict[symbol][date] = {}

# TODO: for now, only support xdyhzm (i.e. 4d1h3m) format.
def process_lag(lag_str):
    time_day = 0
    time_min = 0

    if 'd' in lag_str:
        time_str = lag_str.split("d")
        time_day = int(time_str[0])
        lag_str = time_str[1]
    if 'h' in lag_str:
        time_str = lag_str.split('h')
        time_min += 60 * int(time_str[0])
        lag_str = time_str[1]
    if 'm' in lag_str:
        time_str = lag_str.split('m')
        time_min += int(time_str[0])

    return time_day, time_min

def compute_confusion_matrix(df, lag_strs,sentiment_keyword='Sentiment_Compound', remove_outlier=True, outlier_range=3):
    res = ""
    if remove_outlier: res += "Outlier removed, range: {}\n".format(outlier_range)
    else: res += "Outlier kept\n"
    res += "=" * 100 + "\n"
    for lag_str in lag_strs:
        t = Texttable()

        used_keyword = "used_{}".format(lag_str)
        performance_keyword = "performance_{}".format(lag_str)

        tp, fp, tn, fn = 0, 0, 0, 0
        df_ = df[df[used_keyword]]
        # Remove outlier
        if remove_outlier: df_ = df_[mark_outlier(df_[performance_keyword], outlier_range)]

        for p, s in zip(df_[performance_keyword], df_[sentiment_keyword]):
            if p > 0 and s > 0: tp += 1
            elif p > 0 and s < 0: fp += 1
            elif p < 0 and s > 0: fn += 1
            elif p < 0 and s < 0: tn += 1
        res += "{}\n".format(performance_keyword)
        t.add_row(["", "Sentiment Positive", "Sentiment Negative"])
        t.add_row(["Performance Positive", str(tp), str(fp)])
        t.add_row(["Performance Negative", str(fn), str(tn)])
        res += t.draw() + "\n\n"
    filename = "./data/cm"
    filename += "_{}".format(sentiment_keyword)
    if remove_outlier: filename += "_remove_outlier"
    else: filename += "_keep_outlier"
    filename += "_["
    for lag_str in lag_strs: filename += lag_str+","
    filename = filename[:-1] + "].txt"
    with open(filename, "w") as f:
        f.write(res)
    print(res)

def draw_figure_by_date(compressed_df, use_scaled=True, scaled_ratio=1, lag_str='1d', limit=-1):
    fig, ax = plt.subplots(1, figsize=(20, 10))
    ax2=ax.twinx()

    plt.style.use("bmh")

    if use_scaled: metric_keyword = "performance_{}_scaled".format(lag_str)
    else: metric_keyword = "performance_{}".format(lag_str)

    companies_count = []
    sentiment_neg = []
    sentiment_neu = []
    sentiment_pos = []
    sentiment_compound = []
    performance_scaled = []
    date_gb = compressed_df.groupby("Date_str")

    for _, df in date_gb:
        if len(df.index) < limit: continue
        companies_count.append(len(df.index))
        sentiment_neg.append(df['Sentiment_Neg'].mean())
        sentiment_pos.append(df['Sentiment_Pos'].mean())
        sentiment_neu.append(df['Sentiment_Neu'].mean())
        sentiment_compound.append(df['Sentiment_Compound'].mean())
        performance_scaled.append(df[metric_keyword].mean() * scaled_ratio)
    X = [i for i in range(1, len(companies_count) + 1)]
    

    ax.plot(X, sentiment_compound, label='Sentiment Compound')
    ax.plot(X, performance_scaled, label=metric_keyword.format(lag_str))
    ax.set_ylim(-0.5, 1)
    ax2.bar(X, companies_count, label='Companies Count', color='g')
    ax2.set_ylim(0, 1000)
    ax.legend(loc="upper left")
    ax2.legend(loc='upper right')
    
    filename = "./pics/trend_{}_{}_limit={}_scale={}.png".format(metric_keyword, lag_str, limit, scaled_ratio)
    plt.savefig(filename)

def draw_plot(df, lag_strs, sentiment_keyword = "Sentiment_Compound", use_scaled_data=True, remove_zero_sent=False):
    length = len(lag_strs)
    # n_cols = math.floor(len(lag_strs) / 2)
    fig, ax = plt.subplots(1, len(lag_strs), figsize=(10 * len(lag_strs), 10))

    for i, lag in enumerate(lag_strs):
        index_row = int(i / 2)
        index_col = (i + 1) % 2
        if length > 1: ax_temp = ax[i]
        else: ax_temp = ax
        ax_temp.set_title("Correlation map - {}, {}".format(sentiment_keyword, lag), fontsize=15)

        if use_scaled_data: metric_keyword = "performance_{}_scaled".format(lag)
        else: metric_keyword = "performance_{}".format(lag)

        used_keyword = "used_{}".format(lag)

        df_ = df[df[used_keyword]]

        # remove outlier
        df_=df_[mark_outlier(df_[metric_keyword])]
        if use_scaled_data:
            df_[metric_keyword] = scale_data(df_[metric_keyword])

        if remove_zero_sent: 
            df_ = df_[df_[sentiment_keyword] != 0]
            df_[metric_keyword] = scale_data(df_[metric_keyword])

        ax_temp.scatter(df_[sentiment_keyword], df_[metric_keyword], s=5, c='r')
        ax_temp.set_xlabel(sentiment_keyword)
        ax_temp.set_ylabel(metric_keyword)
        ax_temp.set_xlim(-1, 1)
        if use_scaled_data: ax_temp.set_ylim(-1, 1)

    filename = "./pics/{}".format(sentiment_keyword)
    if use_scaled_data: filename += "_scaled"
    else: filename += "_notScaled"
    if remove_zero_sent: filename += "_removeZeroSent"
    else: filename += "_keepZeroSent"
    filename += "_["
    for lag_str in lag_strs: filename += lag + ","
    filename = filename[:-1] + "]."
    filename += "png"

    plt.savefig(filename)

def scale_data(l, outlier_range=3):
    a = np.array(list(l))
    a_scaled = 2 * (a - a.min()) / (a.max() - a.min()) - 1
    return a_scaled

def mark_outlier(l, outlier_range=3):
    a = np.array(list(l))
    # remove outliers
    mean_ = np.mean(a)
    std_ = np.std(a)
    return (mean_ - outlier_range * std_ < a) & (a < mean_ + outlier_range * std_)