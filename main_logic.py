import datetime
import requests
import pandas as pd
import pickle
import math

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from finvizfinance.quote import finvizfinance

from helper import *

# TODO: Create a file to store variables
MARKET_OPEN_TIME = datetime.datetime.strptime("09:31:00", "%H:%M:%S")
MARKET_CLOSE_TIME = datetime.datetime.strptime("15:38:00", "%H:%M:%S")
ANALYZER = SentimentIntensityAnalyzer()
STOCK_API_TOKEN = "YOUR_API_TOKEN"
cnt_ = 0

# Generate news
def generate_news_df():
    companies_set = generate_sp500_symbol_set()
    time_cut = pd._libs.tslibs.Timestamp('2020-10-26 11:59:00')

    df_list = []
    for i, company in enumerate(companies_set):
        try:
            news_df = finvizfinance(company).TickerNews()
            if(min(news_df["Date"]) >time_cut):
                news_df = finvizfinance(company).TickerNews()
                symbol_list = [company] * len(news_df.index)
                news_df.insert(0, 'symbol', symbol_list)
                df_list.append(news_df)
        except Exception as e:
            print(e)
            continue
    df = pd.concat(df_list).reset_index(drop=True)
    
    # Create datetime str for later use
    date_str = [dt.strftime("%Y-%m-%d") for dt in df['Date']]
    df.insert(1, 'Date_str', date_str)

    return df

# Perform sentiment analysis
def perform_sentiment_analysis(news_df, analyzer, keyword='Title'):
    neg = []
    neu = []
    pos = []
    compound = []

    for t in news_df[keyword]:
        score = analyzer.polarity_scores(t)
        neg.append(score['neg'])
        neu.append(score['neu'])
        pos.append(score['pos'])
        compound.append(score['compound'])
    
    news_df['Sentiment_Neg'] = neg
    news_df['Sentiment_Neu'] = neu
    news_df['Sentiment_Pos'] = pos
    news_df['Sentiment_Compound'] = compound

# Three cases
# Published in trading hours
#   - publish_price = price at that minute
#   - price_after = price after 24 hours
#       - Possible case: target day is not trading day, solution for now: discard this record
# Published before market open time
#   - publish_price = today's open price
#   - price_after = day + lag day's close price
# Published after market close time
#   - price_publish = tomorrow's open price
#   - price_after = tomorrow + lag day's close price
#       - Possible case: target day is not trading day, solution for now: discard this record
def calculate_stock_performance(df, stock_api_token, stock_his_dict, lag_str='1m', keyword_for_date='Date', use_news_notin_tradingtime=False):
    global cnt_

    def get_intraday_list(date, symbol):
        global cnt_

        if check_stock_his_by_keyword(stock_his_dict, symbol, date, "intraday"):
            intra_list = stock_his_dict[symbol][date]["intraday"]
        else:
            intra_list = get_intra_by_date(symbol, stock_api_token, date.replace("-", ""))
            create_datapoint_in_stock_his(stock_his_dict, symbol, date)
            stock_his_dict[symbol][date]["intraday"] = intra_list
            cnt_ += 1
        return intra_list
    
    def get_general_list(date, symbol):
        global cnt_

        if check_stock_his_by_keyword(stock_his_dict, symbol, date, "general"):
            general_list = stock_his_dict[symbol][date]['general']
        else:
            general_list = get_general_info_by_date(symbol, stock_api_token, date.replace("-", ""))
            create_datapoint_in_stock_his(stock_his_dict, symbol, date)
            stock_his_dict[symbol][date]['general'] = general_list
            cnt_ += 1
        return general_list

    def get_intra_average_value(intra_list, index):
        if 'marketAverage' in intra_list[index].keys(): return intra_list[index]['marketAverage']
        if 'average' in intra_list[index].keys(): return intra_list[index]['average']
        return -1
        
    price_publish_list = []
    price_after_list = []
    used = []

    prev_cnt = cnt_
    # Test only 
    # for t in df:
    for t, symbol in zip(df[keyword_for_date], df['symbol']):
        if isinstance(t, datetime.datetime):
            t = t.strftime("%Y-%m-%d %H:%M:%S")
        date_str, t_ = t.split(" ")

        # if cnt_ != prev_cnt:
        #     prev_cnt = cnt_
        #     print(cnt_)
        if cnt_ % 1000 == 0 and cnt_ != 0:
            print("Save stock his dict" + "." * 100)
            pickle.dump(stock_his_dict, open("./data/stock_his_dict.p", "wb"))
            cnt_ += 1
            
        # TODO: Write a function to process date
        # Since the format might change in the future
        # Remove -
        publish_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        publish_time = datetime.datetime.strptime(t_, "%H:%M:%S")

        # Calculate lag
        lag_day, lag_minute = process_lag(lag_str)

        # Calculate target day's date
        after_date = publish_date + datetime.timedelta(days=lag_day)
        # print(after_date == publish_date, lag_day)
        after_date_str = datetime.datetime.strftime(after_date, "%Y-%m-%d")

        if MARKET_OPEN_TIME < publish_time < MARKET_CLOSE_TIME:
            publish_intra_list = get_intraday_list(date_str, symbol)
            after_intra_list = publish_intra_list if after_date == publish_date else get_intraday_list(after_date_str, symbol)

            # If publish date or target date is not trading day, skip for now
            if len(publish_intra_list) == 0 or len(after_intra_list) == 0:
                price_publish_list.append(-1)
                price_after_list.append(-1)
                used.append(False)
                continue

            index_publish = publish_time.hour * 60 + publish_time.minute - 570
            index_after = index_publish + lag_minute

            # There are only 390 trading minutes within a trading day
            if index_publish > len(publish_intra_list) - 1 or index_after > len(after_intra_list) - 1:
                print(index_publish, index_after)
                price_publish_list.append(-1)
                price_after_list.append(-1)
                used.append(False)
                continue

            price_publish = get_intra_average_value(publish_intra_list, index_publish)
            price_after = get_intra_average_value(after_intra_list, index_after)

            price_publish_list.append(price_publish if price_publish else -1)
            price_after_list.append(price_after if price_after else -1)
            if price_publish != -1 and price_after != -1 and price_publish and price_after: used.append(True)
            else: used.append(False)
            continue

        # Not published in the trading hours, but we decide to use it
        if use_news_notin_tradingtime and lag_day > 0:
            # We only use open and close price - i.e. 1d lag - same day's open and close price
            after_date -= datetime.timedelta(days=1)
            after_date_str = datetime.datetime.strftime(after_date, "%Y-%m-%d")

            if publish_time >= MARKET_CLOSE_TIME:
                publish_date += datetime.timedelta(days=1)
                date_str = datetime.datetime.strftime(publish_date, "%Y-%m-%d")

                after_date += datetime.timedelta(days=1)
                after_date_str = datetime.datetime.strftime(after_date, "%Y-%m-%d")

            publish_general_list = get_general_list(date_str, symbol)
            after_general_list = publish_general_list if after_date == publish_date else get_general_list(after_date_str, symbol)

            # If not in the trade day, skip for now
            if len(publish_general_list) == 0 or len(after_general_list) == 0: 
                price_publish_list.append(-1)
                price_after_list.append(-1)
                used.append(False)
                continue

            price_publish = publish_general_list[0]['open']
            price_after = after_general_list[0]['close']

            price_publish_list.append(price_publish)
            price_after_list.append(price_after)
            used.append(True)
            continue
        else:
            price_publish_list.append(-1)
            price_after_list.append(-1)
            used.append(False)
            continue

    df_publish_list = df['price_publish']
    df_publish_list = [v2 if v1 == -1 else v1 for v1, v2 in zip(df_publish_list, price_publish_list)]
    df['price_publish'] = df_publish_list
    df['price_after_' + lag_str] = price_after_list
    df['used_' + lag_str] = used

def calculate_stock_performance_with_different_lag(total_df, STOCK_API_TOKEN, stock_his_dict, lag_strs=['1m', '5m', '1h', '1d', '7d']):
    for lag_str in lag_strs:
        calculate_stock_performance(total_df, STOCK_API_TOKEN, stock_his_dict, lag_str=lag_str, keyword_for_date='Date', use_news_notin_tradingtime=True)

def compress_df_by_date(df, lag_str):
    used_keyword = "used_" + lag_str
    price_after_keyword = "price_after_" + lag_str
    compressed_data = []

    columns_to_keep = ['symbol', 'Date_str', 'Sentiment_Neg', "Sentiment_Neu", "Sentiment_Pos", "Sentiment_Compound", "price_publish", price_after_keyword, used_keyword]

    for symbol, df_1 in df.groupby("symbol"):
        for date_str, df_2 in df_1.groupby("Date_str"):
            df_2 = df_2[df_2[used_keyword]]
            if len(df_2.index) == 0: continue
            l = [symbol, date_str]
            for cl in columns_to_keep[2:-1]: l.append(df_2[cl].mean())
            l.append(True)
            compressed_data.append(l)
    
    return pd.DataFrame(compressed_data, columns=columns_to_keep)

def calculate_metric_within_oneday(df, lag_strs=['1m', '5m', '1h']):
    for lag_str in lag_strs:
        price_publish_key = "price_publish"
        price_after_key = "price_after_" + lag_str
        used_key = "used_" + lag_str
        performance_key = "performance_" + lag_str
        performance_key_scaled = "performance_" + lag_str + "_scaled"

        performance_list = []
        performance_valid_list = []
        for pp, pa, u in zip(df[price_publish_key], df[price_after_key], df[used_key]):
            if not u: 
                performance_list.append(-1)
                continue
            performance = math.log(pa/pp)
            # Debug use
            if math.isnan(performance): print(pa, pp)
            
            performance_list.append(performance)
            performance_valid_list.append(performance)
        # Scale the list between -1 and 1
        performance_valid_array_scaled = scale_data(performance_valid_list)

        performance_scaled_list = []
        cnt = 0
        for p in performance_list:
            if p == -1: performance_scaled_list.append(-1)
            else: 
                performance_scaled_list.append(performance_valid_array_scaled[cnt])
                cnt += 1
        df[performance_key] = performance_list
        df[performance_key_scaled] = performance_scaled_list