from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from base import stockMongo
import numpy as np
import pandas as pd



MIN_PREDICTION_COUNT = 3000
columnsNotForLearning = ['Date', '_id', 'Symbol', 'Adj Low', 'Adj High', 'Adj Open', 'Adj Volume', 'Ex-Dividend', 'Split Ratio', 'index', 'level_0', 'result', 'nextClose', 'nextClosePercentage']


def checkPredictionCount(df, machineLearningMode, text="prediction"):
    if len(df) == 0 :
        return 0
    countResult = df.loc[df['_id'] == machineLearningMode.MODE]
    if len(countResult) > 0 :   
        if countResult['count'].values[0] <= MIN_PREDICTION_COUNT :
            print('{} for {} is less than {}'.format(text, machineLearningMode.MODE, MIN_PREDICTION_COUNT)) 
    else:
        print('{} for {} not found'.format(text, machineLearningMode.MODE))
    return countResult


def determineResult(threshold1, threshold2, value):
    if value > threshold1:
        return 1
    if value < threshold2:
        return -1
    return 0  


def getDefaultClassifier():
    return VotingClassifier([
            ('lsvc', svm.LinearSVC()),
            ('knn', neighbors.KNeighborsClassifier()),
            ('rfor', RandomForestClassifier())
            ])


def shiftValues(df, symbol, day, propertyName):
    df['{}_preDay{}_{}'.format(symbol, day, propertyName)] = df[propertyName].shift(-day)


def weaveInHistoryData(df, symbol, history_days):
    for i in range(1, history_days + 1):
        shiftValues(df, symbol, i, 'Close')
        shiftValues(df, symbol, i, 'Open')
        shiftValues(df, symbol, i, 'High')
        shiftValues(df, symbol, i, 'Low')
        shiftValues(df, symbol, i, 'Volume')


def removeUnusedColumnsAndRows(df):
    # remove unused columns
    df.drop(['Adj Low', 'Adj High', 'Adj Open', 'Adj Volume', 'Ex-Dividend', 'Split Ratio', 'index', 'level_0'], axis=1, inplace=True, errors='ignore')
    # replace string 'null' value with np.nan
    df = df.replace('null', np.nan)
    # remove rows contains nan
    # df = df.dropna(how='any')
    df.dropna(how='any', subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    return df


def addOtherQuoteData(df, symbol, days):
    endDate = df['Date'].max()
    startDate = df['Date'].min()
    quotes = stockMongo.findQuotesBySymbolPeriod(symbol, startDate, endDate)
    df1 = pd.DataFrame(list(quotes))
    df1 = removeUnusedColumnsAndRows(df1)
    weaveInHistoryData(df1, symbol, days)
    df1 = df1.rename(columns={'Open': symbol + '_Open'})
    df1 = df1.rename(columns={'Close': symbol + '_Close'})
    df1 = df1.rename(columns={'High': symbol + '_High'})
    df1 = df1.rename(columns={'Low': symbol + '_Low'})
    df1 = df1.rename(columns={'Volume': symbol + '_Volume'})
    
    cols_to_use = df1.columns.difference(df.columns)
    cols_to_use = cols_to_use.insert(0, 'Date')  # put back the join column
    df = pd.merge(df, df1[cols_to_use], on='Date', how='left')
    return df


def extract_X(df):
    return df.drop(columnsNotForLearning, axis=1, errors='ignore').values


def extract_y(df):
    return df['result'].values
