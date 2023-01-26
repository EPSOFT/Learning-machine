import functools

from sklearn.model_selection import train_test_split

from base import dateUtil
from base import fileUtil
from base import stockMongo
from base.parallel import runToAllDone
import numpy as np
import pandas as pd


def removeUnusedColumnsAndRows(df):
    # remove unused columns
    df.drop(['Adj Low', 'Adj High', 'Adj Open', 'Adj Volume', 'Ex-Dividend', 'Split Ratio', 'index', 'level_0'], axis=1, inplace=True, errors='ignore')
    # replace string 'null' value with np.nan
    df = df.replace('null', np.nan)
    df.dropna(how='any', subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    return df


def getQuoteData(machineLearningMode, startDate, endDate, symbol, history_days=0):
    quotes = stockMongo.findLatestQuotesPeriod(symbol, machineLearningMode.QUOTES_NUMBER, startDate, endDate)
    quotes = list(quotes)
    
    if history_days is not None and history_days > 0:
        quotes1 = stockMongo.findLatestQuotesPeriod(symbol, history_days, None, startDate, includeEndDate=False)
        quotes.extend(list(quotes1))
    
    df = pd.DataFrame(list(quotes))
    
    if len(df) == 0:
        return;

    df = removeUnusedColumnsAndRows(df)
    return df


def moreThanOneClassResult(array):
    if array is None:
        return False
    uniqData = set(array)
    return len(uniqData) > 1


def getLastRowDate(df):
    return df['Date'].values[-1]


def getPickleName(symbol, mode):
    return symbol + "_" + mode


def initialMachineLearning(machineLearningMode, startDate, endDate, symbol):
    quotes = getQuoteData(machineLearningMode, startDate, endDate, symbol)
    if quotes is None  or len(quotes) < machineLearningMode.MINIMUN_MACHINE_LEARNING_NUMBERS:
        return

    X, y, df = machineLearningMode.extract_featureset(quotes)
    
    if not moreThanOneClassResult(y):
        print("only has 1 classes result, no training...")
        return
    
    lastRecordDate = getLastRowDate(df)
    
    clf = machineLearningMode.getClassifier()
    clf.fit(X, y)  # use full data for training
    data = {"lastRecordDate": lastRecordDate, "clf": clf}
    fileUtil.pickleIt(getPickleName(symbol, machineLearningMode.MODE), data)
    print(symbol, ' machine learning is done...')  


def getClassifierAccuracy(machineLearningMode, startDate, endDate, symbol):
    quotes = getQuoteData(machineLearningMode, startDate, endDate, symbol)
    if quotes is None or len(quotes) < machineLearningMode.MINIMUN_MACHINE_LEARNING_NUMBERS :
        return None, 0
    
    X, y, df = machineLearningMode.extract_featureset(quotes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=machineLearningMode.TEST_SIZE)
    
    if not moreThanOneClassResult(y_train):
        print("only has 1 classes result, no training...")
        return None, 0
    
    clf = machineLearningMode.getClassifier()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    return confidence, len(X_train)


def saveAccuracy(machineLearningMode, startDate, endDate, symbol):
    try:
        confidence, trainNumber = getClassifierAccuracy(machineLearningMode, startDate, endDate, symbol)
        print(confidence, trainNumber)
        if trainNumber > 0 : 
            stockMongo.saveLearnAccuracy(symbol, machineLearningMode.MODE, confidence, trainNumber)
    except Exception as e:
        print('...error while saveAccuracy for ', symbol)
        print(str(e))
        print('.............................')


def quotePredict(machineLearningMode, symbol, X, dates): 
    if X is None or len(X) == 0 :
#         print('no featureset data for ', symbol)
        return
    
    ml_data = fileUtil.loadPickle(getPickleName(symbol, machineLearningMode.MODE))
    if ml_data is None:
        return
    
    prediction = ml_data['clf'].predict(X)
    # TODO get minimun date and it shall be great than the machine learning date
    return list(zip(prediction, dates))


def predict(machineLearningMode, date, symbol):
    startDate = date
    endDate = date
    quotes = getQuoteData(machineLearningMode, startDate, endDate, symbol, history_days=machineLearningMode.QUOTE_HISTORY_DAYS)
    X, y, df = machineLearningMode.extract_featureForPredict(quotes, startDate, endDate)  
    return quotePredict(machineLearningMode, symbol, X, df['Date'].values)


def verifyQuote(machineLearningMode, startDate, endDate, symbol):
    print("verifyQuote", symbol)
    history_days = 10
    quotes = getQuoteData(machineLearningMode, startDate, endDate, symbol, history_days=history_days)
    X, y, df = machineLearningMode.extract_featureset(quotes)
    predictions = quotePredict(machineLearningMode, symbol, X, df['Date'].values)
        
    # verify if prediction is correct
    df['result'] = list(map(machineLearningMode.determineResult, df['nextClosePercentage']))
    prediction_df = pd.DataFrame(predictions, columns=['prediction', 'Date'])
    # qualified is prediction either 1 or -1
    prediction_df = pd.merge(prediction_df, df[['Date', 'result']], on='Date')
    
    # save prediction
    for index, row in prediction_df.iterrows():
        predictObj = {"Symbol":symbol, "Date":row['Date'], "Prediction":int(row['prediction']), "Mode":machineLearningMode.MODE, "isCorrect": (row['prediction'] == row['result']), "Result":row['result']}
        stockMongo.savePrediction(predictObj)
    
    # only care None-Zero result
    accuracy_prediction_df = prediction_df.loc[(prediction_df['result'] != 0) | (prediction_df['prediction'] != 0)]
    correct_prediction_df = accuracy_prediction_df.loc[accuracy_prediction_df['result'] == accuracy_prediction_df['prediction']]
    if len(accuracy_prediction_df) == 0 :
        print("{} prediction_df has no Non-Zero data".format(symbol))
        return
    
    result = len(correct_prediction_df) / len(accuracy_prediction_df)
    if result > 0.75 :
        print(result)
                

# the data before sepDate will be used as machine learning data
# the data after sepDate will be used for verification
def learn(machineLearningMode, sepDate, symbols=[]):
    if not sepDate:
        sepDate = dateUtil.toString(dateUtil.addDays(dateUtil.nowString(), -45))  #-45 days is roughly two months
    if not symbols:
        symbols = pd.DataFrame(list(stockMongo.findAllActiveSymbols()))['Symbol'].values
#     symbols = ['KO']
    learningFunction = functools.partial(initialMachineLearning, machineLearningMode, None, sepDate)
    runToAllDone(learningFunction, [(symbol,) for symbol in symbols])        
           
    saveAccuracyFunc = functools.partial(saveAccuracy, machineLearningMode, None, sepDate)
    runToAllDone(saveAccuracyFunc, [(symbol,) for symbol in symbols])
        
    # verify the prediction of the rest data
    startDate = sepDate
    endDate = None
    runToAllDone(functools.partial(verifyQuote, machineLearningMode, startDate, endDate), [(symbol,) for symbol in symbols])
    

        
