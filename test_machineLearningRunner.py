import random
from unittest import mock

from machinelearning import machineLearningRunner
import pandas as pd


def test_removeUnusedColumnsAndRows():
    df = pd.DataFrame([{'Adj Low':0, 'Adj High':1, 'Adj Open':2, 'Adj Volume':3, 'Ex-Dividend':4, 'Split Ratio':5, 'index':6, 'level_0':7, 'otherField':8, 'Open':9, 'High':10, 'Low':11, 'Close':12, 'Volume':13},
                       {'Adj Low':0, 'Adj High':1, 'Adj Open':2, 'Adj Volume':3, 'Ex-Dividend':4, 'Split Ratio':5, 'index':6, 'level_0':7, 'otherField':8, 'Open':9, 'High':10, 'Low':11, 'Close':12}])
    df = machineLearningRunner.removeUnusedColumnsAndRows(df)
    assert len(df) == 1
    assert not {'Adj Low', 'Adj High', 'Adj Open', 'Adj Volume', 'Ex-Dividend', 'Split Ratio', 'index', 'level_0'}.issubset(df.columns)
    
    
@mock.patch('machinelearning.machineLearningMode2.machineLearningMode2')    
def test_getQuoteData(mocked_machineLearningMode):
    startDate = '2017-06-15'
    endDate = '2017-06-17'  # not included
    symbol = 'KO'
    mocked_machineLearningMode.QUOTES_NUMBER = 100
    df = machineLearningRunner.getQuoteData(mocked_machineLearningMode, startDate, endDate, symbol)
    assert len(df) == 2
    
    df = machineLearningRunner.getQuoteData(mocked_machineLearningMode, startDate, endDate, symbol, history_days=3)
    assert len(df) == 5
    assert not {'Adj Low', 'Adj High', 'Adj Open', 'Adj Volume', 'Ex-Dividend', 'Split Ratio', 'index', 'level_0'}.issubset(df.columns)
    

def test_moreThanOneClassResult():
    assert machineLearningRunner.moreThanOneClassResult([1, 1, 0]) == True
    assert machineLearningRunner.moreThanOneClassResult([1, 1, 1]) == False
    assert machineLearningRunner.moreThanOneClassResult(None) == False
    

def test_getLastRowDate():
    x = pd.DataFrame({'Date': [1, 2, 3], 'y': [3, 4, 5]})
    '''
       Date y
    0  1   3
    1  2   4
    2  3   5
    '''
    assert machineLearningRunner.getLastRowDate(x) == 3
    
    
def test_getPickleName():
    assert machineLearningRunner.getPickleName('symbolX', 'modeY') == 'symbolX_modeY' 
    

@mock.patch('machinelearning.machineLearningRunner.getQuoteData')
@mock.patch('base.fileUtil.pickleIt')
@mock.patch('sklearn.ensemble.VotingClassifier') 
@mock.patch('machinelearning.machineLearningMode2.machineLearningMode2')    
def test_initialMachineLearning(mocked_machineLearningMode, mocked_classifier, mocked_pickleIt, mocked_getQuoteData):
    startDate = '2017-06-15'
    endDate = '2017-06-17'  # not included
    symbol = 'Test123'
    mocked_machineLearningMode.MODE = 'modeTest'
    mocked_machineLearningMode.QUOTES_NUMBER = 100
    mocked_machineLearningMode.MINIMUN_MACHINE_LEARNING_NUMBERS = 100
    df = pd.DataFrame({'Date': [1, 2, 3], 'y': [3, 4, 5]})
    mocked_getQuoteData.return_value = df
    result = machineLearningRunner.initialMachineLearning(mocked_machineLearningMode, startDate, endDate, symbol)
    mocked_getQuoteData.assert_called_once_with(mocked_machineLearningMode, startDate, endDate, symbol)
    assert result == None
     
    mocked_machineLearningMode.MINIMUN_MACHINE_LEARNING_NUMBERS = 1
    X = [[1], [2], [3]]
    y = [1, 1, 0]
    mocked_machineLearningMode.extract_featureset.return_value = (X, y, df)
    mocked_machineLearningMode.getClassifier.return_value = mocked_classifier
    result = machineLearningRunner.initialMachineLearning(mocked_machineLearningMode, startDate, endDate, symbol)
    mocked_machineLearningMode.extract_featureset.assert_called_once_with(df)
    mocked_classifier.fit.assert_called_once_with(X, y)
    mocked_pickleIt.assert_called_once_with("Test123_modeTest", {"lastRecordDate": 3, "clf": mocked_classifier})
     
 
@mock.patch('sklearn.ensemble.VotingClassifier')
@mock.patch('machinelearning.machineLearningRunner.getQuoteData') 
@mock.patch('machinelearning.machineLearningMode2.machineLearningMode2')       
def test_getClassifierAccuracy(mocked_machineLearningMode, mocked_getQuoteData, mocked_classifier):
    startDate = '2017-06-15'
    endDate = '2017-06-17'  # not included
    symbol = 'Zzzzzzz'
    mocked_machineLearningMode.MODE = 'modeTest'
    mocked_machineLearningMode.QUOTES_NUMBER = 100
    mocked_machineLearningMode.MINIMUN_MACHINE_LEARNING_NUMBERS = 100
    df = pd.DataFrame({'Date': [1, 2, 3], 'y': [3, 4, 5]})
    mocked_getQuoteData.return_value = df
    result = machineLearningRunner.getClassifierAccuracy(mocked_machineLearningMode, startDate, endDate, symbol)
    assert result == (None, 0)
     
    mocked_machineLearningMode.MINIMUN_MACHINE_LEARNING_NUMBERS = 1
    mocked_machineLearningMode.TEST_SIZE = 0.25
    X = [[1], [2], [3], [4]]
    y = [1, 1, 0, 0]
    mocked_machineLearningMode.extract_featureset.return_value = (X, y, df)
    mocked_machineLearningMode.getClassifier.return_value = mocked_classifier;
    mocked_classifier.score.return_value = 99
    result = machineLearningRunner.getClassifierAccuracy(mocked_machineLearningMode, startDate, endDate, symbol)
    mocked_classifier.fit.assert_called_once()
    mocked_classifier.score.assert_called_once()
    print('predict result', result)
    assert result == (99, 3)


@mock.patch('base.stockMongo.saveLearnAccuracy')
@mock.patch('machinelearning.machineLearningRunner.getClassifierAccuracy')
@mock.patch('machinelearning.machineLearningMode2.machineLearningMode2') 
def test_saveAccuracy(mocked_machineLearningMode, mocked_getClassifierAccuracy, mocked_saveLearnAccuracy):
    startDate = '2017-06-15'
    endDate = '2017-06-17'  # not included
    symbol = 'Zzzzzzz'
    confidence = random.random()
    trainNumber = random.randrange(1, 1000)
    mocked_getClassifierAccuracy.return_value = confidence, trainNumber
    mocked_machineLearningMode.MODE = "testMode"
    machineLearningRunner.saveAccuracy(mocked_machineLearningMode, startDate, endDate, symbol)
    mocked_saveLearnAccuracy.assert_called_once_with(symbol, mocked_machineLearningMode.MODE, confidence, trainNumber)


@mock.patch('sklearn.ensemble.VotingClassifier')  
@mock.patch('base.fileUtil.loadPickle')  
@mock.patch('machinelearning.machineLearningMode2.machineLearningMode2')     
def test_quotePredict(mocked_machineLearningMode, mocked_loadPickle, mocked_classifier):
    mocked_machineLearningMode.MODE = 'testMode'
    symbol = 'Zzzzzzz'
    X = []
    dates = ['2017-08-01', '2017-08-02']    
    machineLearningRunner.quotePredict(mocked_machineLearningMode, symbol, X, dates)
    
    X = [[1], [2]]
    mocked_loadPickle.return_value = {'clf':mocked_classifier}
    prediction1 = random.random()
    prediction2 = random.random()
    mocked_classifier.predict.return_value = [prediction1, prediction2]
    result = machineLearningRunner.quotePredict(mocked_machineLearningMode, symbol, X, dates)
    mocked_classifier.predict.assert_called_once_with(X)
    assert [(prediction1, dates[0]), (prediction2, dates[1])] == result


def getSampleDataFrame():
    date = ['2017-06-20', '2017-06-21', '2017-06-23']
    Close = [1, 2, 3]
    return pd.DataFrame(data={'Date':date, 'Close':Close}, index=range(3))
    

@mock.patch('machinelearning.machineLearningRunner.quotePredict')
@mock.patch('machinelearning.machineLearningRunner.getQuoteData')
@mock.patch('machinelearning.machineLearningMode2.machineLearningMode2')        
def test_predict(mocked_machineLearningMode, mocked_getQuoteData, mocked_quotePredict):
    date = '2017-08-01'
    symbol = 'Zzzzzzz'
    mocked_machineLearningMode.QUOTE_HISTORY_DAYS = 10
    mocked_getQuoteData.return_value = getSampleDataFrame()
    X = [[1], [2]]
    y = [0, 1]
    mocked_machineLearningMode.extract_featureForPredict.return_value = X, y, getSampleDataFrame()
    machineLearningRunner.predict(mocked_machineLearningMode, date, symbol)
    mocked_getQuoteData.assert_called_once_with(mocked_machineLearningMode, date, date, symbol, history_days=mocked_machineLearningMode.QUOTE_HISTORY_DAYS)
    mocked_machineLearningMode.extract_featureForPredict.assert_called_once()  # cannot compare arguments here as first argument is dataframe, dataframe1 == dataframe2 returns a dataframe not a boolean value    
    # mocked_quotePredict.assert_called_once_with(mocked_machineLearningMode, symbol, X, getSampleDataFrame()['Date'].values)
    # cannot compare arguments here, getSampleDataFrame()['Date'].values return type is 'numpy.ndarray'. 
    # getSampleDataFrame()['Date'].values == getSampleDataFrame()['Date'].values  returns array([ True,  True,  True], dtype=bool)
    mocked_quotePredict.assert_called_once()
    
    
