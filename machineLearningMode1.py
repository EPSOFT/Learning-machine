import multiprocessing

from machinelearning import machineLearningRunner
from machinelearning import machineLearningUtil
import pandas as pd


class machineLearningMode1:
    MODE = 'mode1'
    trainingDataColumns = ['Open', 'High', 'Low', 'Close', 'Volume', '3_mean', '7_mean', '3_mean_volume', '7_mean_volume', 'closeOpenPercentage']
    MINIMUN_MACHINE_LEARNING_NUMBERS = 100
    TEST_SIZE = 0.25
    QUOTES_NUMBER = 999999999999999  # 1 year approximately 200 quotes
    QUOTE_HISTORY_DAYS = 10
    THRESHOLD1 = 0.02
    THRESHOLD2 = -0.02
    
    
    def getClassifier(self):
        return machineLearningUtil.getDefaultClassifier()
    
    
    def prepareMachineLearningData(self, quotes, history_days=0):
        df = quotes.apply(pd.to_numeric, errors='ignore')
        
        df = df.sort_values(['Date'])
        df['3_mean'] = df['Close'].rolling(window=3).mean()
        df['7_mean'] = df['Close'].rolling(window=7).mean()
        df['3_mean_volume'] = df['Volume'].rolling(window=3).mean()
        df['7_mean_volume'] = df['Volume'].rolling(window=7).mean()
        df['closeOpenPercentage'] = (df['Close'] - df['Open']) / df['Close']
        
        if history_days is not None and history_days > 0 :
            machineLearningUtil.weaveInHistoryData(df, '', history_days)
        
        return df
    

    def determineResult(self, value):
        return machineLearningUtil.determineResult(self.THRESHOLD1, self.THRESHOLD2, value)
    
    def extract_X(self, df):
        return df[self.trainingDataColumns].values

    def extract_featureset(self, df):
        df = self.prepareMachineLearningData(df)    
        if 'nextClosePercentage' not in df.columns :
            return None, None, df
          
        df['result'] = list(map(self.determineResult, df['nextClosePercentage']))  # use nextAdjClosePercentage
        df = df.dropna(how='any')
        return self.extract_X(df), machineLearningUtil.extract_y(df), df
    
    
    def extract_featureForPredict(self, df, startDate, endDate):
        df = self.prepareMachineLearningData(df)
        # columns remove 
        df.drop(['nextClose', 'nextClosePercentage'], axis=1, inplace=True, errors='ignore')
        df = df.loc[(df['Date'] >= startDate) & (df['Date'] <= endDate)]
        return self.extract_X(df), '', df


if __name__ == '__main__':      
    multiprocessing.freeze_support() 
#     testOneClassifierAccuracy()
#     test20ClassifierAccuracy()
    sepDate = '2017-10-01'
    machineLearningRunner.learn(machineLearningMode1(), sepDate)
        
