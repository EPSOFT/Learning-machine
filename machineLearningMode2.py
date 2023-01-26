import multiprocessing

from machinelearning import machineLearningRunner
from machinelearning import machineLearningUtil
import pandas as pd


class machineLearningMode2:
    MODE = 'mode2'
    MINIMUN_MACHINE_LEARNING_NUMBERS = 100
    TEST_SIZE = 0.25
    QUOTES_NUMBER = 1200  # 1 year approximately 200 quotes
    QUOTE_HISTORY_DAYS = 10
    THRESHOLD1 = 0.01
    THRESHOLD2 = -0.01
    
    
    def getClassifier(self):
        return machineLearningUtil.getDefaultClassifier()
    
    
    def prepareMachineLearningData(self, quotes, history_days=10):
        df = quotes.apply(pd.to_numeric, errors='ignore')
        machineLearningUtil.weaveInHistoryData(df, '', history_days)
        df = machineLearningUtil.addOtherQuoteData(df, '^GSPC', days=history_days)
        return df
    
    
    def determineResult(self, value):
        return machineLearningUtil.determineResult(self.THRESHOLD1, self.THRESHOLD2, value)


    def extract_featureset(self, df):
        df = self.prepareMachineLearningData(df)    
        if 'nextClosePercentage' not in df.columns :
            return None, None, df
          
        df['result'] = list(map(self.determineResult, df['nextClosePercentage']))  # use nextAdjClosePercentage
        df = df.dropna(how='any')
        return machineLearningUtil.extract_X(df), machineLearningUtil.extract_y(df), df
    
    
    def extract_featureForPredict(self, df, startDate, endDate):
        df = self.prepareMachineLearningData(df)
        # columns remove 
        df.drop(['nextClose', 'nextClosePercentage'], axis=1, inplace=True, errors='ignore')
        df = df.loc[(df['Date'] >= startDate) & (df['Date'] <= endDate)]
        return machineLearningUtil.extract_X(df), '', df


if __name__ == '__main__':      
    multiprocessing.freeze_support() 
#     testOneClassifierAccuracy()
#     test20ClassifierAccuracy()
    sepDate = '2017-06-01'
    machineLearningRunner.learn(machineLearningMode2(), sepDate)
        
