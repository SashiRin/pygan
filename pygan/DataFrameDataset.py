"""
    DataFrameDataset
    ~~~~~~~~~
"""



from . import *

__all__ = ['DataFrameDataset']

class DataFrameDataset(Dataset):
    '''DataFrameDataset Class
    Pandas 데이터 프레임으로 만들 수 있는 데이터 셋 입니다.
    아래와 같이 이용할 수 있으며 Normalize와 Standardize를 위한 여러 함수를 포함하고 있습니다.
    또한 범주형 데이터를 자동으로 찾아 분류하는 기능을 가지고 있습니다.

        >>> df = pd.read_csv('../data/train_set.csv')
        >>> dfdset = DataFrameDataset(df, 'Class')

    '''
    def __init__(self, dataframe, y_label, **kwargs):
        """
        데이터 셋을 생성합니다.

        :param dataframe: 학습에 사용할 Pandas 데이터 프레임
        :param y_label: dataframe에서 레이블
        :param kwargs: checkCategories를 위한 인자 (all_yes, threshold)
        """
        self.dataframe = dataframe
        self.y_label = y_label
        self.x_dim = len(self.dataframe.columns) - 1
        self.class_num = len(pd.unique(self.dataframe[self.y_label]))
        self.categories = self.checkCategories(**kwargs)

        self.df_mean = self.dataframe.mean()
        self.df_max = self.dataframe.max()
        self.df_min = self.dataframe.min()
        self.df_std = self.dataframe.std()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.loc[item]
        x = row.drop(self.y_label).as_matrix().astype('float32')
        y = row[self.y_label].astype('long')

        return x, y

    def checkCategories(self, all_yes=False, threshold=None, **kwargs):
        """
        범주형 데이터를 자동으로 찾아 분류합니다.

        :param all_yes: 범주형 데이터인지에 대한 질문에 모두 예로 답합니다.
        :param threshold: 범주형 데이터가 되기 위한 최대의 서로 다른 값의 수를 의미합니다.
        :return: 분류된 범주형 데이터들의 메타데이터
        """
        if threshold == None:
            threshold = len(self.dataframe) // 20
        categories = {}
        for col in self.dataframe.columns:
            category = np.sort(self.dataframe[col].unique())

            if self.dataframe[col].dtype == np.int or len(category) < threshold:
                if all_yes:
                    categories[col] = category
                    continue

                else:
                    tqdm.write('\n'.join([
                        'The Checker detect that the column {} is categorical value'.format(col),
                        'If you want generated data to be in those categories, type y/n/ay/an: [y]'
                    ]))
                    ans = input()
                    if ans.lower() == 'y' or ans == '':
                        categories[col] = category
                    elif ans.lower() == 'ay':
                        categories[col] = category
                        all_yes = True
                    elif ans.lower() == 'an':
                        break
        return categories

    def normalizeDataFrame(self, dataframe=None):
        """
        데이터를 -1과 1사이로 Normalize합니다.
        dataframe이 주어지면 Normalize된 dataframe을 리턴하며, 기존의 dataframe은 바꾸지 않습니다.
        dataframe이 주어지지 않으면 데이터 셋이 가진 dataframe을 normalize합니다.

        :param dataframe: normalize하려는 dataframe
        :return: normalized dataframe
        """
        if dataframe is None:
            self.dataframe = (self.dataframe - self.df_min) / (self.df_max - self.df_min) * 2 - 1
            return self.dataframe
        else:
            return (dataframe - self.df_min) / (self.df_max - self.df_min) * 2 - 1


    def standardizeDataFrame(self, dataframe=None):
        """
        데이터의 평균이 0이 되도록 Standardize합니다.
        dataframe의 여부에 따른 결과는 normalizeDataFrame과 같습니다.

        :param dataframe: standardize하려는 dataframe
        :return: standardized dataframe
        """
        if dataframe is None:
            self.dataframe = (self.dataframe - self.df_mean) / self.df_std
            return self.dataframe
        else:
            return (dataframe - self.df_mean) / self.df_std

    def denormalizeDataFrame(self, dataframe=None):
        """
        데이터를 원래 영역으로 Denormalize합니다.
        dataframe의 여부에 따른 결과는 normalizeDataFrame과 같습니다.

        :param dataframe: denormalize하려는 dataframe
        :return: denormalized dataframe
        """
        if dataframe is None:
            self.dataframe = (self.dataframe + 1) / 2 * (self.df_max - self.df_min) + self.df_min
            return self.dataframe
        else:
            return (dataframe + 1) / 2 * (self.df_max - self.df_min) + self.df_min

    def destandardizeDataFrame(self, dataframe=None):
        """
        데이터를 원래 영역으로 Destandardize합니다.
        dataframe의 여부에 따른 결과는 normalizeDataFrame과 같습니다.

        :param dataframe: destandardize하려는 dataframe
        :return: destandardized dataframe
        """
        if dataframe is None:
            self.dataframe = self.dataframe * self.df_std + self.df_mean
            return self.dataframe
        else:
            return dataframe * self.df_std + self.df_mean

    def dataRound(self, dataframe=None):
        """
        데이터 프레임의 데이터를 범주형 데이터 메타데이터에 따라 선형 반올림처리합니다.
        dataframe의 여부에 따른 결과는 normalizeDataFrame과 같습니다.

        :param dataframe: round하려는 dataframe
        :return: rounded dataframe
        """
        if dataframe is None:
            dataframe = self.dataframe
        else:
            dataframe = dataframe.copy()

        pbar = tqdm(total = len(self.categories))
        def round_val(mlist, mnum):
            pos = bisect_left(mlist, mnum)
            if pos == 0:
                return mlist[0]
            if pos == len(mlist):
                return mlist[-1]
            before = mlist[pos - 1]
            after = mlist[pos]
            if after - mnum < mnum - before:
                return after
            else:
                return before

        for col in self.categories:
            rounder = self.categories[col]
            for i in range(len(dataframe)):
                dataframe[col][i] = round_val(rounder, dataframe[col][i])
            pbar.update(1)
        return dataframe
