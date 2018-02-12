"""
    My Module
    ~~~~~~~~~
"""



from . import *

class DataFrameDataset(Dataset):
    '''DataFrameDataset Class


        >>> df = pd.read_csv('../data/train_set.csv')
        >>> dfdset = DataFrameDataset(df, 'Class')

    '''
    def __init__(self, dataframe, y_label, **kwargs):
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

    def checkCategories(self, all_yes=False, threshold=None):
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
        if dataframe is None:
            self.dataframe = (self.dataframe - self.df_min) / (self.df_max - self.df_min) * 2 - 1
            return self.dataframe
        else:
            return (dataframe - self.df_min) / (self.df_max - self.df_min) * 2 - 1


    def standardizeDataFrame(self, dataframe=None):
        if dataframe is None:
            self.dataframe = (self.dataframe - self.df_mean) / self.df_std
            return self.dataframe
        else:
            return (dataframe - self.df_mean) / self.df_std

    def denormalizeDataFrame(self, dataframe=None):
        if dataframe is None:
            self.dataframe = (self.dataframe + 1) / 2 * (self.df_max - self.df_min) + self.df_min
            return self.dataframe
        else:
            return (dataframe + 1) / 2 * (self.df_max - self.df_min) + self.df_min

    def destandardizeDataFrame(self, dataframe=None):
        if dataframe is None:
            self.dataframe = self.dataframe * self.df_std + self.df_mean
            return self.dataframe
        else:
            return dataframe * self.df_std + self.df_mean

    def dataRound(self, dataframe=None):
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
