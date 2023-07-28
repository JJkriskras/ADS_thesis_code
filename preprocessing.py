## modules used
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats


#---------------------------------------------------------------------
#-----------------------------------------------------------------------
#---------------------------------------------------------------------
class preprocess():
    def __init__(self, path = './test2/', save_dir = './scratch/',X = ['L8', 'SA', 'SE'],Y = 'L7',  seed = 42):
        self.path = path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.seed = seed
        self.X = X
        self.Y = Y
    

    def __data__(self, f):
        try:
            return pd.read_csv(os.path.join(self.path, f))
            
        except: pd.errors.EmptyDataError
        
    
    def __plotf__(self, dat, v, ax, X, Y):
        ax.scatter((dat[X]/1000), (dat[Y]/1000), alpha = .5)
        ax.set_xlabel(X)
        ax.set_ylabel(Y)
        ax.plot([5, 70], [5, 70])
        ax.set_title(v)


    def get_df(self, print_info = True):
        filelist = os.listdir(os.path.join(self.path)) # get file list
        df = pd.concat(list(map(self.__data__,filelist)))
        self.df = df.reset_index()

        if print_info:
            print(len(self.df))
            print(self.df.describe())

    
    def get_data_with_plot(self, X = 'L8'):

        filelist = os.listdir(os.path.join(self.path)) # get file list
        files = list(map(self.__data__,filelist))

        num_plots = len(filelist)
        grid_size = int(3) + 1

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

        for i, dat in enumerate(files):
            try:
                a = axes[i // grid_size, i % grid_size]
                # if X in dat.columns and self.Y in dat.columns:
                #     self.__plotf__(pd.DataFrame(dat), i + 1, a, X, self.Y)
                self.__plotf__(pd.DataFrame(dat), i+1, a, X, self.Y)
            except KeyError:
                continue
        
        plt.tight_layout()
        plt.legend()
        plt.savefig(fname = os.path.join(self.save_dir,'plot_data_norm.png'), format = 'png')

    def pre_process(self, delmax = True):
        dat = self.df[self.X]
        dat[self.Y] = self.df[self.Y]

        if delmax:
            dat = dat.loc[dat[self.X[0]] != dat[self.X[0]].max()]
            dat = dat.loc[dat[self.Y] != dat[self.Y].max()]

        self.train_features = dat.sample(frac=0.8, random_state=self.seed).copy()
        self.test_features = dat.drop(self.train_features.index).copy()

        self.train_Y = self.train_features.pop(self.Y)
        self.test_Y = self.test_features.pop(self.Y)

        self.prossdf = dat

    def plot_data(self, hist_data = ['L7', 'L8']):

        for d in hist_data:
            plt.hist(self.df[d], bins=30)
            plt.xlabel(d)
            plt.ylabel('Count')
            plt.savefig(fname = os.path.join(self.save_dir,d,'.png'), format = 'png')

            plt.clf()


    
    def output(self):
        try:
            return (self.train_features, self.test_features, self.train_Y, self.test_Y)
        except AttributeError:
            self.get_df()
            self.pre_process()
            return (self.train_features, self.test_features, self.train_Y, self.test_Y)
        
#------------------------------------------
#----------- Small tests ------------------
#------------------------------------------

# test = preprocess(path = './WCL57', save_dir='./L5dat/', X=['L5', 'SA', 'SE'], Y = 'L7')
# test.get_df()
# test.get_data_with_plot()

#=========================================================================================
#============================= Full data class ===========================================
#=========================================================================================

class full_data_preprocess():
    def __init__(self, path = '', save_dir = './scratch/',X = ['L8', 'SA', 'SE'],Y = 'L7'):
        self.path = path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.X = X
        self.Y = Y

        self.df = pd.read_csv(self.path)

    def process_data(self, delmax = True):
        dat = self.df[self.X]
        dat[self.Y] = self.df[self.Y]

        if delmax:
            dat = dat.loc[dat[self.X[0]] != dat[self.X[0]].max()]
            dat = dat.loc[dat[self.Y] != dat[self.Y].max()]
        
        self.xvar = dat[self.X]
        self.yvar = dat[self.Y]

    def plot_data(self, X = 'L8', title = "Comparisson plot"):

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.df[X], self.df[self.Y], alpha = .5)
        ax.set_xlabel(X)
        ax.set_ylabel(self.Y)
        ax.plot([5000, 70000], [5000, 70000])
        ax.set_title(title)
        plt.tight_layout()
        plt.legend()
        plt.savefig(fname = os.path.join(self.save_dir,'plot_data_full_norm.png'), format = 'png')


    def output(self):
        return (self.xvar, self.yvar)
    
# test = full_data_preprocess(path = r"C:\Users\Jurrian\Thesisdata\full_data\csv_nssam_3days_L7and8_full_2015-04-06_w.csv")
# test.process_data()
# test.plot_data()