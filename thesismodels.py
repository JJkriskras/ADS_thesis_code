# This python script contains the class object that has the model creation pipeline.
# The usage of a class allows for easy and tidy model creation and usage 
# and creating a synched way to save important variables and models


### importing modules ###
# data management
import numpy as np
import pandas as pd
import os

# plot libraries
import matplotlib.pyplot as plt
import seaborn as sns

# import the pre processing class used in thesis
from preprocessing import preprocess

# sklearn modules used for linear regression, polynomial regression
# and creating the RMSE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# statsmodel library for a more advanced linear regression model
from statsmodels.api import OLS

# import tensorflow keras NN architecture
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K

#============================================================================================================
#========================================= models class =====================================================
#============================================================================================================

# create one model class containing the pipeline for modelling
# pre processing class will give the data needed, seed for random applications
# xplotname is for the plots and the benchmark regression

class models ():
    def __init__(self, x1, x2, y1, y2, save_dir = './L8dat/', seed = 42, xplotname = 'L8'):
        # variable allocation
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.seed = seed
        self.xplotname = xplotname

        # if save directory does not exist, make save dic.
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    #----------------------------------------------------------------------------------------
    # hidden plot function, these plots are called often, one fuction makes it more efficient.

    def __plot__(self, pred_test, residuals, plottype):
        plt.axes(aspect='equal')
        plt.scatter(self.y2, pred_test, alpha=.2)
        plt.xlabel('L7 true')
        plt.ylabel('L7 pred')
        lims = [0, 55000] # arbitrary range, but it works
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims)
        plt.title(f'L7 predicted vs True from {self.xplotname}\n')
        # save the plot to dic, give it a name that diffirentiates it from other plots
        plt.savefig(fname = os.path.join(self.save_dir,
                                         f'comparisson_L7_predvstrue_{plottype}.png')
                                         , format = 'png')
        
        #clear canvas
        plt.clf()

        plt.hist(residuals, bins=100)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.savefig(fname = os.path.join(self.save_dir,f'residuals_dist_{plottype}.png')\
                    , format = 'png')
        plt.clf()
    
    #--------------------------------------------------------------------
    #------------------------ model functions ---------------------------
    #--------------------------------------------------------------------

    def polyreg(self, maxit = 15, print_process = True, plot = True):

        # create a dictionary containing lists with results from the CV
        results_dict = {
            "Polynomials": [],
            "RMSE Train": [],
            "RMSE Test": []
        }

        # maxit +1 to get the maxit given
        for i in range(maxit+1):
            
            # create the polynomial structure
            poly = PolynomialFeatures(i)
            X_train = poly.fit_transform(self.x1)
            X_test = poly.fit_transform(self.x2)

            # create the polynomial model with LR
            m = LinearRegression()
            m.fit(X_train, self.y1)
            pred_train = m.predict(X_train)
            # obtain the train RMSE
            rmse_train = mean_squared_error(self.y1, pred_train, squared=False)

            # obtain the test RMSE
            pred_test = m.predict(X_test)
            rmse_test = mean_squared_error(self.y2, pred_test, squared=False)

            # print iteration result (optional)
            if print_process:
                print(f'poly = {i}, RMSE_train = {rmse_train}, RMSE_test = {rmse_test}')

            # append the dictionary
            results_dict['Polynomials'].append(i)
            results_dict['RMSE Test'].append(rmse_test)
            results_dict['RMSE Train'].append(rmse_train)

        # plot results (optional)
        if plot:
            # plot the best poly plot
            plt.plot(results_dict['RMSE Train'], label='RMSE_Train')
            plt.plot(results_dict['RMSE Test'], label='RMSE_test')
            plt.xlabel("poly's")
            plt.ylabel('RMSE')
            plt.legend()
            plt.grid(True)
            # add a line by the best poly
            lowest_test_rmse_index = np.argmin(results_dict['RMSE Test'])
            lowest_test_rmse = results_dict['RMSE Test'][lowest_test_rmse_index]
            plt.axvline(x=results_dict['Polynomials'][np.argmin(results_dict['RMSE Test'])],
                        color='r', linestyle='--',
                        label=f'Lowest RMSE Test ({lowest_test_rmse:.2f})')
            # save plot
            plt.savefig(fname = os.path.join(self.save_dir,'plot_best_poly.png'), format = 'png')
            # clean canvas
            plt.clf()
        
        # save the best polynomial as a callable variable in the class
        self.best_p = results_dict['Polynomials'][np.argmin(results_dict['RMSE Test'])]

        # create the polymodel with the best poly structure
        self.bestpolymodel(self.best_p, print_process=print_process, plot = plot)
        
    # a function creating the best poly model. sepperate function from the CV function
    # to allow the creation of poly models with differing polynomials 
    # (the CV function did not allow this)

    def bestpolymodel(self, best_p, print_process = True, plot = True, latex = False):
        # take the best polynomial (not from the callable class variable) 
        # and create the poly structure
        poly = PolynomialFeatures(best_p)
        X_train = poly.fit_transform(self.x1)
        X_train.shape
        X_test = poly.fit_transform(self.x2)

        # if using statsmodels.api, the variables do not have names
        # extract names from the sklearn poly class and add these
        # in the statsmodels OLS class variable name variable
        self.bestpoly = OLS(self.y1, X_train)
        self.bestpoly.exog_names[:] = poly.get_feature_names_out()
        self.bestpoly = self.bestpoly.fit()

        # ===============================================
        # ====== optional pipeline for sklearn =========
        # ===============================================

        # self.bestpoly = LinearRegression()    #=======
        # self.bestpoly.fit(X_train, self.y1)   #======= 

        # ===============================================

        # get predicted results and residuals
        pred_test = self.bestpoly.predict(X_test)
        residuals = pred_test - self.y2

        self.rmse_bestpoly = mean_squared_error(self.y2, pred_test, squared=False)

        # print results (optional)
        if print_process:
            print(f'poly = {best_p}')
            print(self.bestpoly.summary())

        #plot results (optional)
        if plot:
            self.__plot__(pred_test, residuals, 'poly')

        # print in latex format (for thesis apendix)
        if latex:
            print(self.bestpoly.summary().as_latex())


    #------------------------------------------------------------------------
    #---------------------- Neural network regression -----------------------
    #------------------------------------------------------------------------

    # create NN model function. maxit is epoch

    def NNmodel(self, maxit = 250, print_process = True, plot = True):

        # create loss function, RMSE through Keras API
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        # add normalisation layer, otherwise model will not converge
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.x1))

        # create model, one layer, linear activation
        linear_model = tf.keras.Sequential([
            normalizer,
            layers.Dense(units = 1, activation = 'linear')
        ])

        # print model summary before running (optional)
        if print_process:
            print(normalizer.mean.numpy())
            print(linear_model.summary())
        
        # add RMSE loss function
        linear_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss=rmse, metrics=[rmse])
        
        # run model. epochs = maxit, verbose is printing the process
        history = linear_model.fit(
            self.x1.astype('float32'),
            self.y1.astype('float32'),
            epochs=maxit,
            verbose=print_process,
            # Calculate validation results on 20% of the training data.
            validation_split = 0.2)

        # plot best epoch (optional)
        if plot:
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True)
            plt.savefig(fname = os.path.join(self.save_dir,'plot_best_NN.png'), 
                        format = 'png')
            plt.clf() # clean canvas

            # get predicitons and residuals
            test_predictions = linear_model.predict(self.x2.astype('float32')).flatten()
            residuals = test_predictions-self.y2
            self.__plot__(test_predictions, residuals, 'NNmodel')
        
        # save best model as callable class variable
        self.bestNNmodel = linear_model

        test_results = linear_model.evaluate(self.x2.astype('float32'), 
                                                 self.y2.astype('float32'), 
                                                 verbose=0, 
                                                 return_dict = True)
        self.rmseNN = test_results['rmse']

        if print:
            print(f'test results:\n{test_results}')
            print(f'\nkernel weigts:\n{self.bestNNmodel.layers[1].kernel}')

    #----------------------------------------------------------------------------
    #------------------------- Benchmark Linear Regression ----------------------
    #----------------------------------------------------------------------------   

    def benchmarreg(self, print_process = True, plot = True, latex = False):
        # make both models a callable class variable.
        # first regression is with only the Landsat data, no SA or SE
        # regression is performed using OLS from statsmodels
        self.onlyLreg = OLS(self.y1, self.x1[self.xplotname]).fit()
        # get RMSE on test set
        pred_test = self.onlyLreg.predict(self.x2[self.xplotname])
        self.rmse_onlyr = mean_squared_error(self.y2, pred_test, squared=False)

        # create linear fit with Landsat SA and SE
        self.linreg = OLS(self.y1, self.x1).fit()
        # get RMSE
        pred_test2 = self.linreg.predict(self.x2)
        self.rmse_benchR = mean_squared_error(self.y2, pred_test2, squared=False)

        # print results (optional)
        if print_process:
            print(self.onlyLreg.summary()) 
            print(f'\nRMSE only L regression = { self.rmse_onlyr}\n')     

            print(self.linreg.summary())  
            print(f'\nRMSE linear regression = {self.rmse_benchR}\n')
        
        # plot results (optional)
        if plot:
            residuals1 = pred_test - self.y2
            self.__plot__(pred_test, residuals1, 'benchmark_L_only_reg')
            
            residuals2 = pred_test2 - self.y2
            self.__plot__(pred_test2, residuals2, 'benchmark_lin_reg')

        # latex results (for thesis apendix)
        if latex:
            print(self.onlyLreg.summary().as_latex())
            print(self.linreg.summary().as_latex())
    #---------------------------------------------------------------------------
    #------------------------- overall summary ---------------------------------
    #---------------------------------------------------------------------------
    def summary(self):
        results = {
            'Benchmark only Landsat RMSE': self.rmse_onlyr,
            'Benchmark OLS Regression RMSE': self.rmse_benchR,
            'Polynomial model RMSE': self.rmse_bestpoly,
            'Neural Network model RMSE': self.rmseNN
        }

        print(results)


           

#----------------------------------------------------------
#----------------------------------------------------------

# some tests

# L87dat = preprocess(path = './nssamL57', save_dir='./L5dat/', X=['L5', 'SA', 'SE'], Y = 'L7')
# L87dat.get_df()
# L87dat.pre_process()
# L87dat.get_data_with_plot(X='L5')

# x1, x2, y1, y2 = L87dat.output()

# test = models(x1,x2,y1,y2, xplotname='L5', save_dir='./L5dat/')
# test.benchmarreg()
# print('\nPolyreg\n')
# test.polyreg(8,plot=True)
# test.NNmodel(maxit = 1500)
# test.summary()


#--------------------------------------------
#--------------------------------------------

