'''## Support Functions
1. lift_plot_model: Function to plot Lift Chart
2. plot_roc: Function to plot ROC Chart
3. evaluate_model: Function to calculate and return key model performance metrics and return them in a dataframe
4. plot_grid_search: Function to plot the validation curve for the GridSearchCV paramter tuning'''

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve 
from sklearn import metrics
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, 
                             mean_squared_error, log_loss, precision_recall_curve, classification_report, 
                             precision_recall_fscore_support)

import scikitplot as skplt
import os

def lift_plot_model(ytest, yprob):
    '''
    Objective: Function to plot Lift Chart
    Argument : Actual Take up rate(1/0), predicted probabilities
    Returns  : Lift chart, Lift table
    Output   : Lift Chart

    '''

    n_bins = 10

    actual_ser = pd.Series(ytest).rename('actuals').reset_index()
    proba_ser = pd.Series(yprob).rename('probabilities').reset_index()

    # Join table and drop indicies
    lift_table = pd.concat([actual_ser, proba_ser], axis=1).fillna(0)
    #lift_table.drop('index', inplace=True)
    actual_col = 'actuals'

    probability_col = 'probabilities'
    
    lift_table.sort_values(by=probability_col, ascending=False, inplace=True)

    rows = []

    # Split the data into the number of bins desired.
    for group in np.array_split(lift_table, n_bins):
        score = group[(group[actual_col] == 1)][actual_col].sum()

        rows.append({'NumCases': len(group), 'NumCorrectPredictions': score})

    lift = pd.DataFrame(rows)

    #Cumulative Gains Calculation
    lift['RunningCompleted'] = lift['NumCases'].cumsum() - lift['NumCases']

    lift['PercentCorrect'] = lift['NumCorrectPredictions'].cumsum() / \
    lift['NumCorrectPredictions'].sum() * 100

    lift['AvgCase'] = lift['NumCorrectPredictions'].sum() / len(lift)
    lift['CumulativeAvgCase'] = lift['AvgCase'].cumsum()
    #lift['PercentAvgCase'] = lift['CumulativeAvgCase'].apply(
    #    lambda x: (x*1.0 / lift['NumCorrectPredictions'].sum()) * 100)

    #Lift Chart
    lift['LiftLine'] = 1
    lift['Lift'] = lift['NumCorrectPredictions'] / lift['AvgCase']

    plt.plot(lift['Lift'], label= 'Response rate for model');

    plt.plot(lift['LiftLine'], 'r-', label='Normalised \'response rate\' \
    with no model');

    plt.xlabel(str(100/len(lift)) + '% Increments');
    plt.ylabel('Lift');
    plt.legend();
    plt.title("Lift Chart");
    plt.show();
    return lift
    #plt.gcf().clear()


def plot_roc(ytest_roc,yprob_roc):
        '''
        Objective: Function to plot ROC Graph
        Argument : ytest: Actual Take up rate(1/0), yprob: predcicted probabilities
        Returns  : ROC Plot
        Output   : ROC Plot

        '''
        fig = plt.figure(1, figsize=(6, 6));

        false_positive_rate, true_positive_rate, thresholds = \
        roc_curve(ytest_roc, yprob_roc)

        roc_auc = auc(false_positive_rate, true_positive_rate)

        plt.title("Receiving Operator Characteristic");

        plt.plot(false_positive_rate, true_positive_rate, 'b', \
        label='AUC = %0.2f' % roc_auc);

        plt.legend(loc='lower right');
        plt.plot([0,1], [0,1], 'r--');
        plt.xlim([-0.1, 1.2]);
        plt.ylim([-0.1, 1.2]);
        plt.ylabel("True Positive Rate");
        plt.xlabel("False Positive Rate");
        plt.tight_layout();

        nfig = plt.figure(2, figsize=(6, 6));
        plt.show();
        #plt.gcf().clear()


def evaluate_model(model_name, trained_model, xtrain, xtest, ytrain, ytest, verbose = False, threshold = 0.5):
    '''
        Objective: Function to calculate and return key model performance metrics
        Arguments: 7 arguments
                    1) model_name: Name of the model
                    2) trained_model: Trained model
                    3) xtrain: Training data set for features
                    4) xtest: testing dataset for features
                    5) ytrain: Training data set for target
                    6) ytest: testing dataset for target
                    7) verbose: print key performance metrics if True (default False)
                    8) threshold: Decision threshold used to classify the predicted probabilities
        Returns  : pd.DataFrame containing all key performance metrics
        Output   : pd.DataFrame containing all key performance metrics, ROC plot, Lift plot

    '''
    #Predict using trained model for training and test datasets (with and without probabilities) 
    prob_test = trained_model.predict_proba(xtest)
    prob_train = trained_model.predict_proba(xtrain)
    pred_test = (prob_test [:,1] >= threshold).astype('int')
    pred_train = (prob_train [:,1] >= threshold).astype('int')
    
    #Calculate AUC
    auc_score = roc_auc_score(ytest, prob_test[:,1])
    
    #Calculate train and test accuracy
    train_acc = accuracy_score(ytrain.values.ravel(), pred_train)
    test_acc = accuracy_score(ytest.values.ravel(), pred_test)
   
    #Calculate log loss value
    log_loss_value = log_loss(ytest, prob_test[:,1],eps=1e-15, normalize=True)
    
    #Generate confusion matrix
    conf_matrix = confusion_matrix(ytest.values.ravel(), pred_test)

    #Calculate classification model evaluation metrics like precision, recall, f1 score
    report = classification_report(ytest, pred_test)
    precision,recall,fscore,support=precision_recall_fscore_support(ytest,pred_test)
    
    
    print ("Lift plot for validation Sample")
    lift_table = lift_plot_model(ytest.values.ravel(), prob_test[:,1])

    print ("ROC curve for the validaton Sample")
    plot_roc(ytest.values.ravel(), prob_test[:,1])
    
    #Collate all key performance metrics into a dataframe
    model_evaluation_metrics = pd.DataFrame({'Model': [model_name], 'AUC': [auc_score], 'Test Accuracy': [test_acc]
              , 'Recall_1': [recall[1]], 'Precision_1': [precision[1]], 'F1 Score_1': [fscore[1]]                           
              , 'Log loss': [log_loss_value]})
    model_evaluation_metrics.columns
    model_evaluation_metrics = model_evaluation_metrics[['Model', 'AUC', 'Test Accuracy',
                                                    'Recall_1', 'Precision_1', 'F1 Score_1','Log loss']]

    #Get lifts in top n deciles, where n is defined below
    n=2
    lift_table.reset_index(inplace = True)
    lift_table['index'] = lift_table['index'] + 1
    lift_table['Decile'] = lift_table['index'].apply(lambda x: 'Decile_' + str(x) + ' Lift %')
    top_decile_lifts = lift_table[0:n][['Decile','Lift']].T
    top_decile_lifts.columns = top_decile_lifts.iloc[0]
    top_decile_lifts = top_decile_lifts.reindex(top_decile_lifts.index.drop('Decile'))
    top_decile_lifts.reset_index(drop=True,inplace=True)
    
    #Add lift values for top deciles to the model key performance metrics
    model_evaluation_metrics = pd.concat([model_evaluation_metrics,top_decile_lifts],axis=1)
    
    #Print key performance metrics if verbose = True is passed as an argument
    if verbose == True:
        print ("Trained model :: ", trained_model)
        print ("\n\nModel ROC-AUC score for validation sample: %.3f" \
                                          % auc_score)
        print ("\n\nTrain Accuracy :: ", train_acc)
        print ("\n\nTest Accuracy :: ", test_acc)
        print ("\n\nLog Loss Without for validation sample:", \
                        log_loss_value)
        
        print ("\n\n Confusion matrix \n")
        skplt.metrics.plot_confusion_matrix(ytest.values.ravel(), pred_test, title="Confusion Matrix",
            figsize=(4,4),text_fontsize='large')
        plt.show()
        
        print("\n\n Classification report (weighted average across classes) ::\n", classification_report(ytest, pred_test))

    return model_evaluation_metrics
	
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    '''
        Objective: To plot Validation Curve for GridSearchCV parameter tuning results 
        Arguments: 5 arguments
                    1) cv_results: Cross validation results from tuning
                    2) grid_param_1: List of parameter 1 values used for tuning
                    3) grid_param_2: List of parameter 2 values used for tuning
                    4) name_param_1: Parameter 1 name
                    5) name_param_2: Parameter 2 name
        Output   : Validation Curve plot with both parameters and CV results

    '''
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid('on')