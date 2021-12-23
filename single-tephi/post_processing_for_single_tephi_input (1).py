from preprocess_for_single_tephi_input import *
from build_model_for_single_tephi_input import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras.backend as K
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from numpy import interp
from itertools import cycle
from sklearn.metrics import classification_report
from reliability_diagram import *
from sklearn.calibration import calibration_curve


def plot_image(date, date_list, predictions_array, true_label, img, class_names=['Small DV', 'Normal DV', 'Large DV']):
    i = date_list.index(date)
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    img = read_img(img)
    img = decode_img(img)
    plt.imshow(img)#, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_prediction_bar(date, date_list, predictions_array, true_label, n_class):
    i = date_list.index(date)
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(n_class))
    #plt.yticks()

    thisplot = plt.bar(range(n_class), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[predicted_label].set_label('Predicted')
    thisplot[true_label].set_color('blue')
    thisplot[true_label].set_label('True')
    plt.legend(loc='best')
    
    
def classify_class_to_str(int_class, n_class):
    if n_class == 3:
        if int_class == 0:
            return 'Small'
        elif int_class == 1:
            return 'Normal'
        elif int_class == 2:
            return 'Large'
        else:
            raise Exception('int DV classes are 0, 1, 2')
    elif n_class == 2:
        if int_class == 0:
            return 'Normal'
        elif int_class == 1:
            return 'Large'
        else:
            raise Exception('int DV classes are 0, 1')
        

def classify_class_to_int(str_class, n_class):
    if n_class == 3:
        if str_class == 'Small':
            return 0
        elif str_class == 'Normal':
            return 1
        elif str_class == 'Large':
            return 2
        else:
            raise Exception('str DV classes are Small, Normal, Large')
    elif n_class == 2:
        if str_class == 'Normal':
            return 0
        elif str_class == 'Large':
            return 1
        else:
            raise Exception('str DV classes are Normal, Large')

            
def get_pred_label(y_pred, threshold=0.5):
    # threshold=0.5 is for binary classification where output probability > 0.5 counts as class 1 else class 0, you can adjust the threshold
    out_shape = y_pred.shape[1]
    if out_shape == 3:
        n_class = out_shape
        pred_label = []
        for out_prob in y_pred:
            condition_1 = len(np.unique(out_prob)) == 1
            condition_2 = (out_prob[0] == out_prob[1]) & (out_prob[2] == np.min(out_prob))
            condition_3 = (out_prob[0] == out_prob[2]) & (out_prob[1] == np.min(out_prob))
            condition_4 = (out_prob[1] == out_prob[2]) & (out_prob[0] == np.min(out_prob))
            if condition_1 | condition_2 | condition_3 | condition_4:
                pred_label.append(1)
            else:
                max_class = np.argmax(out_prob)
                pred_label.append(max_class)
    elif out_shape == 1:
        n_class = 2
        pred_bool = y_pred[:, 0] > threshold
        pred_label = pred_bool.astype(int)
        
    else:
        raise Exception('For n_class > 3, please define your conditions and add to this function')
    
    return pred_label


def get_str_class(label, n_class):
    # 0-->Small ; 1-->Normal ; 2-->Large / binary (n_class=2): 0-->Normal ; 1-->Large
    classify_class_to_str_v = np.vectorize(classify_class_to_str)
    str_class = classify_class_to_str_v(label, n_class)
    return str_class


def get_crosstab(pred_str_class, y_true):
    # Just like confusion matrix
    crosstab = pd.crosstab(pred_str_class, y_true, rownames=['Forecast'], colnames=['Actual'])
    if len(np.unique(y_true)) == 2:
        classes = ['Normal', 'Large']
    elif len(np.unique(y_true)) == 3:
        classes = ['Small', 'Normal', 'Large']
        
    for i in classes:
        if i not in crosstab.index:
            zero_df = pd.DataFrame([0]*len(classes)).T
            zero_df.index = [i]
            zero_df.columns = crosstab.columns
            crosstab = pd.concat([crosstab, zero_df], axis=0)
    return crosstab


def get_score_df(crosstab):
    # crosstab[column_name][row_name] in this case, don't be confused with DataFrame.iloc[row_index][column_index]
    # POD, FAR, CSI for each class
    if crosstab.shape[1] == 3:
        tp_0 = crosstab[0]['Small']
        fa_0 = crosstab[1]['Small'] + crosstab[2]['Small']
        miss_0 = crosstab[0]['Large'] + crosstab[0]['Normal']
        
        POD_0 = tp_0 / (tp_0 + miss_0)
        try:
            FAR_0 = fa_0 / (tp_0 + fa_0)
        except ZeroDivisionError:
            FAR_0 = np.nan
        CSI_0 = tp_0 / (tp_0 + fa_0 + miss_0)

        tp_1 = crosstab[1]['Normal']
        fa_1 = crosstab[0]['Normal'] + crosstab[2]['Normal']
        miss_1 = crosstab[1]['Large'] + crosstab[1]['Small']
        
        POD_1 = tp_1 / (tp_1 + miss_1)
        try:
            FAR_1 = fa_1 / (tp_1 + fa_1)
        except ZeroDivisionError:
            FAR_1 = np.nan
        CSI_1 = tp_1 / (tp_1 + fa_1 + miss_1)

        tp_2 = crosstab[2]['Large']
        fa_2 = crosstab[0]['Large'] + crosstab[1]['Large']
        miss_2 = crosstab[2]['Normal'] + crosstab[2]['Small']
        
        POD_2 = tp_2 / (tp_2 + miss_2)
        try:
            FAR_2 = fa_2 / (tp_2 + fa_2)
        except ZeroDivisionError:
            FAR_2 = np.nan
        CSI_2 = tp_2 / (tp_2 + fa_2 + miss_2)
        
        df_score = pd.DataFrame({'Small':[POD_0, FAR_0, CSI_0], 'Normal':[POD_1, FAR_1, CSI_1], 'Large':[POD_2, FAR_2, CSI_2]}, 
                                index=['POD', 'FAR', 'CSI'])
        for column in df_score.columns:
            df_score[column] = df_score[column].apply(lambda x: round_(x, 2))
        
        return df_score
                    
    elif crosstab.shape[1] == 2:
        tp_0 = crosstab[0]['Normal']
        fa_0 = crosstab[1]['Normal']
        miss_0 = crosstab[0]['Large']

        tp_1 = crosstab[1]['Large']
        fa_1 = crosstab[0]['Large']
        miss_1 = crosstab[1]['Normal']

        POD_0 = tp_0 / (tp_0 + miss_0)
        FAR_0 = fa_0 / (tp_0 + fa_0)
        CSI_0 = tp_0 / (tp_0 + fa_0 + miss_0)

        POD_1 = tp_1 / (tp_1 + miss_1)
        FAR_1 = fa_1 / (tp_1 + fa_1)
        CSI_1 = tp_1 / (tp_1 + fa_1 + miss_1)
        
        df_score = pd.DataFrame({'Normal':[POD_0, FAR_0, CSI_0], 'Large':[POD_1, FAR_1, CSI_1]}, 
                                index=['POD', 'FAR', 'CSI'])
        for column in df_score.columns:
            df_score[column] = df_score[column].apply(lambda x: round_(x, 2))

        return df_score


def get_result(pred_str_class, y_true, to_file=None):
    # Combine df_score and crosstab
    crosstab = get_crosstab(pred_str_class, y_true)
    df_score = get_score_df(crosstab)
    score = df_score.T.iloc[::-1].copy()
    score = score.reset_index().rename(columns={'index':'class'}).copy()
    crosstab = crosstab.reset_index(drop=True)
    result = pd.concat([score, crosstab], axis=1)
    if to_file is not None:
        result.to_csv(to_file, index=False)
    return result


def split_cold_hot(date, y_pred, y_true):
    # To see the performance in cold and hot period
    # Cold: Nov-Apr ; Hot: May-Oct
    date_pred_couple = list(zip(date, y_pred))
    date_true_couple = list(zip(date, y_true))

    hot = list(filter(lambda x: datetime.strptime(x[0], '%Y%m%d%H').month in range(5, 10+1), date_pred_couple))
    hot_date = [x[0] for x in hot]

    cold = list(filter(lambda x: not x[0] in hot_date, date_pred_couple))
    
    cold_pred = np.asarray([x[1] for x in cold])
    hot_pred = np.asarray([x[1] for x in hot])
    
    cold_true = list(filter(lambda x: not x[0] in hot_date, date_true_couple))
    cold_true = np.asarray([x[1] for x in cold_true])
    hot_true = list(filter(lambda x: x[0] in hot_date, date_true_couple))
    hot_true = np.asarray([x[1] for x in hot_true])
    return cold_pred, hot_pred, cold_true, hot_true


def plot_history(history, n_class, to_file=None):
    # Plot history of loss function and other metrics for training and validation set
    if n_class >= 3:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        plot_y1 = ['loss', 'accuracy']
        plot_y2 = ['val_' + y for y in plot_y1]
        plot_y = list(zip(plot_y1, plot_y2))
        label_1 = ['training ' + y for y in plot_y1]
        label_2 = ['validation ' + y for y in plot_y1]
        labels = list(zip(label_1, label_2))
        colors = [('C0', 'C1'), ('b', 'g')]
        
        for i in range(2):
            ax[i].plot(history.history[plot_y[i][0]], label=labels[i][0])
            ax[i].plot(history.history[plot_y[i][1]], label=labels[i][1])
            ax[i].legend(loc='best')

        plt.show()
        
    elif n_class == 2:
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        plot_y1 = ['loss', 'accuracy', 'auc', 'prc']
        plot_y2 = ['val_' + y for y in plot_y1]
        plot_y = list(zip(plot_y1, plot_y2))
        label_1 = ['training ' + y for y in plot_y1]
        label_2 = ['validation ' + y for y in plot_y1]
        labels = list(zip(label_1, label_2))
        colors = [('C0', 'C1'), ('b', 'g'), ('royalblue', 'darkorange'), ('navy', 'orangered')]
        
        for i in range(2):
            for j in range(4):
                if i == 0:
                    ax[i][j//2].plot(history.history[plot_y[j//2][j%2]], label=labels[j//2][j%2], c=colors[j//2][j%2])
                    ax[i][j//2].legend(loc='best')
                else:
                    ax[i][j//2].plot(history.history[plot_y[2:][j//2][j%2]], label=labels[2:][j//2][j%2], c=colors[2:][j//2][j%2])
                    ax[i][j//2].legend(loc='best')

        plt.show()
    
    if to_file is not None:
        fig.savefig(to_file, bbox_inches="tight", pad_inches=0.2)  
    plt.close()

        
def print_best_epoch(history, n_class):
    # Best epoch: min val_loss / max val_accuracy / max val_auc / max val_prc
    if n_class == 2:
        best = ['min', 'max', 'max', 'max']
        val_metrics = ['val_loss', 'val_accuracy', 'val_auc', 'val_prc']
    elif n_class >= 3:
        best = ['min', 'max']
        val_metrics = ['val_loss', 'val_accuracy']
        
    val_list = [history.history[i] for i in val_metrics]
    best_value = []
    best_epoch = []
    for i in range(len(val_list)):
        if val_metrics[i] == 'val_loss':
            best_value.append(np.min(val_list[i]))
            best_epoch.append(val_list[i].index(best_value[i]) + 1)
        else:
            best_value.append(np.max(val_list[i]))
            best_epoch.append(val_list[i].index(best_value[i]) + 1)
            
    for m, metrics, value, epoch in zip(best, val_metrics, best_value, best_epoch):
        print('{} {}: {:.4f} ; Epoch {}\n'.format(m, metrics, value, epoch))
    
        
def plot_roc_curve(y_true, y_pred, classes, lw=2, directory=None):
    n_classes = len(classes)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred[:, 0])
        roc_auc = auc(fpr, tpr)
        
        fig = plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
        
    elif n_classes >= 3:
        y_binary = label_binarize(y_true, classes=classes)

        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_binary.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        fig = plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
    
    if directory is not None:
        fig.savefig(os.path.join(directory, 'ROC_curve.png'), bbox_inches="tight", pad_inches=0.2)   
        
        
def plot_pr_curve(y_true, y_pred, classes, lw=2, directory=None):
    n_classes = len(classes)
    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_pred[:, 0])
        average_precision = average_precision_score(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
            
        ax.plot(recall, precision, color='mediumblue', lw=lw, label='Precision-recall curve (AP = %0.2f)' % average_precision)
        
        # add the legend for the iso-f1 curves
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall curve')
        ax.legend(handles=handles, labels=labels, loc="best")
        plt.show()
        
    elif n_classes >= 3:
        # Use label_binarize to be multi-label like settings
        y_binary = label_binarize(y_true, classes=classes)

        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_binary[:, i], y_pred[:, i])
            average_precision[i] = average_precision_score(y_binary[:, i], y_pred[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_binary.ravel(), y_pred.ravel()
        )
        average_precision["micro"] = average_precision_score(y_binary, y_pred, average="micro")

        # setup plot details
        colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

        fig, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        ax.plot(recall["micro"], precision["micro"],
                 label='micro-average precision-recall curve (AP = {0:0.2f})'
                       ''.format(average_precision["micro"]),
                 color='gold', linestyle=':', linewidth=4)

        for i, color in zip(range(n_classes), colors):
            ax.plot(recall[i], precision[i], color=color, lw=lw,
                     label='Precision-recall curve of class {0} (AP = {1:0.2f})'
                     ''.format(i, average_precision[i]))
        
        # add the legend for the iso-f1 curves
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title("Precision-Recall curve")
        ax.legend(handles=handles, labels=labels, loc="best")
        plt.show()
   
    if directory is not None:
        fig.savefig(os.path.join(directory, 'Precision-Recall_curve.png'), bbox_inches="tight", pad_inches=0.2)
        
        
def get_review(y_true, y_pred, date, df_dv, verification, directory=None):
    n_class = len(np.unique(y_true))
    true_str_class = get_str_class(y_true, n_class)
    
    if y_pred.shape[1] == 1:
        prob_0 = 1 - y_pred[:, 0]
        prob_1 = y_pred[:, 0]
        prob_2 = None
        confidence = np.vectorize(lambda x: 1 - x if x <= 0.5 else x)(y_pred[:, 0])
    elif y_pred.shape[1] == 3:
        prob_0 = y_pred[:, 0]
        prob_1 = y_pred[:, 1]
        prob_2 = y_pred[:, 2]
        confidence = np.max(y_pred, axis=1)
        
    pred_label = get_pred_label(y_pred, threshold=0.5)
    pred_str_class = get_str_class(pred_label, n_class)
    
    test_date_df = pd.DataFrame({'date':date})
    df_dv = test_date_df.merge(df_dv, how='left', on='date')
    DV_test = df_dv['Abs_DV'].values
    
    #day_maxTT = df_dv['Day_maxT'].values
    #am_minTT = df_dv['Morn_minT'].values
    
    abs_maxTT = df_dv['Abs_maxT'].values
    abs_minTT = df_dv['Abs_minT'].values
    abs_DV = df_dv['Abs_DV'].values
            
    df_review = pd.DataFrame({'Date':date, 'Prob_0':prob_0, 'Prob_1':prob_1, 'Prob_2':prob_2,
                              'Predicted DV class':pred_str_class, 'Actual DV class':true_str_class, 'Actual DV':DV_test,
                              'true_label':y_true, 'pred_label':pred_label, 'confidence':confidence,
                              'abs_minTT':abs_minTT, 'abs_maxTT':abs_maxTT, 'abs_DV':abs_DV,
                              'verification':verification})
    if directory is not None:
        if verification == True:
            df_review.to_csv(os.path.join(directory, 'model_review_EC.csv'), index=False)
        elif verification == False:
            df_review.to_csv(os.path.join(directory, 'model_review.csv'), index=False)
        
    return df_review


def plot_reliability_diagram(df, num_bins=10, avg=False, 
                             classes=None, start_UTC=None, lead_day=None, directory=None):
    # Since for avg=True, the max probability at least 1/3 for 3 classes, range from 0.333 to 1
    # You can adjust num_bins to 7 to increase the range in each bin for the above case
    y_true = df.true_label.values
    y_pred = df.pred_label.values
    y_conf = df.confidence.values
    if avg:
        # Sanity check: compute top-1 accuracy.
        accuracy = (df.true_label == df.pred_label).sum() / len(df)
        avg_conf = df.confidence.mean()
        print('Accuracy:', accuracy)
        print('Mean confidence:', avg_conf)

        # Override matplotlib default styling.
        plt.style.use("seaborn")
        plt.rc("font", size=12)
        plt.rc("axes", labelsize=12)
        plt.rc("xtick", labelsize=12)
        plt.rc("ytick", labelsize=12)
        plt.rc("legend", fontsize=12)
        plt.rc("axes", titlesize=16)
        plt.rc("figure", titlesize=16)

        title = 'Reliability diagram'
        fig = reliability_diagram(y_true, y_pred, y_conf, num_bins=num_bins, draw_ece=True,
                                  draw_bin_importance="alpha", draw_averages=True,
                                  title=title, figsize=(6, 6), dpi=100, 
                                  return_fig=True)

        bin_data = compute_calibration(y_true, y_pred, y_conf, num_bins=num_bins)
        
        if directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
            sub_dir = os.path.join(directory, 'reliability_diagram')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            if df['verification'].all():
                fig.savefig(os.path.join(sub_dir, "reliability_diagram_{}Z_lead-{}d_avg.png".format(start_UTC, lead_day)), 
                            dpi=144, bbox_inches="tight", pad_inches=0.2)
            else:
                fig.savefig(os.path.join(sub_dir, "reliability_diagram_avg.png"), dpi=144, bbox_inches="tight", pad_inches=0.2)
    
    else:
        y_binary = label_binarize(y_true, classes=classes)
        ## For Small DV ##
        small_true = y_binary[:, 0]
        small_pred = df.Prob_0.values
        fop_small, mpv_small = calibration_curve(small_true, small_pred, n_bins=num_bins)
        
        ## For Normal DV ##
        normal_true = y_binary[:, 1]
        normal_pred = df.Prob_1.values
        fop_normal, mpv_normal = calibration_curve(normal_true, normal_pred, n_bins=num_bins)
        
        ## For Large DV ##
        large_true = y_binary[:, 2]
        large_pred = df.Prob_2.values
        fop_large, mpv_large = calibration_curve(large_true, large_pred, n_bins=num_bins)
        
        fop_mpv = [(fop_small, mpv_small), (fop_normal, mpv_normal), (fop_large, mpv_large)]
        probs = [small_pred, normal_pred, large_pred]
        classes = ['Small', 'Normal', 'Large']
        # fop: Fraction of positives ; mpv: Mean predicted value
        
        fig2, ax = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
        ace = ['( a )', '( b )', '( c )']
        bdf = ['( d )', '( e )', '( f )']
        for i in range(len(classes)):
            # plot perfectly calibrated
            ax[0][i].plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')

            # plot model reliability for each class
            ax[0][i].plot(fop_mpv[i][1], fop_mpv[i][0], marker='.', markersize=12, label='Reliability for %s DV' % classes[i])
            ax[0][i].set_ylabel('Observed relative frequency')
            ax[0][i].annotate(ace[i], xy=(ax[0][i].get_xlim()[0], ax[0][i].get_ylim()[1]))
            ax[0][i].legend(loc='best')

            # plot histogram of prob for each class
            ax[1][i].hist(probs[i], bins=num_bins, range=(0, 1), label='%s DV' % classes[i], histtype="bar", rwidth=0.8) #lw=2
            ax[1][i].set_xlabel('Forecast probability') 
            ax[1][i].set_ylabel('Count')
            ax[1][i].annotate(bdf[i], xy=(ax[1][i].get_xlim()[0], ax[1][i].get_ylim()[1]))
            ax[1][i].legend(loc='best')

        plt.tight_layout()
        plt.show()
        
        if directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
            sub_dir = os.path.join(directory, 'reliability_diagram')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            if df['verification'].all():
                fig2.savefig(os.path.join(sub_dir, "reliability_diagram_{}Z_lead-{}d.png".format(start_UTC, lead_day)), 
                             dpi=144, bbox_inches="tight", pad_inches=0.2)
            else:
                fig2.savefig(os.path.join(sub_dir, "reliability_diagram.png"), dpi=144, bbox_inches="tight", pad_inches=0.2)

                
def plot_cam(date, model, df_review, test_img, test_dataset, batch_size, layer_name='vgg19', ax=None):
    '''
    Class Activation Map (CAM)
        activation='relu' in the last fully connect layer (i.e. Dense) may generate many zeros that leads to invalid CAM, 
        may change to activation='tanh'
    '''
    test_date_ = df_review['Date'].tolist()    
    if date in test_date_:
        batch_idx = test_date_.index(date) // batch_size
        idx = test_date_.index(date) % batch_size
        
        img_tensor = np.expand_dims(list(test_dataset)[batch_idx][0][0][idx].numpy(), axis=0)
        sounding_tensor = np.expand_dims(list(test_dataset)[batch_idx][0][1][idx].numpy(), axis=0)
        
        # Review
        review = df_review[df_review['Date']==date]
        Prob0 = review['Prob_0'].values[0]
        Prob1 = review['Prob_1'].values[0]
        Prob2 = review['Prob_2'].values[0]
        predicted_class = review['Predicted DV class'].values[0]
        actual_class = review['Actual DV class'].values[0]
        actual_DV = review['Actual DV'].values[0]
        pred_label = review['pred_label'].values[0]
        true_label = review['true_label'].values[0]
        
        with tf.GradientTape() as tape:
            last_conv_layer = model.get_layer(layer_name)
            iterate = models.Model(inputs=[model.inputs], outputs=[model.output, last_conv_layer.get_output_at(0)])
            model_out, last_conv_layer = iterate([img_tensor, sounding_tensor])
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap[0]
        
        # Plotting
        img = cv2.imread(test_img[batch_size*batch_idx + idx])
        
        INTENSITY = 0.5
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        cam = heatmap * INTENSITY + img
        cam = cam / np.max(cam) #np.uint8(cam / np.max(cam) * 255)
        cam_rgb = cam[:,:,::-1]
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            if Prob2 is None:
                ax.annotate('[%.2f, %.2f]' % (Prob0, Prob1), (10, 30))
            else:
                ax.annotate('[%.2f, %.2f, %.2f]' % (Prob0, Prob1, Prob2), (10, 30))
            ax.annotate('Predicted: %s' % (predicted_class), (10, 50))
            ax.annotate('Actual: %s' % (actual_class), (10, 70))
            ax.annotate('DV: %s' % (actual_DV), (10, 90))
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(cam_rgb)
            
        print(date)
        #plt.show()
        #plt.close()
    else:
        print('{} is not included'.format(date))
    
    
def save_cam(date, model, df_review, test_img, test_dataset,
             batch_size, layer_name_list, classes, directory,
             start_UTC=None, lead_day=None, row=4, column=4):
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    if df_review['verification'].all():
        sub_dir = os.path.join(directory, 'heatmap_{}Z_day{}'.format(start_UTC, lead_day))
    else:
        sub_dir = os.path.join(directory, 'heatmap_test')
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
        for i in classes:
            for j in classes:
                os.makedirs(os.path.join(sub_dir, 'pred{}_true{}'.format(i, j)))
    for d in date:
        pl = df_review[df_review['Date']==d]['pred_label'].values[0]
        tl = df_review[df_review['Date']==d]['true_label'].values[0]
        dv = df_review[df_review['Date']==d]['Actual DV'].values[0]
        
        fig, ax = plt.subplots(row, column, figsize=(8, 10))
        for i in range(len(layer_name_list)):
            cam = plot_cam(d, model, df_review, test_img, test_dataset, batch_size, 
                           layer_name=layer_name_list[i], 
                           ax=ax[i//column][i%column])
            
            ax[i//column][i%column].set_title(layer_name_list[i])

        fig.savefig(os.path.join(sub_dir, 'pred{}_true{}'.format(pl, tl), 
                                 '{}_DV-{}_heatmap.png'.format(d, dv)), bbox_inches='tight', pad_inches=0.2)
        plt.close()

        
def get_wdir_from_uv(args):
    # EC data has no wind direction, so need to derive from U (zonal wind speed - dx/dt) and V (meridional wind speed - dy/dt)
    u = args[0]
    v = args[1]
    
    if (u == 0) & (v > 0):
        return 180
    
    elif (u == 0) & (v < 0):
        return 360
    
    elif (u > 0) & (v == 0):
        return 270
    
    elif (u < 0) & (v == 0):
        return 90
    
    elif (u == 0) & (v == 0):
        return 0
    
    elif (u > 0) & (v > 0):
        return 270 - abs(np.rad2deg(np.arctan(v/u)))
    
    elif (u < 0) & (v > 0):
        return 90 + abs(np.rad2deg(np.arctan(v/u)))
    
    elif (u > 0) & (v < 0):
        return 270 + abs(np.rad2deg(np.arctan(v/u)))
    
    elif (u < 0) & (v < 0):
        return 90 - abs(np.rad2deg(np.arctan(v/u)))
    
    else:
        return np.nan
           

def get_ec_data_list(start_UTC='00', lead_day=1): 
    # start_UTC='00'/'12' ; lead_day=1,2,...9
    model_run = sorted(os.listdir('/home/deeplearn/cslau/input_full_137'))
    model_run = [x + '00' for x in model_run if x[-2:]==start_UTC]

    if start_UTC == '00':
        EC_tephi_list = glob.glob('/home/deeplearn/cslau/input_full_137/*00/*0000_1.grb-HK05.out')
        forecast = [datetime.strftime(datetime.strptime(x[:-4], '%Y%m%d')+timedelta(days=lead_day), '%Y%m%d') + '0000' 
                        for x in model_run]
    elif start_UTC == '12':
        EC_tephi_list = glob.glob('/home/deeplearn/cslau/input_full_137/*12/*0000_1.grb-HK05.out')
        forecast = [datetime.strftime(datetime.strptime(x[:-4], '%Y%m%d')+timedelta(days=1+lead_day), '%Y%m%d') + '0000' 
                        for x in model_run]
    else:
        raise TypeError('start_UTC only includes "00" and "12"')
        
    start_end = ['_'.join(x) for x in zip(model_run, forecast)]
    EC_tephi_list = sorted([x for x in EC_tephi_list if x.split('_')[-3] + '_' + x.split('_')[-2] in start_end])
    return EC_tephi_list, model_run, forecast


def get_ec_df(EC_tephi_list, feature_to_scale, scaler):
    ver_df = pd.DataFrame()

    for i in range(len(EC_tephi_list)):
        data = pd.read_csv(EC_tephi_list[i], skiprows=4, delim_whitespace=True, names=["level", "PRES", "TEMP", "DWPT", "U", "V", "SPFH"])
        data = data.loc[data['PRES']>=50]
        data['model_run'] = EC_tephi_list[i].split('_')[-3]
        data['forecast'] = EC_tephi_list[i].split('_')[-2]
        data = data.sort_values(by=['PRES']).reset_index(drop=True)

        data['Wdir'] = data[['U', 'V']].apply(get_wdir_from_uv, axis=1)
        data['Wspd'] = np.sqrt(data['U']**2 + data['V']**2)

        ver_df = pd.concat([ver_df, data])
        
    ver_df['depression'] = ver_df['TEMP'] - ver_df['DWPT']
    feature_scaled = [i + '_scaled' for i in feature_to_scale]
    ver_df[feature_scaled] = scaler.transform(ver_df[feature_to_scale].values)
        
    return ver_df


def copy_ec_tephi_and_save_sounding(source_dir, target_dir, df, feature_scaled, model_run, forecast):
    copy_tephi(source_dir, target_dir, model_run=model_run, forecast=forecast)
    save_scaled_sounding_data(df, target_dir, feature_scaled, model_run=model_run, forecast=forecast)


def process_path_ec(img_path, npy_path):
    encoded_img = read_img(img_path)
    img = decode_img(encoded_img)
    nparray = tf_npload(npy_path)
    return (img, nparray), None


def verify_ecmwf(start_UTC, lead_day, model, feature_to_scale, scaler, batch_size, df_dv, n_class, public_dir_ec, ver_dir): 
    # start_UTC='00'/'12' ; lead_day=1,2,...9
    EC_tephi_list, model_run, forecast = get_ec_data_list(start_UTC=start_UTC, lead_day=lead_day)
    ec_df = get_ec_df(EC_tephi_list, feature_to_scale, scaler)
    feature_scaled = [i + '_scaled' for i in feature_to_scale]
    target_dir = os.path.join(ver_dir, 'ec_tephi_data_{}Z_lead-{}d'.format(start_UTC, lead_day))
    
    for mr, fc in zip(model_run, forecast):
        source_dir = os.path.join(public_dir_ec, mr)
        copy_ec_tephi_and_save_sounding(source_dir, target_dir, ec_df, feature_scaled, mr, fc)
    
    ec_img = sorted(glob.glob(os.path.join(ver_dir, 'ec_tephi_data_{}Z_lead-{}d/*.png'.format(start_UTC, lead_day))))
    ec_date = [x.split('_')[-1].split('.')[0][:-4] + '00' for x in ec_img]
    df_dv = df_dv[df_dv['date'].apply(lambda x: x in ec_date)]
    ec_date = df_dv['date'].values
    ec_img = [x for x in ec_img if x.split('_')[-1].split('.')[0][:-4] + '00' in ec_date]
    ec_npy = [x.replace('.png', '.npy') for x in ec_img]
    ec_dataset = tf.data.Dataset.from_tensor_slices((ec_img, ec_npy))
    
    AUTOTUNE = tf.data.AUTOTUNE
    ec_dataset = ec_dataset.map(process_path_ec, num_parallel_calls=AUTOTUNE)
    ec_dataset = ec_dataset.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
    ec_pred = model.predict(ec_dataset)
    
    y_true = df_dv['dv_class'].values
    return y_true, ec_pred, ec_date, ec_img, ec_dataset


def get_9day_result(model, feature_to_scale, scaler, batch_size, df_dv, n_class, public_dir_ec, ver_dir, directory=None):
    df_9day_00 = pd.DataFrame()
    df_9day_12 = pd.DataFrame()
    for start_UTC in ['00', '12']:
        for lead_day in range(1, 10):
            y_true, ec_pred, ec_date, ec_img, ec_dataset = verify_ecmwf(start_UTC, lead_day, model, feature_to_scale, scaler, 
                                                                        batch_size, df_dv, n_class, public_dir_ec, ver_dir)
            ec_pred_label = get_pred_label(ec_pred, threshold=0.5)
            ec_pred_class = get_str_class(ec_pred_label, n_class)
            
            day = pd.DataFrame({'day':[lead_day]*n_class})
            result = get_result(ec_pred_class, y_true, to_file=None)
            df_one_day = pd.concat([day, result], axis=1)
            
            if start_UTC == '00':
                df_9day_00 = pd.concat([df_9day_00, df_one_day], axis=0)
            elif start_UTC == '12':
                df_9day_12 = pd.concat([df_9day_12, df_one_day], axis=0)
                
    if directory is not None:
        df_9day_00.to_csv(os.path.join(directory, 'model_9day_result_00Z.csv'), index=False)
        df_9day_12.to_csv(os.path.join(directory, 'model_9day_result_12Z.csv'), index=False)
    
    return df_9day_00, df_9day_12
