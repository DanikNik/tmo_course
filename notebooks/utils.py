import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, accuracy_score, plot_confusion_matrix,
                             roc_curve, f1_score, mean_absolute_error, mean_squared_error, r2_score, )

# some mappings to clarify the dataset
MAPPINGS = dict()


def create_attribute_mapping(attribute):
    attr_values = attribute.unique()
    mapping = dict(zip(attr_values, range(len(attr_values))))
    MAPPINGS[attribute.name] = mapping
    return mapping


class MetricLogger:

    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
             'alg': pd.Series([], dtype='str'),
             'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric'] == metric) & (self.df['alg'] == alg)].index, inplace=True)
        # Добавление нового значения
        temp = [{'metric': metric, 'alg': alg, 'value': value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric'] == metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values

    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5,
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a, b in zip(pos, array_metric):
            plt.text(0.5, a - 0.05, str(round(b, 3)), color='white')
        plt.show()

METRIC_LOGGER = MetricLogger()

# Отрисовка ROC-кривой
# def draw_roc_curve(y_true, y_score, pos_label=1, average='micro'):
#     fpr, tpr, thresholds = roc_curve(y_true, y_score,
#                                      pos_label=pos_label)
#     roc_auc_value = roc_auc_score(y_true, y_score, average=average)
#     plt.figure()
#     lw = 2
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.show()


# def classification_train_model(x_train, y_train, x_test, y_test, model_name, model, clasMetricLogger):
#     model.fit(x_train, y_train)
#     Y_pred = model.predict(x_test)
#     precision = precision_score(y_test.values, Y_pred)
#     recall = recall_score(y_test.values, Y_pred)
#     f1 = f1_score(y_test.values, Y_pred)
#     roc_auc = roc_auc_score(y_test.values, Y_pred)
#
#     clasMetricLogger.add('precision', model_name, precision)
#     clasMetricLogger.add('recall', model_name, recall)
#     clasMetricLogger.add('f1', model_name, f1)
#     clasMetricLogger.add('roc_auc', model_name, roc_auc)
#
#     print('*****************************************************')
#     print(model)
#     print('*****************************************************')
#     draw_roc_curve(y_test.values, Y_pred)
#
#     plot_confusion_matrix(model, x_test, y_test.values,
#                           display_labels=['0', '1'],
#                           cmap=plt.cm.Blues, normalize='true')
#     plt.show()


def regression_train_model(x_train, y_train, x_test, y_test, model_name, model, metric_logger):
    model.fit(x_train, y_train)
    Y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, Y_pred)
    mse = mean_squared_error(y_test, Y_pred)
    r2 = r2_score(y_test, Y_pred)

    metric_logger.add('MAE', model_name, mae)
    metric_logger.add('MSE', model_name, mse)
    metric_logger.add('R2', model_name, r2)

    print('*****************************************************')
    print(model)
    print()
    print('MAE={}, MSE={}, R2={}'.format(
        round(mae, 3), round(mse, 3), round(r2, 3)))
    print('*****************************************************')
