import click
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def evaluate_classification(results_rep, results_vbow):
    data = results_rep['data']
    classifier = results_rep['algorithm']
    click.echo('\nResults for classification of {} with {}:\n'.format(data, classifier))
    table = PrettyTable(['', 'Runtime', 'Accuracy'])
    table.add_row(['feature representation',
                   '{}s'.format(round(results_rep['runtime'], 2)),
                   '{}'.format(round(results_rep['accuracy'], 4))])
    table.add_row(['visual bag of words',
                   '{}s'.format(round(results_vbow['runtime'], 2)),
                   '{}'.format(round(results_vbow['accuracy'], 4))])
    mean_acc = (results_rep['accuracy'] + results_vbow['accuracy']) / 2
    mean_runtime = (results_rep['runtime'] + results_vbow['runtime']) / 2
    table.add_row(['mean',
                   '{}s'.format(round(mean_runtime, 2)),
                   '{}'.format(round(mean_acc, 4))])
    click.echo(table)


def create_confusion_matrix(prediction, test_labels, data, algorithm, input_data=None):
    # Plot non-normalized confusion matrix
    label_names = get_label_names(data)
    cm = confusion_matrix(test_labels, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues)
    title = 'Confusion matrix for {} with {}'.format(data, algorithm)
    if input_data is not None:
        title += '\nInput data: {}'.format(input_data)
    disp.ax_.set_title(title)
    plt.xticks(rotation=45)
    click.echo('Close the figure to continue...')
    plt.show()


def get_label_names(data):
    if data == 'CIFAR-10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                'Ankle boot']
