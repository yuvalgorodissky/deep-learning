import numpy as np
from matplotlib import pyplot as plt



def collect_samples_indices(dataloader, preds):
    # Initialize the dictionary with categories of interest
    samples_dict = {
        'true_positive': [],  # Predicted 1, actual 1
        'false_positive': [],  # Predicted 1, actual 0
        'true_negative': [],  # Predicted 0, actual 0
        'false_negative': []  # Predicted 0, actual 1
    }

    for i, pred in enumerate(preds):
        label = dataloader.dataset.pairs[i][2]
        image1 = dataloader.dataset.pairs[i][0]
        image2 = dataloader.dataset.pairs[i][1]
        if pred == 1 and label == 1:
            option = 'true_negative'
        elif pred == 1 and label == 0:
            option = 'false_negative'
        elif pred == 0 and label == 0:
            option = 'true_positive'
        else:
            option = 'false_positive'
        samples_dict[option].append((image1, image2))

    return samples_dict



def save_samples_to_file(samples_dict):
    for key, value in samples_dict.items():
        # Use the first two pairs of samples if available
        pairs_to_display = value[:2] if len(value) >= 2 else value

        # Create a single figure with rows for the number of pairs and 2 columns
        if pairs_to_display:  # Check if there are pairs to display
            fig, axs = plt.subplots(len(pairs_to_display), 2, figsize=(8, 4 * len(pairs_to_display)), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})

            # Configure the subplots to minimize spacing and remove axes
            for i, pair in enumerate(pairs_to_display):
                for j in range(2):
                    ##load the images from the path
                    path_to_image1 = pair[j]
                    loaded_image1 = plt.imread(path_to_image1)
                    axs[i, j].imshow(loaded_image1, cmap='gray')
                    axs[i, j].axis('off')
                    axs[i, j].set_xticklabels([])
                    axs[i, j].set_yticklabels([])
                    axs[i, j].set_aspect('auto')

            plt.subplots_adjust(wspace=0, hspace=0)  # Adjust the spacing between images
            plt.savefig(f'images/{key}.png', bbox_inches='tight', pad_inches=0)  # Save the figure with less padding
            plt.show()  # Display the figure
            plt.close()

def confusion_matrix(samples_dict):
    # Define labels for the categories
    categories = ['Positive', 'Negative']
    # Map the counts to the correct categories for a confusion matrix
    counts = [
        len(samples_dict['true_positive']),  # True Positive
        len(samples_dict['false_positive']),  # False Positive
        len(samples_dict['false_negative']),  # False Negative
        len(samples_dict['true_negative'])    # True Negative
    ]
    # Create a 2x2 confusion matrix
    confusion_mat = np.array(counts).reshape((2, 2))

    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_mat, cmap='Blues')
    plt.title('Confusion Matrix')
    fig.colorbar(cax)

    # Set ticks and tick labels
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)

    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(confusion_mat):
        ax.text(j, i, str(val), ha='center', va='center', color='black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('images/confusion_matrix.png')
    plt.show()
    plt.close()