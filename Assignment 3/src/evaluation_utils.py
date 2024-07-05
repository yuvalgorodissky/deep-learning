import csv
import datetime
import glob
import torch


#
# def jaccard_similarity_overall(predictions, targets):
#     # Initialize variables to store the total intersection and union sizes
#     results = []
#     # Ensure predictions and targets are paired correctly
#     for pred, targ in zip(predictions, targets):
#         # Convert lists to sets for set operations
#         pred_set = set(pred)
#         targ_set = set(targ)
#
#         # Calculate the intersection and union of the two sets
#         intersection = pred_set.intersection(targ_set)
#         union = pred_set.union(targ_set)
#
#         # Calculate the Jaccard similarity index
#         if len(union) == 0:
#             similarity = 0.0
#         else:
#             similarity = len(intersection) / len(union)
#         results.append(similarity)
#
#
#     return results


def generate_ngrams(text, n):
    # Split the text into words and generate n-grams
    words = text.split()
    return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]


def calculate_jacard_similarity(song1, song2, n):
    # Generate n-grams for both songs
    ngrams1 = generate_ngrams(song1, n)
    ngrams2 = generate_ngrams(song2, n)

    # Use sets to find common n-grams and calculate similarity
    set1 = set(ngrams1)
    set2 = set(ngrams2)
    common_ngrams = set1.intersection(set2)
    total_ngrams = set1.union(set2)

    # Avoid division by zero
    if not total_ngrams:
        return 0.0

    # Calculate similarity as the ratio of common n-grams to total n-grams
    similarity = len(common_ngrams) / len(total_ngrams)
    return similarity


def compare_songs(prediction_songs, target_songs, n):
    # Ensure the lists are of the same length
    if len(prediction_songs) != len(target_songs):
        raise ValueError("Error: Song lists must be of the same length.")

    # Calculate similarity for each pair of songs
    results = []
    for pred, target in zip(prediction_songs, target_songs):
        similarity_score = calculate_jacard_similarity(pred, target, n)
        results.append(similarity_score)
    return results


def load_model_path_eval(args, path):
    from model import lstm_seq2seq
    files = glob.glob(path + "*")
    size = int(files[-1].split("_")[-1].split(".")[0])
    model = lstm_seq2seq(
        input_size_encoder=args.input_size_encoder,
        hidden_size_encoder=args.hidden_size_encoder,
        input_size_decoder=args.input_size_decoder,
        hidden_size_decoder=args.hidden_size_decoder,
        vect_size_decoder=size,
        word2vec_model=None,  # Placeholder, will be replaced during loading
        word_to_index=None,  # Placeholder, will be replaced during loading
        vocabulary=None,  # Placeholder, will be replaced during loading
        num_layers=args.num_layers
    )

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.word2vec = checkpoint['word2vec']
    model.word_to_index = checkpoint['word_to_index']
    model.vocabulary = checkpoint['vocabulary']
    print("Model and additional information loaded from", path)
    return model


import csv
import datetime


def export_result_dict(result_dict,output_path):
    # Format the current date and time as a string that is safe for filenames
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the path with the formatted datetime
    filename = f'{output_path}/results_' + now + '.csv'

    # Open the file for writing
    with open(filename, 'w', newline='') as f:
        # Create a CSV writer object
        writer = csv.writer(f)

        # Write the header row
        headers = ['Model', 'Strategy', 'Start_Token']
        # Append headers for each song and Jaccard score
        for i in range(1, 6):  # Assuming there are 5 songs
            headers.extend([f'Song-{i}_jacard_1', f'Song-{i}_jacard_2'])
        writer.writerow(headers)

        # Prepare to write data rows
        # Organize data by (model_path, strategy, token_kind)
        organized_results = {}
        for key, value in result_dict.items():
            song_idx, model_path, strategy, token_kinds = key
            model_path = " ".join(model_path.split("_")[:-1])
            _, _, jacard_result_n_1, jacard_result_n_2 = value
            jacard_result_n_1 = round(jacard_result_n_1,4)
            jacard_result_n_2 = round(jacard_result_n_2,4)

            # Initialize the key if not already present
            if (model_path, strategy, token_kinds) not in organized_results:
                organized_results[(model_path, strategy, token_kinds)] = [[] for _ in
                                                                          range(5)]  # 5 songs, 2 scores each

            # Append the scores in the correct position based on song_idx
            organized_results[(model_path, strategy, token_kinds)][song_idx].extend(
                [jacard_result_n_1, jacard_result_n_2])

        # Write organized results to CSV
        for (model_path, strategy, token_kinds), scores_list in organized_results.items():
            # Start row with model, strategy, and token type
            row = [model_path, strategy, token_kinds]

            # Flatten the scores list and append to the row
            for scores in scores_list:
                row.extend(scores)

            # Write the complete row
            writer.writerow(row)

