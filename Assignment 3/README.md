# Deep Learning Assignment 3

## Overview
This project aims to build a recurrent neural network (RNN) for generating lyrics based on provided melodies using MIDI files. The model is trained to predict the next word of the lyrics given the current word and the melody information.

## Structure
- `data/`: Contains MIDI files and lyrics data.
- `src/`: Source code for data preprocessing, model definition, training, and lyric generation.
- `notebooks/`: Jupyter notebook for the assignment.
- `reports/`: Report documenting the approach and results.
- `requirements.txt`: List of required Python packages.

## Instructions
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Preprocess the data:
   2.1 using lowercase names:
```bash
   cd "/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/data/midi_files"

# Loop through all files in the directory
for file in *; do
  # Check if the file is not already lowercase
  if [ "$file" != "${file,,}" ]; then
    # Rename the file to lowercase
    mv "$file" "${file,,}"
  fi
done
   ```
         ```bash
         python src/data_preprocessing.py
         ```

awk -F, '{
    # Output the first two comma-separated fields as they are
    printf "%s,%s", $1, $2;
    
    # Combine the rest of the fields into one, removing additional commas
    rest = "";
    for (i = 3; i <= NF; i++) {
        rest = rest $i;  # Append each field without adding commas
    }
    print "," rest;  # Print the combined rest with a leading comma
}' input_file > output_file

3. Train the model:
   ```bash
   python src/train.py
   ```

4. Generate lyrics:
   ```bash
   python src/generate_lyrics.py
   ```

5. View the results and analysis in the `assignment3.ipynb` notebook.

## Submission
Submit the entire project as a single zip file (`Assignment3.zip`) containing both the report (in PDF form) and the code.

