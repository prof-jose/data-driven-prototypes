"""
Create an index of the predicted words for each image in a directory.

Usage: python index.py <directory_name>
"""

import os
import sys
import pandas as pd
from PIL import Image
from transformers import pipeline

TH = 0.4  # Probability threshold to consider a prediction


def process_result(result):
    """
    Iterate over the classification results to return a list of
    words with probability higher than TH.

    Parameters
    ----------
    result : dict
        The classification results.

    Returns
    -------
    list
        A list of words with probability higher than TH.
    """
    output = []
    for classification in result:
        # Print the key and the value
        if classification["score"] > TH:
            print(classification)
            classification["label"] = \
                classification["label"].replace("_", ", ")
            words = classification["label"].replace(",", "").split(" ")
            for word in words:
                output.append(word)
    return output


def process_dir(folder, model):
    """
    Classify all files in a directory and add the results to an index.

    Parameters
    ----------
    folder : str
        The path to the directory.
    model : torch.nn.Module
        The model to use for classification.

    Returns
    -------
    pandas.DataFrame
        A dataframe with the classification results.

    """
    # Create empty dataframe with two columns, "word" and "file"
    df_index = pd.DataFrame(columns=["word", "file"])

    for file in os.listdir(folder):
        if file.endswith(".jpeg") or file.endswith(".png"):
            full_path = os.path.join(folder, file)
            image = Image.open(full_path)
            result = model.predict(image)
            output = process_result(result)

            # Create a dataframe whith a row for each word
            df = pd.DataFrame(output, columns=["word"])
            df['file'] = full_path

            # Append the dataframe to the empty dataframe
            df_index = pd.concat([df_index, df], ignore_index=True)

    return df_index


def main():
    """
    Run the main function.
    """
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)

    model = pipeline("image-classification", model="microsoft/resnet-50")

    folder = sys.argv[1]
    index = process_dir(folder, model)
    index.drop_duplicates(inplace=True)
    index.to_csv("index.csv", index=False)


if __name__ == "__main__":
    main()
