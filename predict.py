import os
import pathlib
from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import zipfile
import json
import tempfile

root_folder = tempfile.gettempdir()

def unzip_model(zip_model_path):
    p = pathlib.Path(zip_model_path)
    # root = str(p.parents[0])
    print(f'start unzip: {root_folder}')
    model_path = os.path.join(root_folder, os.path.splitext(os.path.basename(zip_model_path))[0] + '/')
    if not os.path.isdir(model_path):
        with zipfile.ZipFile(zip_model_path, "r") as zip_ref:
            zip_ref.extractall(model_path)
        
    print(f"finish zipping model to {model_path}")
    return model_path


def load_model_and_tokenizer(model_path):
    """
    Load
    """
    # Load pretrained tokenizer: TAPAS finetuned on WikiTable Questions
    tokenizer = TapasTokenizer.from_pretrained(model_path)

    # Load pretrained model: TAPAS finetuned on WikiTable Questions
    model = TapasForQuestionAnswering.from_pretrained(model_path)

    # Return tokenizer and model
    return tokenizer, model


def prepare_inputs(table_json_path, queries, tokenizer):
    """
    Convert dictionary into data frame and tokenize inputs given queries.
    """
    with open(table_json_path, "r") as istr:
        data = json.load(istr)
    # Prepare inputs
    table = pd.DataFrame.from_dict(data)
    inputs = tokenizer(table=table, queries=queries, padding='max_length', return_tensors="pt")

    # Return things
    return table, inputs


def generate_predictions(inputs, model, tokenizer):
    """
    Generate predictions for some tokenized input.
    """
    # Generate model results
    outputs = model(**inputs)

    # Convert logit outputs into predictions for table cells and aggregation operators
    predicted_table_cell_coords, predicted_aggregation_operators = tokenizer.convert_logits_to_predictions(
            inputs,
            outputs.logits.detach(),
            outputs.logits_aggregation.detach()
    )

    # Return values
    return predicted_table_cell_coords, predicted_aggregation_operators


def postprocess_predictions(predicted_aggregation_operators, predicted_table_cell_coords, table):
    """
    Compute the predicted operation and nicely structure the answers.
    """
    # Process predicted aggregation operators
    aggregation_operators = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3:"COUNT"}
    aggregation_predictions_string = [aggregation_operators[x] for x in predicted_aggregation_operators]

    # Process predicted table cell coordinates
    answers = []
    for coordinates in predicted_table_cell_coords:
        if len(coordinates) == 1:
            # 1 cell
            answers.append(table.iat[coordinates[0]])
        else:
            # > 1 cell
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
                answers.append(", ".join(cell_values))
        
    # Return values
    return aggregation_predictions_string, answers


def show_answers(queries, answers, aggregation_predictions_string):
    """
    Visualize the postprocessed answers.
    """
    result = []
    for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
        res = {"query": query}
        if predicted_agg == "NONE":
            res["answer"] = answer
        else:
            res["answer"] = predicted_agg + " > " + answer
        result.append(res)

    return result
