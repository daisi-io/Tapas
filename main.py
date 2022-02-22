import pandas as pd
from predict import unzip_model, load_model_and_tokenizer, prepare_inputs, generate_predictions, postprocess_predictions, show_answers

# get model_zip_path
# model_zip_path = "/pebble_source/90ce7da0-3233-4665-84fd-305adb0883b3/tapas-base-finetuned-wtq.zip"
# model_zip_path = "/Users/zhenshanjin/Documents/Belmont/sandy/UtilityDaisies/tapas-base-finetuned-wtq/tmp/tapas-base-finetuned-wtq.zip"
# model_path = unzip_model(model_zip_path)
model_path = "/pebble_tmp/tmp/model_tapas"
tokenizer, model = load_model_and_tokenizer(model_path)


def run_tapas(table_json_path:pd.DataFrame, query):
    """
    Invoke the TAPAS model.
    """
    queries = [query]
    table, inputs = prepare_inputs(table_json_path, queries, tokenizer)
    predicted_table_cell_coords, predicted_aggregation_operators = generate_predictions(inputs, model, tokenizer)
    aggregation_predictions_string, answers = postprocess_predictions(predicted_aggregation_operators, predicted_table_cell_coords, table)
    answers = show_answers(queries, answers, aggregation_predictions_string)

    return {"answers": answers}


if __name__ == '__main__':
    # Define the table
    # data = {'Cities': ["Paris, France", "London, England", "Lyon, France"], 'Inhabitants': ["2.161", "8.982", "0.513"]}
    data = "/path/to/table/json"
    # Define the questions
    # queries = ["Which city has most inhabitants?", "What is the average number of inhabitants?", "How many French cities are in the list?", "How many inhabitants live in French cities?"]
    query = "Which city has most inhabitants?"
    res = run_tapas(data, query)