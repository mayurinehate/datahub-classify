import glob
import itertools
import json
import logging
import os
import re
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import classification_report, confusion_matrix

from datahub_classify.helper_classes import (
    ColumnInfo,
    ColumnMetadata,
    TableInfo,
    TableMetadata,
)
from datahub_classify.similarity_predictor import check_similarity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("libraries Imported..................")

SEED = 100
PRUNING_THRESHOLD = 0.8
FINAL_THRESHOLD = 0.6
COLUMN_SIMILARITY_THRESHOLD = 0.8
PLATFORMS = ["A", "B", "C", "D", "E"]


def load_df(dataset_name):
    path = all_datasets_paths[dataset_name]
    if path.endswith("csv"):
        df = pd.read_csv(path, nrows=2)
    elif path.endswith("xlsx"):
        df = pd.read_excel(path, nrows=2)
    else:
        df = None
    return df


def load_jsons(input_jsons_dir):
    with open(
            os.path.join(input_jsons_dir, "table_similarity_labels_IDEAL.json")
    ) as filename:
        table_similarity_labels_ideal_ = json.load(filename)
    with open(
            os.path.join(input_jsons_dir, "pruning_table_similarity_labels_EXPECTED.json")
    ) as filename:
        pruning_table_similarity_labels_expected_ = json.load(filename)
    with open(
            os.path.join(
                input_jsons_dir, "post_pruning_table_similarity_labels_EXPECTED.json"
            )
    ) as filename:
        post_pruning_table_similarity_labels_expected_ = json.load(filename)

    with open(
            os.path.join(input_jsons_path, "column_similarity_scores_EXPECTED.json")
    ) as filename_:
        column_similarity_scores_expected_ = json.load(filename_)

    with open(
            os.path.join(input_jsons_path, "column_similarity_labels_IDEAL.json")
    ) as filename_:
        column_similarity_labels_ideal_ = json.load(filename_)

    return (
        table_similarity_labels_ideal_,
        pruning_table_similarity_labels_expected_,
        post_pruning_table_similarity_labels_expected_,
        column_similarity_scores_expected_,
        column_similarity_labels_ideal_,
    )


def get_predicted_expected_similarity_scores_mapping_for_tables(
        predicted_similarity_labels_unit_testing,
        expected_similarity_labels_unit_testing,
):
    """generate mapping of predicted - expected similarity scores, required for unit testing"""

    mapping = []
    for key_ in predicted_similarity_labels_unit_testing.keys():
        pair = key_.split("_SPLITTER_", 1)
        if expected_similarity_labels_unit_testing.get(key_, None):
            expected_similarity_label_unit_testing = (
                expected_similarity_labels_unit_testing[key_]
            )
            mapping.append(
                (
                    pair[0],
                    pair[1],
                    predicted_similarity_labels_unit_testing[key_],
                    expected_similarity_label_unit_testing,
                )
            )
    return mapping


def get_predicted_expected_similarity_scores_mapping_for_columns(
        predicted_similarity_scores_unit_testing,
        expected_similarity_scores_unit_testing,
):
    """generate mapping of predicted - expected similarity scores, required for unit testing"""

    mapping = []
    for pair in predicted_similarity_scores_unit_testing.keys():
        key_ = f"{pair[0]}_COLSPLITTER_{pair[1]}"
        if key_ in expected_similarity_scores_unit_testing.keys():
            if expected_similarity_scores_unit_testing[key_] is None:
                expected_similarity_label = "not_similar"
                expected_similarity_score = 0
            elif (
                    expected_similarity_scores_unit_testing[key_]
                    >= COLUMN_SIMILARITY_THRESHOLD
            ):
                expected_similarity_label = "similar"
                expected_similarity_score = expected_similarity_scores_unit_testing[
                    key_
                ]
            else:
                expected_similarity_label = "not_similar"
                expected_similarity_score = expected_similarity_scores_unit_testing[
                    key_
                ]

            if predicted_similarity_scores_unit_testing[pair] is None:
                predicted_similarity_label = "not_similar"
                predicted_similarity_score = 0
            elif (
                    predicted_similarity_scores_unit_testing[pair]
                    >= COLUMN_SIMILARITY_THRESHOLD
            ):
                predicted_similarity_label = "similar"
                predicted_similarity_score = predicted_similarity_scores_unit_testing[
                    pair
                ]
            else:
                predicted_similarity_label = "not_similar"
                predicted_similarity_score = predicted_similarity_scores_unit_testing[
                    pair
                ]
            column_similarity_predicted_labels[key_] = predicted_similarity_label
            mapping.append(
                (
                    pair[0],
                    pair[1],
                    predicted_similarity_score,
                    predicted_similarity_label,
                    expected_similarity_score,
                    expected_similarity_label,
                )
            )
    return mapping


def generate_report_for_table_similarity(true_labels, predicted_labels, threshold=None):
    keys = list(predicted_labels.keys())
    y_pred = [0 if predicted_labels[key] == "not_similar" else 1 for key in keys]
    y_true = [0 if true_labels[key] == "not_similar" else 1 for key in keys]
    target_names = ["not_similar", "similar"]
    misclassification_report: Dict[str, list] = {"tp": [], "fp": [], "tn": [], "fn": []}
    for i in range(len(keys)):
        if y_true[i] == 0 and y_pred[i] == 0:
            misclassification_report["tn"].append(keys[i])
        elif y_true[i] == 0 and y_pred[i] == 1:
            misclassification_report["fp"].append(keys[i])
        elif y_true[i] == 1 and y_pred[i] == 0:
            misclassification_report["fn"].append(keys[i])
        else:
            misclassification_report["tp"].append(keys[i])
    return (
        confusion_matrix(y_true, y_pred),
        classification_report(y_true, y_pred, target_names=target_names),
        misclassification_report,
    )


def generate_report_for_column_similarity(
        true_labels, predicted_labels
):
    keys = list(predicted_labels.keys())
    y_pred_labels = []
    y_true_labels = []
    for key in keys:
        y_pred_labels.append(predicted_labels[key])
        if key not in true_labels.keys():
            y_true_labels.append("not_similar")
        else:
            y_true_labels.append(true_labels[key])

    y_pred = [0 if label == "not_similar" else 1 for label in y_pred_labels]
    y_true = [0 if label == "not_similar" else 1 for label in y_true_labels]
    target_names = ["not_similar", "similar"]
    misclassification_report: Dict[str, list] = {"tp": [], "fp": [], "tn": [], "fn": []}
    for i in range(len(keys)):
        if y_true[i] == 0 and y_pred[i] == 0:
            misclassification_report["tn"].append(keys[i])
        elif y_true[i] == 0 and y_pred[i] == 1:
            misclassification_report["fp"].append(keys[i])
        elif y_true[i] == 1 and y_pred[i] == 0:
            misclassification_report["fn"].append(keys[i])
        else:
            misclassification_report["tp"].append(keys[i])
    return (
        confusion_matrix(y_true, y_pred),
        classification_report(y_true, y_pred, target_names=target_names),
        misclassification_report,
    )


def populate_tableinfo_object(dataset_name):
    """populate table info object for a dataset"""
    df = load_df(dataset_name)
    np.random.seed(SEED)
    table_meta_info = {
        "Name": dataset_name,
        "Description": f"This table contains description of {dataset_name}",
        "Platform": PLATFORMS[np.random.randint(0, 5)],
        "Table_Id": dataset_name,
    }
    col_infos = []
    for col in df.columns:
        fields = {
            "Name": col,
            "Description": f" {col}",
            "Datatype": str(df[col].dropna().dtype),
            "Dataset_Name": dataset_name,
            "Column_Id": dataset_name + "_SPLITTER_" + col,
        }
        metadata_col = ColumnMetadata(fields)
        # parent_cols = list()
        col_info_ = ColumnInfo(metadata_col)
        col_infos.append(col_info_)

    metadata_table = TableMetadata(table_meta_info)
    # parent_tables = list()
    table_info = TableInfo(metadata_table, col_infos)
    return table_info


def populate_similar_tableinfo_object(dataset_name):
    """populate table info object for a dataset by randomly adding some additional
    columns to the dataset, thus obtain a logical copy of input dataset"""
    df = load_df(dataset_name)
    df.columns = ["data1" + "_" + col for col in df.columns]
    random_df_key = list(all_datasets_paths.keys())[
        np.random.randint(0, len(all_datasets_paths))
    ]
    while random_df_key == dataset_name:
        random_df_key = list(all_datasets_paths.keys())[
            np.random.randint(0, len(all_datasets_paths))
        ]
    random_df = load_df(random_df_key).copy()
    random_df.columns = ["data2" + "_" + col for col in random_df.columns]
    second_df = pd.concat([df, random_df], axis=1)
    cols_to_keep = list(df.columns) + list(random_df.columns[:2])
    second_df = second_df[cols_to_keep]
    np.random.seed(SEED)
    table_meta_info = {
        "Name": dataset_name + "_LOGICAL_COPY",
        "Description": f" {dataset_name}",
        "Platform": PLATFORMS[np.random.randint(0, 5)],
        "Table_Id": dataset_name + "_LOGICAL_COPY",
    }
    col_infos = []
    swap_case = ["yes", "no"]
    # fmt:off
    common_variations = ["#", "$", "%", "&", "*", "-", ".", ":", ";", "?",
                         "@", "_", "~", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                         ]
    # fmt:on
    for col in second_df.columns:
        col_name = col.split("_", 1)[1]
        col_name_with_variation = str(
            common_variations[np.random.randint(0, len(common_variations))]
        )
        for word in re.split("[^A-Za-z]", col_name):
            random_variation = str(
                common_variations[np.random.randint(0, len(common_variations))]
            )
            is_swap_case = swap_case[np.random.randint(0, 2)]
            if is_swap_case:
                word = word.swapcase()
            col_name_with_variation = col_name_with_variation + word + random_variation

        fields = {
            "Name": col_name_with_variation,
            "Description": f'{col.split("_", 1)[1]}',
            "Datatype": str(second_df[col].dropna().dtype),
            "Dataset_Name": dataset_name + "_LOGICAL_COPY",
            "Column_Id": dataset_name
                         + "_LOGICAL_COPY_"
                         + "_SPLITTER_"
                         + col.split("_", 1)[1],
        }
        metadata_col = ColumnMetadata(fields)
        parent_cols = [col if col in df.columns else None]
        col_info_ = ColumnInfo(metadata_col)
        col_info_.parent_columns = parent_cols
        col_infos.append(col_info_)
    metadata_table = TableMetadata(table_meta_info)
    parent_tables = [dataset_name]
    table_info = TableInfo(metadata_table, col_infos, parent_tables)
    return table_info


column_similarity_predicted_labels: Dict[str, str] = dict()
columns_predicted_scores: Dict[str, float] = dict()
current_wdr = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_wdr, "datasets")
input_jsons_path = os.path.join(current_wdr, "expected_output")

all_datasets_paths = {
    os.path.basename(file).rsplit(".", 1)[0]: file
    for file in glob.glob(f"{input_dir}/*")
}

pruning_mode_output_PREDICTED: Dict[str, str] = dict()
post_pruning_mode_output_PREDICTED: Dict[str, str] = dict()
pruning_mode_results: Dict[str, Tuple] = dict()
post_pruning_mode_results: Dict[str, Tuple] = dict()

(
    table_similarity_labels_ideal,
    pruning_table_similarity_labels_expected,
    post_pruning_table_similarity_labels_expected,
    column_similarity_scores_expected,
    column_similarity_labels_ideal,
) = load_jsons(input_jsons_path)

logger.info("Creating Tables Info Objects.............")
table_infos = {key: populate_tableinfo_object(key) for key in all_datasets_paths.keys()}
table_info_copies = {
    f"{key}_LOGICAL_COPY": populate_similar_tableinfo_object(key)
    for key in all_datasets_paths.keys()
}

logger.info("Creating Table Pairs List................")
table_pairs = list(itertools.combinations(table_infos.keys(), 2))
table_infos.update(table_info_copies)
for key in all_datasets_paths.keys():
    table_pairs.append((key, f"{key}_LOGICAL_COPY"))

logger.info("Starting check similarity.............")
pruning_mode_start_time = time.time()
for table_pair in table_pairs:
    table_pair = sorted(table_pair, key=str.lower)
    pruning_mode_results[
        f"{table_pair[0]}_SPLITTER_{table_pair[1]}"
    ] = check_similarity(
        table_infos[table_pair[0]],
        table_infos[table_pair[1]],
        pruning_mode=True,
        use_embeddings=False,
    )
pruning_mode_end_time = time.time()
pruning_mode_output_PREDICTED = {
    key: ("not_similar" if value[0].score <= PRUNING_THRESHOLD else "similar")
    for key, value in pruning_mode_results.items()
}

post_pruning_mode_combinations = [
    key for key, value in pruning_mode_output_PREDICTED.items() if value == "similar"
]

post_pruning_mode_start_time = time.time()
for comb in post_pruning_mode_combinations:
    tables = comb.split("_SPLITTER_")
    post_pruning_mode_results[comb] = check_similarity(
        table_infos[tables[0]],
        table_infos[tables[1]],
        pruning_mode=False,
        use_embeddings=False,
    )
post_pruning_mode_end_time = time.time()
total_pruning_time = pruning_mode_end_time - pruning_mode_start_time
total_post_pruning_time = post_pruning_mode_end_time - post_pruning_mode_start_time

post_pruning_mode_output_PREDICTED = {
    key: ("not_similar" if value[0].score <= FINAL_THRESHOLD else "similar")
    for key, value in post_pruning_mode_results.items()
}

pruning_tables_similarity_mapping_unit_testing = (
    get_predicted_expected_similarity_scores_mapping_for_tables(
        pruning_mode_output_PREDICTED, pruning_table_similarity_labels_expected
    )
)

for i, data_pair in enumerate(post_pruning_mode_results.keys()):
    for key, value in post_pruning_mode_results[data_pair][1].items():
        columns_predicted_scores[key] = value.score

columns_similarity_mapping_unit_testing = (
    get_predicted_expected_similarity_scores_mapping_for_columns(
        columns_predicted_scores,
        column_similarity_scores_expected,
    )
)

# ###############################
# # Generate Performance Report #
# ###############################
# pruning_report = generate_report(all_combinations_with_labels, pruning_mode_results, threshold=PRUNING_THRESHOLD)
pruning_report = generate_report_for_table_similarity(
    table_similarity_labels_ideal,
    pruning_mode_output_PREDICTED,
    threshold=PRUNING_THRESHOLD,
)
final_results = {}
for key in pruning_mode_output_PREDICTED.keys():
    if key in post_pruning_mode_output_PREDICTED:
        final_results[key] = post_pruning_mode_output_PREDICTED[key]
    else:
        final_results[key] = "not_similar"
post_pruning_report = generate_report_for_table_similarity(
    table_similarity_labels_ideal,
    post_pruning_mode_output_PREDICTED,
    threshold=FINAL_THRESHOLD,
)
final_table_report = generate_report_for_table_similarity(
    table_similarity_labels_ideal, final_results, threshold=FINAL_THRESHOLD
)
column_similarity_report = generate_report_for_column_similarity(
    column_similarity_labels_ideal, column_similarity_predicted_labels
)

with open("Similarity_predictions.txt", "w") as file_:
    # TABLE SIMILARITY REPORT:
    print("-------------------------------------------", file=file_)
    print(f"PRUNING THRESHOLD: {PRUNING_THRESHOLD}", file=file_)
    print(f"FINAL THRESHOLD: {FINAL_THRESHOLD}", file=file_)
    print("-------------------------------------------", file=file_)
    print("Number of Tables", len(table_infos), file=file_)
    print(
        "Total Pruning Time: ",
        total_pruning_time,
        f"for {len(pruning_mode_results)} pairs",
        file=file_,
    )
    print(
        "Total Post Pruning Time: ",
        total_post_pruning_time,
        f"for {len(post_pruning_mode_results)} pairs",
        file=file_,
    )
    print("Total Time", total_post_pruning_time + total_pruning_time, file=file_)

    print(
        "PRUNING MODE CLASSIFICATION REPORT\n",
        pruning_report[0],
        "\n",
        pruning_report[1],
        "\nFalse Negatives",
        pruning_report[2]["fn"],
        "\n",
        file=file_,
    )

    # # print("Total post pruning pairs: ", len(post_pruning_mode_results), file=f)
    print(
        "POST PRUNING MODE CLASSIFICATION REPORT\n",
        post_pruning_report[0],
        "\n",
        post_pruning_report[1],
        "\nFalse Negatives\n",
        post_pruning_report[2]["fn"],
        "\n",
        file=file_,
    )
    print(
        "FINAL CLASSIFICATION REPORT\n",
        final_table_report[0],
        "\n",
        final_table_report[1],
        "\nFalse Negatives\n",
        final_table_report[2]["fn"],
        "\n",
        file=file_,
    )

    # COLUMN SIMILARITY REPORT:
    print(
        "COLUMN SIMILARITY CLASSIFICATION REPORT\n",
        column_similarity_report[0],
        "\n",
        column_similarity_report[1],
        "\nFalse Negatives",
        column_similarity_report[2]["fn"],
        "\n",
        file=file_,
    )

############################
# Start unit testing #
############################
# Unit Test for Columns Similarity #
logger.info("--- Unit Test for Columns Similarity ---")


@pytest.mark.parametrize(
    "col_id_1, col_id_2, predicted_score, predicted_label, expected_score, expected_label",
    [
        (a, b, c, d, e, f)
        for a, b, c, d, e, f in columns_similarity_mapping_unit_testing
    ],
)
def test_columns_similarity_public_datasets(
        col_id_1,
        col_id_2,
        predicted_score,
        predicted_label,
        expected_score,
        expected_label,
):
    assert (
            predicted_label == expected_label
    ), f"Test1 failed for column pair: '{(col_id_1, col_id_2)}'"
    if predicted_score is not None and expected_score is not None:
        assert (
                predicted_score >= np.floor(expected_score * 10) / 10
        ), f"Test2 failed for column pair: '{(col_id_1, col_id_2)}'"


# Unit Test for Table Similarity #
@pytest.mark.parametrize(
    "table_id_1, table_id_2, predicted_label, expected_label",
    [(a, b, c, d) for a, b, c, d in pruning_tables_similarity_mapping_unit_testing],
)
def test_pruning_tables_similarity_public_datasets(
        table_id_1,
        table_id_2,
        predicted_label,
        expected_label,
):
    assert (
            predicted_label == expected_label
    ), f"Pruning mode test failed for table pair: '{(table_id_1, table_id_2)}'"


post_pruning_tables_similarity_mapping_unit_testing = (
    get_predicted_expected_similarity_scores_mapping_for_tables(
        post_pruning_mode_output_PREDICTED,
        post_pruning_table_similarity_labels_expected,
    )
)


@pytest.mark.parametrize(
    "table_id_1, table_id_2, predicted_label, expected_label",
    [
        (a, b, c, d)
        for a, b, c, d in post_pruning_tables_similarity_mapping_unit_testing
    ],
)
def test_post_pruning_tables_similarity_public_datasets(
        table_id_1,
        table_id_2,
        predicted_label,
        expected_label,
):
    assert (
            predicted_label == expected_label
    ), f"Non Pruning Mode test failed for table pair: '{(table_id_1, table_id_2)}'"
