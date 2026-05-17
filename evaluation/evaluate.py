import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from metrics.fluency.xcomet import CometFluency
from metrics.similarity import SimilarityConfig, SimilarityMeasurement
from metrics.toxicity import ToxicityConfig, ToxicityMeasurement
from utils import RequiredColumns, read_dataframes

REFERENCE_COLUMN = "references"
EVALUATION_PATH = Path(__file__).resolve()
SUBMISSION_FOLDER = Path(EVALUATION_PATH.parent.parent, "sample_submissions/")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate text similarity between original and rewritten texts."
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path(SUBMISSION_FOLDER, "dev_duplicate.tsv"),
        help="Path to submission TSV file",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            SUBMISSION_FOLDER, "dev_duplicate.tsv"
        ),  # we do not provide real refences as for now
        help="Optional path to reference texts TSV file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for computations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing in similarity and toxicity models",
    )
    parser.add_argument(
        "--fluency_batch_size",
        type=int,
        default=512,
        help="Batch size for processing in fluency models",
    )
    parser.add_argument(
        "--efficient",
        type=bool,
        default=False,
        help="Use efficient similarity calculation",
    )

    args = parser.parse_args()

    # Read input files
    submission_df, reference_df = read_dataframes(args.submission, args.reference)

    # Merge dataframes to allign with submission with reference dataframe
    submission_df_merged = pd.merge(
        reference_df,
        submission_df,
        on=[RequiredColumns.TOXIC_SENTENCE, RequiredColumns.LANG],
        how="left",
    )

    submission_df_merged = submission_df_merged[
        [
            RequiredColumns.TOXIC_SENTENCE,
            RequiredColumns.NEUTRAL_SENTENCE + "_x",
            RequiredColumns.NEUTRAL_SENTENCE + "_y",
            RequiredColumns.LANG,
        ]
    ]
    submission_df_merged = submission_df_merged.rename(
        columns={
            RequiredColumns.NEUTRAL_SENTENCE + "_x": REFERENCE_COLUMN,
            RequiredColumns.NEUTRAL_SENTENCE + "_y": RequiredColumns.NEUTRAL_SENTENCE,
        }
    )

    original_texts = submission_df_merged[RequiredColumns.TOXIC_SENTENCE].tolist()
    rewritten_texts = submission_df_merged[RequiredColumns.NEUTRAL_SENTENCE].tolist()
    reference_texts = submission_df_merged[REFERENCE_COLUMN].tolist()

    # Configure and run similarity measurement
    sim_config = SimilarityConfig(
        batch_size=args.batch_size,
        efficient_version=args.efficient,
        device=args.device,
    )
    similarity_measurer = SimilarityMeasurement(sim_config)
    sim_scores = similarity_measurer.evaluate_similarity(
        original_texts=original_texts,
        rewritten_texts=rewritten_texts,
        reference_texts=reference_texts,
    )

    # Configure and run toxicity measurement
    tox_config = ToxicityConfig(
        batch_size=args.batch_size,
        device=args.device,
    )
    toxicity_measurer = ToxicityMeasurement(tox_config)
    tox_scores = toxicity_measurer.compare_toxicity(
        original_texts=original_texts,
        rewritten_texts=rewritten_texts,
        reference_texts=reference_texts,
    )

    # Configure and run fluency measurement
    fluency_measurer = CometFluency()

    comet_input: list[dict[str, str]] = []
    for original_sent, rewritten_sent, reference_sent in zip(
        original_texts, rewritten_texts, reference_texts
    ):
        comet_input.append(
            {"src": original_sent, "mt": rewritten_sent, "ref": reference_sent}
        )

    fluency_scores = fluency_measurer.get_scores(
        input_data=comet_input, batch_size=args.fluency_batch_size
    )

    # Get Final Metric
    J = np.array(sim_scores) * np.array(tox_scores) * np.array(fluency_scores)
    submission_df_merged["J"] = J
    submission_df_merged["STA"] = tox_scores
    submission_df_merged["SIM"] = sim_scores
    submission_df_merged["XCOMET"] = fluency_scores
    results = submission_df_merged.groupby("lang").agg(
        {"STA": "mean", "SIM": "mean", "XCOMET": "mean", "J": "mean"}
    )
    print(results.reset_index().to_markdown())
    print(results.reset_index().to_dict(orient="records"))
    submission_df_merged.to_csv("submission_df_merged.csv", index=False)

def select_best_candidate(
    original_texts: list[str],
    rewritten_texts: list[list[str]],
) -> list[str]:
    """
    Select the best candidate for each original text based on the J scores.
    """
    # duplicate the original texts to match the number of rewritten texts
    duplicate_original_texts = [
        original_texts[i] for i in range(len(original_texts)) for _ in range(len(rewritten_texts[i]))
    ]
    # flatten the rewritten texts
    flatten_rewritten_texts = [text for sublist in rewritten_texts for text in sublist]
    # calculate the J scores for each original and rewritten text pair using batch processing
    sim_config = SimilarityConfig(
        batch_size=32,
        efficient_version=False,
        device="cuda",
    )
    similarity_measurer = SimilarityMeasurement(sim_config)
    sim_scores = similarity_measurer.evaluate_similarity(
        original_texts=duplicate_original_texts,
        rewritten_texts=flatten_rewritten_texts,
    )
    # calculate the toxicity scores for each original and rewritten text pair using batch processing
    tox_config = ToxicityConfig(
        batch_size=32,
        device="cuda",
    )
    toxicity_measurer = ToxicityMeasurement(tox_config)
    tox_scores = toxicity_measurer.compare_toxicity(
        original_texts=duplicate_original_texts,
        rewritten_texts=flatten_rewritten_texts,
    )
    # calculate the fluency scores for each original and rewritten text pair using batch processing
    fluency_measurer = CometFluency()
    comet_input: list[dict[str, str]] = []
    for original_sent, rewritten_sent in zip(duplicate_original_texts, flatten_rewritten_texts):
        comet_input.append({"src": original_sent, "mt": rewritten_sent, "ref": rewritten_sent})
    fluency_scores = fluency_measurer.get_scores(
        input_data=comet_input, batch_size=128
    )
    # calculate the J scores for each original and rewritten text pair
    J = np.array(sim_scores) * np.array(tox_scores) * np.array(fluency_scores) 
    # select the best candidate for each original text based on the J scores
    best_candidates = []
    for i in range(len(original_texts)):
        # get the indices of the rewritten texts for the current original text
        start_index = i * len(rewritten_texts[i])
        end_index = start_index + len(rewritten_texts[i])
        # get the J scores for the current original text
        j_scores = J[start_index:end_index]
        # get the index of the best candidate
        best_candidate_index = np.argmax(j_scores)
        # get the best candidate
        best_candidate = rewritten_texts[i][best_candidate_index]
        best_candidates.append(best_candidate)
    return best_candidates
if __name__ == "__main__":
    main()
