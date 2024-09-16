"""Main for PIILO"""

import logging
import os
import pandas as pd
import argparse
import pkg_resources
import yaml

# assumes piilo is in site-packages
from piilo.engines.analyzer import CustomAnalyzer
from piilo.engines.anonymizer import SurrogateAnonymizer
from piilo.models.anonymize import AnonymizeRequest, AnonymizeResponse
from piilo.configs import piilo_config

cfg = piilo_config

configuration = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_sm"},
    ],
}

logger = logging.getLogger("piilo")

logger.info("Loading Custom Presidio Analyzer and Anonymizer...")
analyzer = CustomAnalyzer(configuration)
anonymizer = SurrogateAnonymizer()
logger.info("Loading Successful!")


def analyze(raw_text: str, entities=None, language="en"):
    analyzer_result = analyzer.analyze(
        raw_text,
        entities=entities,
        language=language,
    )
    return analyzer_result


def anonymize(raw_text: str, entities=None, language="en") -> str:

    analyzer_result = analyze(raw_text, entities, language)

    return anonymizer.anonymize(raw_text, analyzer_result)


def get_anonymize(anon_req: AnonymizeRequest) -> AnonymizeResponse:

    anonymizer_result = anonymize(
        anon_req.raw_text,
        entities=anon_req.entities,
        language=anon_req.language,
    )

    anonymize_response = AnonymizeResponse(anonymized_text=anonymizer_result)

    return anonymize_response

def anonymize_batch(dir: str, entities=cfg['supports']['entities'], language="en", file_format="csv") -> None:
    """
    Anonymize all files in a directory and return as csv or text files.
    """

    res = []

    file_formats = cfg['supports']['file_formats']

    if file_format not in file_formats:
        raise ValueError(f"File format {file_format} not supported. Please choose from {file_formats}")

    try:
        target_files = [w for w in os.listdir(dir) if w.endswith('txt')]
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory {dir} not found. Please enter a valid directory.")

    if target_files == []:
        no_files_err_msg = f"Directory {dir} does not contain any text files."
        if dir == os.getcwd():
            raise ValueError(no_files_err_msg + " You did not enter any directory so we used your current directory as default.")
        raise ValueError(no_files_err_msg)

    for file in target_files:
        try:
            with open(os.path.join(dir, file), 'r', encoding="utf-8") as f:
                raw_text = f.read()
                analyzer_result = analyze(raw_text, entities, language)
                anonymizer_result = anonymizer.anonymize(raw_text, analyzer_result)
                res.append((file, anonymizer_result.text))

        except Exception as e:
            logger.error(f"File {file} could not be anonymized due to the following error: {e}")
            print(f"File {file} could not be anonymized due to the following error: {e}")
            continue

    if file_format == "txt":
        for file_name, anonymized_text in res:
            output_file = os.path.join(dir, f"{file_name}_anonymized.txt")
            with open(output_file, 'w', encoding="utf-8") as f:
                f.write(anonymized_text)
    elif file_format == "csv":
        output_file = os.path.join(dir, "anonymized_results.csv")
        df = pd.DataFrame(res, columns=["file_name", "anonymized_text"])
        df.to_csv(output_file, index=False, encoding="utf-8")

    return None


def anonymize_batch_cli():
    parser = argparse.ArgumentParser(description="Anonymize text files in a directory.")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing text files to anonymize; defaults to the current directory")
    parser.add_argument("--entities", type=str, nargs='*', default=None, help="Entities to anonymize")
    parser.add_argument("--language", type=str, default="en", help="Language of the text files; currently only supports 'en'")
    parser.add_argument("--file_format", type=str, choices=["csv", "txt"], default="csv", help="Output file format; currently supports 'csv' and 'txt'")

    args = parser.parse_args()

    if args.dir is None:
        print(f"""Anonymizing text files in the current directory: {os.getcwd()}. If this is not what you want, 
              specify your target directory using the --dir flag (for example, <obfuscate --dir /path/to/your/directory>)""")
        print("Type 'y' to continue or anything else to exit.")
        user_input = input()
        args.dir = os.getcwd()
        if user_input.lower() != 'y':
            print("Exiting...")
            return None

    anonymize_batch(args.dir, args.entities, args.language, args.file_format)


if __name__ == "__main__":
    pass
