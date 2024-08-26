import logging
import spacy
import yaml
import os
import pandas as pd
from typing import List, Optional, Set, Tuple

from presidio_analyzer import (
    AnalysisExplanation,
    AnalyzerEngine,
    LocalRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngineProvider

logger = logging.getLogger("presidio-analyzer")
CONFIG_FILE_PATH = os.path.join("configs", "kaggle_third.yaml")

# Leaving this in in case we need to use it later
# class CustomSpacyRecognizer(LocalRecognizer):
#     ENTITIES = [
#         "STUDENT",
#     ]

#     DEFAULT_EXPLANATION = "Identified as {} by a Student Name Detection Model"

#     CHECK_LABEL_GROUPS = [
#         ({"STUDENT"}, {"PERSON"}),
#     ]

#     MODEL_LANGUAGES = {
#         "en": "en_core_web_sm",
#     }

#     def __init__(
#         self,
#         supported_language: str = "en",
#         supported_entities: Optional[List[str]] = None,
#         check_label_groups: Optional[Tuple[Set, Set]] = None,
#         ner_strength: float = 0.85,
#     ):
#         self.ner_strength = ner_strength

#         self.check_label_groups = (
#             check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
#         )
#         supported_entities = supported_entities if supported_entities else self.ENTITIES

#         super().__init__(
#             supported_entities=supported_entities,
#             supported_language=supported_language,
#         )

#     def load(self) -> None:
#         """Load the model, not used. Model is loaded during initialization."""
#         pass

#     def get_supported_entities(self) -> List[str]:
#         """
#         Return supported entities by this model.
#         :return: List of the supported entities.
#         """
#         return self.supported_entities

#     def build_spacy_explanation(
#         self, original_score: float, explanation: str
#     ) -> AnalysisExplanation:
#         """
#         Create explanation for why this result was detected.
#         :param original_score: Score given by this recognizer
#         :param explanation: Explanation string
#         :return:
#         """
#         explanation = AnalysisExplanation(
#             recognizer=self.__class__.__name__,
#             original_score=original_score,
#             textual_explanation=explanation,
#         )
#         return explanation

#     def analyze(
#         self, text: str, entities: List[str] = None, nlp_artifacts: NlpArtifacts = None
#     ):
#         """Analyze input using Analyzer engine and input arguments (kwargs)."""

#         if not entities or "All" in entities:
#             entities = None

#         results = []

#         if not nlp_artifacts:
#             logger.warning("Skipping SpaCy, nlp artifacts not provided...")
#             return results

#         ner_entities = nlp_artifacts.entities

#         for entity in entities:
#             if entity not in self.supported_entities:
#                 continue
#             for ent in ner_entities:
#                 if not self.__check_label(entity, ent.label_, self.check_label_groups):
#                     continue
#                 textual_explanation = self.DEFAULT_EXPLANATION.format(ent.label_)
#                 explanation = self.build_spacy_explanation(
#                     self.ner_strength, textual_explanation
#                 )
#                 spacy_result = RecognizerResult(
#                     entity_type=entity,
#                     start=ent.start_char,
#                     end=ent.end_char,
#                     score=self.ner_strength,
#                     analysis_explanation=explanation,
#                     recognition_metadata={
#                         RecognizerResult.RECOGNIZER_NAME_KEY: self.name
#                     },
#                 )
#                 results.append(spacy_result)

#         return results

#     @staticmethod
#     def __check_label(
#         entity: str, label: str, check_label_groups: Tuple[Set, Set]
#     ) -> bool:
#         return any(
#             [entity in egrp and label in lgrp for egrp, lgrp in check_label_groups]
#         )

class KaggleThirdAnalyzer(LocalRecognizer):
    """
    Custom recognizer adapting the third place winning solution from the Kaggle competition.

    Original code can be found here: 
    https://www.kaggle.com/code/devinanzelmo/piidd-efficiency-3rd-inference
    Discussion can be found here: 
    https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/506133

    Code is a non-transformer solution comprising two parts: a rule-based approach, 
    and a feature-based approach.
    """

    def __init__(
        self,
        config_file_path: str = None,
    ):
        
        # Looked at a few configuration options. 
        # Decided to go with yaml for now since it's a dependency of presidio.
        # Making a switch should be easy if needed.
        with open(config_file_path, 'r') as f:
            self.configuration = yaml.safe_load(f)

        # Black lists
        self.email_extensions_bl = self.configuration["black_lists"]['email_extensions']
        self.url_extensions_bl = self.configuration["black_lists"]['url_extensions']
        self.zip_codes_bl = self.setup_parquets(self.configuration, "zip_codes")
        self.names_bl = self.setup_parquets(self.configuration, "names")

        # White lists
        self.url_extensions_wl = self.configuration["white_lists"]['url_extensions']

        # Delimiters
        self.phone_delimiters = self.configuration["delimiters"]['phone_delimiters']
        
        super().__init__(
            supported_language=self.configuration["supports"]["languages"],
            supported_entities=self.configuration["supports"]["entities"],
        )

    def load(self) -> None:
        pass

    def setup_parquets(self, config, target) -> pd.Series:
        loaded_parquet = pd.read_parquet(os.path.join("data", config['parquets'][target]['path']))
        relevant_series = loaded_parquet[config['parquets'][target]['column']]
        return relevant_series

    def create_result(
        self, entity: str, token:spacy.tokens.token.Token, score: float=0.95, explanation: str="Placeholder"
    ) -> RecognizerResult:
        """
        Create recognizer result object. Assumes token is a spacy token.
        """
        thirdplace_results = RecognizerResult(
            entity_type=entity,
            start=token.idx,
            end=token.idx + len(token.text),
            score=score,
            analysis_explanation=explanation,
            recognition_metadata={
                RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
            },
        )
        return thirdplace_results

    def build_spacy_explanation(
        self, original_score: float, explanation: str
    ) -> AnalysisExplanation:
        """
        Create explanation for why this result was detected.
        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :return:
        """
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    def token_pass(self, tokens, pad_size=2, window_size=12):
        """
        Any processing done on the token level should be done here.
        Currently comprises two processes:
        1. Application of rule-based approach
        2. Creation of features for feature-based approach
        """

        # For holding token-level predictions
        results = []


        #== For handling all feature-production process ==#
        #== Needs to be done on the token-level ==#
        
        # Configurations
        pad = ["gfsda"] * pad_size 
        padded_text = pad + [w.text for w in tokens] + pad
        # For holding any feature-related data
        features = {
            "splitter_data": [],
            "meta_data": [],
        }

        # pad_size = window_size//2
        # pad = ["gfsda"] * pad_size

        #==================================================#

        for ix, token in enumerate(tokens):
            res = None
            text = token.text

            # ID rule
            if len(text) > 8 and (text.isdigit() or (not text[:2].isdigit() and text[-4:].isdigit())): 
                res = self.create_result("ID_NUM", token)

            # Email rule 
            # (removed len(text) > 2 because it's reduant with the extension check)
            elif "@" in text and any(ext in text for ext in self.email_extensions_bl):
                res = self.create_result("EMAIL_ADDRESS", token)

            # URL rule
            # (added or "www" in text because the original ignored anything without http)
            elif ("http" in text or "www" in text) and all(ext not in text for ext in self.url_extensions_bl):
                exclude = any(word in text.lower() for word in self.url_extensions_wl)
                if not exclude:
                    res = self.create_result("URL", token)

            # Address rule
            # This currently also catches phone numbers..
            elif self.zip_codes_bl.eq(text).any():
                res = self.create_result("STREET_ADDRESS", token)
            
            # Phone number rules
            for delimiter in self.phone_delimiters:
                phone_number_parts = text.split(delimiter)
                if (len(phone_number_parts) == 2 and
                    all(part.isdigit() and len(part) > 2 for part in phone_number_parts)):
                    res = self.create_result("PHONE_NUMBER", token)
                    break
            
            # Phone number rules (exception; doesn't follow the delimiter rule)
            phone_number_parts = text.split(".")
            if (len(phone_number_parts) == 3 and 
                len(phone_number_parts[0]) == 3 and 
                len(phone_number_parts[1]) == 3 and 
                len(phone_number_parts[2]) == 4):
                res = self.create_result("PHONE_NUMBER", token)

            # Name rules
            # if self.names_bl.eq(text).any():
            #     if (doc, k) in c1d:
            #         feats.append(tokpad[k:k+pad_size*2+1])
            #         res = self.create_result("B-NAME_STUDENT", token)
            #     elif (doc, k) in c2d:
            #         feats.append(tokpad[k:k+pad_size*2+1])
            #         res = self.create_result("I-NAME_STUDENT", token)

            if res:
                results.append(res)

            # This might have to be moved to the feature generation process
            # Multiple passes required for this approach
            if self.names_bl.eq(text).any():
                features["meta_data"].append([tokens, ix - pad_size])
                features["splitter_data"].append(padded_text[ix - pad_size: ix + pad_size + 1])

        return results, features

    def generate_features(self, tokens, window_size=12):
        """
        Any additional features that are needed for the feature-based approach should be generated here.
        """
        pass

    def feature_pass(self, results, features):
        """
        Any processing done on the feature level should be done here.
        """
        pass

    def analyze(
            self, text: str, entities: List[str] = None, nlp_artifacts: NlpArtifacts = None
        ):
        """Analyze input using Analyzer engine and input arguments (kwargs)."""

        results = []

        if not nlp_artifacts:
            logger.warning("Skipping SpaCy, nlp artifacts not provided...")
            return results

        # Generate features here
        # features = self.generate_features(nlp_artifacts.tokens)

        # Currently only supports token-level processing
        # Part of token-level processing requires some feature generation.
        results, features = self.token_pass(nlp_artifacts.tokens)

        # Next steps should involve feature generation and processing
        # results = self.feature_pass(results, features)

        return results


class CustomAnalyzer(AnalyzerEngine):
    """Custom Analyzer Engine for Presidio."""

    def __init__(self, configuration):
        # spacy_recognizer = CustomSpacyRecognizer()
        kaggle_third_recognizer = KaggleThirdAnalyzer(config_file_path=CONFIG_FILE_PATH)

        # Create NLP engine based on configuration
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # add rule-based recognizers
        registry = RecognizerRegistry()
        # registry.add_recognizer(spacy_recognizer)
        registry.add_recognizer(kaggle_third_recognizer) # only use custom recongizers

        super().__init__(
            nlp_engine=nlp_engine, registry=registry, supported_languages=["en"]
        )
