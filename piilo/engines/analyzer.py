import logging
import spacy
import yaml
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import fbeta_score
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
            self.cnfg = yaml.safe_load(f)

        # =======================================
        # Black lists
        # Black lists in the config file follow this structure:
        # ----------------
        # black_lists:
        #   [name_of_relevant_entity]: List[str]
        self.email_extensions_bl = self.cnfg["black_lists"]['email_extensions']
        self.url_extensions_bl = self.cnfg["black_lists"]['url_extensions']
        self.zip_codes_bl = self.setup_parquets("zip_codes").iloc[:, 0]
        self.first_names_bl = self.setup_parquets("first_names")
        self.last_names_bl = self.setup_parquets("last_names")
        self.names_bl = pd.concat([self.first_names_bl['first_name'], self.last_names_bl['last_name']]).reset_index(drop=True)
        # =======================================

        # =======================================
        # White lists
        # Whilte lists follow the same structure as black lists.
        self.url_extensions_wl = self.cnfg["white_lists"]['url_extensions']
        # =======================================

        # =======================================
        # Delimiters (anything that can be used to split and identify entities; e.g., phone numbers)
        self.phone_delimiters = self.cnfg["delimiters"]['phone_delimiters']
        # =======================================
        
        # =======================================
        # For feature generation
        # =======================================

        # Separate dataframes for high and low frequency names
        # Thresholds are customizable (set in the configuration file under "thresholds")
        self.first_name_high = self.first_names_bl[
            self.first_names_bl['count'] >= self.cnfg["thresholds"]['first_name_high']
        ]
        self.first_name_low = self.first_names_bl[
            self.first_names_bl['count'] >= self.cnfg["thresholds"]['first_name_low']
        ]
        self.last_name_high = self.last_names_bl[
            self.last_names_bl['count'] >= self.cnfg["thresholds"]['last_name_high']
        ]
        self.last_name_low = self.last_names_bl[
            self.last_names_bl['count'] >= self.cnfg["thresholds"]['last_name_low']
        ]

        # Dictionaries for membership checks and value lookups
        # All dictionaries required to run the vectorizer and the xgboost models submitted by third place winner
        self.first_name_full_dict = self.first_names_bl.set_index('first_name')['count'].to_dict()
        self.first_name_high_dict = self.first_name_high.set_index('first_name')['count'].to_dict()
        self.first_name_low_dict = self.first_name_low.set_index('first_name')['count'].to_dict()
        self.last_name_full_dict = self.last_names_bl.set_index('last_name')['count'].to_dict()
        self.last_name_high_dict = self.last_name_high.set_index('last_name')['count'].to_dict()
        self.last_name_low_dict = self.last_name_low.set_index('last_name')['count'].to_dict()

        self.words_10k = self.setup_parquets("words_10k")
        self.words_1k = self.setup_parquets("words_1k")
        self.words_popular = set(self.setup_parquets("words_popular").iloc[:, 0])

        # Creates key value pairs where the key is the word and the value is len(df) - index
        # Not sure what this accomplishes but it's set up this way in the orignial code
        self.words_10k_dict = dict(zip(self.words_10k.iloc[:,0].to_list(), range(self.words_10k.shape[0], 0, -1)))
        self.words_1k_dict = dict(zip(self.words_1k.iloc[:,0].to_list(), range(self.words_1k.shape[0], 0, -1)))
        
        # Name lists where any overlapping values with 10k popular words have been removed
        # e.g., russia, laugh, attack, rome, etc. for first names
        self.first_name_diff = set(self.setup_parquets("first_name_diff").iloc[:, 0])
        self.last_name_diff = set(self.setup_parquets("last_name_diff").iloc[:, 0])

        super().__init__(
            supported_language=self.cnfg["supports"]["languages"],
            supported_entities=self.cnfg["supports"]["entities"],
        )

    def setup_parquets(
        self, target: str
    ) -> pd.DataFrame:
        """
        Uses the class configuration dictionary and a target column name to load a parquet file 
        and return a Pandas DataFrame.
        
        Parquet file in the configuration follow this structure:
        ----------------
        parquets:
          [name_of_relevant_entity]:
            path: str -> path to the parquet file
            column: str -> name of the relevant column in the parquet file
        """

        loaded_parquet = pd.read_parquet(os.path.join("data", self.cnfg['parquets'][target]['path']))

        return loaded_parquet

    def create_result(
        self, entity: str, token:spacy.tokens.token.Token, score: float=0.95, explanation: str="Placeholder"
    ) -> RecognizerResult:
        """
        Create recognizer result object. Assumes token is a spacy token.
        Score and explanation are placeholders for now.
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

    def generate_padded_name_tokens(
        self, tokens: spacy.tokens.token.Token, pad_size:int =2
    ):
        """
        Iterates over tokens to find names and returns each hit with a window of pad_size*2 + 1.
        """
        pad = ["gfsda"] * pad_size
        padded_text = pad + [w.text for w in tokens] + pad

        name_indices = []
        padded_data = []

        for ix, token in enumerate(tokens):
            target_string = token.text
            if self.names_bl.eq(target_string).any():
                name_indices.append(ix)
                padded_data.append(padded_text[ix:ix+pad_size*2+1])

        return name_indices, padded_data

    def token_pass(
        self, tokens, name_indices, window_size=12
    ):
        """
        Any processing done on the token level should be done here.
        Currently comprises two processes:
        1. Application of rule-based approach
        2. Creation of features for feature-based approach
        """

        # For holding token-level predictions
        results = []
        predictions = []

        #== For handling all feature-production process ==#
        #== Needs to be done on the token-level ==#
        features = []
        features_pos = []
        
        # Configurations
        pad_size = window_size//2
        pad = ["gfsda"] * pad_size
        padded_text = pad + [w.text for w in tokens] + pad

        # Eliminate need to POS tag each split
        # Filler token "gfsda" is treated as a noun by spacy
        pos_pad = ["NOUN"] * pad_size
        padded_pos = pos_pad + [w.pos_ for w in tokens] + pos_pad

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
            if self.names_bl.eq(text).any():
                if ix in name_indices['first_name_indices']:
                    features.append(padded_text[ix:ix+pad_size*2+1])
                    features_pos.append(padded_pos[ix:ix+pad_size*2+1])
                    res = self.create_result("B-NAME_STUDENT", token)
                elif ix in name_indices['last_name_indices']:
                    features.append(padded_text[ix:ix+pad_size*2+1])
                    features_pos.append(padded_pos[ix:ix+pad_size*2+1])
                    res = self.create_result("I-NAME_STUDENT", token)

            if res:
                results.append(res)

        return results, features, features_pos

    def generate_features(
        self, target_strings: List[str]
    ):
        """
        Any additional features that are needed for the feature-based approach should be generated here.
        Mostly uses string manipulation and dictionary lookups.
        Here are the list of checks:
        - Checks if the first character is uppercase
        - Checks the length of the string
        - Checks if the string is a newline character
        - Checks if the string is a common punctuation
        - Checks if the string is padding
        - Checks if the string is a digit
        - Checks if the string is in the 10k/1k most common words or popular words
        - Checks if the string is in the high/low frequency first/last name lists
        - Get value counts for the corresponding 10k/1k most common words 
          and high/low frequency first/last names
        """

        string_feature_list_full = []

        for target_string in target_strings:

            # This is currently not customizable because the xgboost model was trained 
            # on a fixed set of features.

            conditions = [
            target_string[0].isupper(),
            len(target_string),
            target_string == "\n\n",
            target_string == "-",
            target_string == ".",
            target_string == ",",
            target_string == "?",
            target_string == ":",
            target_string == ";",
            target_string == "gfsda",
            target_string.lower() == "by",
            target_string.lower() == "name",
            target_string.lower() == "author",
            target_string.isdigit(),
            target_string.lower() in self.words_10k_dict,
            target_string.lower() in self.words_1k_dict,
            target_string.lower() in self.words_popular,
            target_string in self.first_name_full_dict,
            target_string in self.first_name_low_dict,
            target_string in self.first_name_high_dict,
            target_string in self.last_name_full_dict,
            target_string in self.last_name_low_dict,
            target_string in self.last_name_high_dict,
            target_string in self.last_name_low_dict, # This was repeated for some reason; probably a mistake
            target_string.lower() in self.last_name_diff,
            target_string.lower() in self.first_name_diff
            ]

            string_feature_list = [int(cond) for cond in conditions]

            string_feature_list.extend([i.get(target_string.lower(), 0) for i in [self.words_10k_dict, 
                                                                                  self.words_1k_dict]])
            string_feature_list.extend([i.get(target_string, 0) for i in [self.first_name_full_dict, 
                                                                                self.first_name_low_dict, 
                                                                                self.first_name_high_dict, 
                                                                                self.last_name_full_dict, 
                                                                                self.last_name_low_dict, 
                                                                                self.last_name_high_dict]])

            string_feature_list_full += string_feature_list

        return string_feature_list_full

    def load_models(self):
        # TODO: Change this into being controlled by self.cnfg
        models_splitter = []
        for model_path in os.listdir("models"):
            if "xgb_splitter_final" in model_path:
                m = xgb.XGBClassifier()
                m.load_model(f"models/{model_path}")
                models_splitter.append(m)

        models_fp_remove = []
        for model_path in os.listdir("models"):
            if "xgb_final" in model_path:
                m = xgb.XGBClassifier()
                m.load_model(f"models/{model_path}")
                models_fp_remove.append(m)

        return models_splitter, models_fp_remove

    def load_vectorizers(self):
        # TODO: Change this into being controlled by self.cnfg
        with open(os.path.join("models", "vectorizer2_raw_final.pkl"), "rb") as m1:
            vectorizer_raw = pickle.load(m1)

        with open(os.path.join("models", "vectorizer2_postags_final.pkl"), "rb") as m2:
            vectorizer_pt = pickle.load(m2)

        return vectorizer_raw, vectorizer_pt

    def make_predictions(self, models, feat):
        """
        Make predictions using the models and features. Return average predictions.
        """
        preds = [m.predict_proba(feat) for m in models]
        return sum(preds) / len(models)

    def use_split_data(self, padded_data_predictions, named_indices, threshold: float = 0.9995):
        name_indices = {"first_name_indices": set(), "last_name_indices": set()}

        for i in range(padded_data_predictions.shape[0]):
            if padded_data_predictions[i,0] < threshold:
                if padded_data_predictions[i,1] > padded_data_predictions[i,2]:
                    name_indices["first_name_indices"].add(named_indices[i])
                else:
                    name_indices["last_name_indices"].add(named_indices[i])
        
        return name_indices

    def feature_pass(self, results, feature_predictions):
        """
        Any processing done on the feature level should be done here.
        """

        adjusted_predictions = []
        counter = 0

        for res in results:
            label = res.entity_type
            if label == "B-NAME_STUDENT":
                if feature_predictions[counter] > 0:
                    res.entity_type = "NAME_STUDENT"
                    adjusted_predictions.append(res)
            elif label == "I-NAME_STUDENT":
                if feature_predictions[counter] > 0:
                    res.entity_type = "NAME_STUDENT"
                    adjusted_predictions.append(res)
                counter += 1
            else:
                adjusted_predictions.append(res)
        
        return adjusted_predictions

    def analyze(
            self, text: str, entities: List[str] = None, nlp_artifacts: NlpArtifacts = None
        ):
        """Analyze input using Analyzer engine and input arguments (kwargs)."""

        results = []

        if not nlp_artifacts:
            logger.warning("Skipping SpaCy, nlp artifacts not provided...")
            return results

        # Load models
        models_splitter, models_fp_remove = self.load_models()
        vectorizer_raw, vectorizer_pt = self.load_vectorizers()

        # Some feature generation needs to be done first using because 
        # part of token-level processing requires some byproducts of feature generation.

        # ==================================================================
        # First pass: find names and return them with a window of pad_size*2 + 1
        # Pad size is set to 2 by default
        named_indices, padded_data = self.generate_padded_name_tokens(nlp_artifacts.tokens)

        # Generate features for the name tokens
        # Mostly uses string manipulation and dictionary lookups
        name_features = np.array([self.generate_features(x) for x in padded_data])
        
        padded_data_predictions = self.make_predictions(models_splitter, name_features)
        name_indices = self.use_split_data(padded_data_predictions, named_indices)

        # ==================================================================
        # Second pass: token-level processing.
        results, features, pos_features = self.token_pass(nlp_artifacts.tokens, name_indices)

        feature_array = np.array([self.generate_features(x) for x in features])
        tfidf_raw = np.array(vectorizer_raw.transform(features).todense())
        tfidf_pt = np.array(vectorizer_pt.transform(pos_features).todense())

        # ==================================================================
        # Final step: feature-level processing

        concatenated_features = np.concatenate([feature_array, tfidf_raw, tfidf_pt], axis=1)

        feature_predictions = self.make_predictions(models_fp_remove, concatenated_features)
        feature_threshold = self.cnfg["thresholds"]['feature']
        feature_predictions = (feature_predictions[:,1] > feature_threshold).astype(np.int32)

        results = self.feature_pass(results, feature_predictions)

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
