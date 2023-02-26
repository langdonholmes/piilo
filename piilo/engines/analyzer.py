import logging
from typing import Optional, List, Tuple, Set

from presidio_analyzer import (AnalysisExplanation, AnalyzerEngine,
                               LocalRecognizer, RecognizerRegistry,
                               RecognizerResult)
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngineProvider

logger = logging.getLogger('presidio-analyzer')

class CustomSpacyRecognizer(LocalRecognizer):
    ENTITIES = [
        'STUDENT',
    ]

    DEFAULT_EXPLANATION = 'Identified as {} by a Student Name Detection Model'

    CHECK_LABEL_GROUPS = [
        ({'STUDENT'}, {'STUDENT'}),
    ]

    MODEL_LANGUAGES = {
        'en': 'langdonholmes/en_student_name_detector',
    }

    def __init__(
        self,
        supported_language: str = 'en',
        supported_entities: Optional[List[str]] = None,
        check_label_groups: Optional[Tuple[Set, Set]] = None,
        ner_strength: float = 0.85,
    ):
        self.ner_strength = ner_strength
        
        self.check_label_groups = (
            check_label_groups if check_label_groups
            else self.CHECK_LABEL_GROUPS
        )
        supported_entities = (
            supported_entities if supported_entities
            else self.ENTITIES
            )
        
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
        )

    def load(self) -> None:
        '''Load the model, not used. Model is loaded during initialization.'''
        pass

    def get_supported_entities(self) -> List[str]:
        '''
        Return supported entities by this model.
        :return: List of the supported entities.
        '''
        return self.supported_entities

    def build_spacy_explanation(
        self, original_score: float, explanation: str
    ) -> AnalysisExplanation:
        '''
        Create explanation for why this result was detected.
        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :return:
        '''
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    def analyze(self,
                text: str,
                entities: List[str] = None,
                nlp_artifacts: NlpArtifacts = None):
        '''Analyze input using Analyzer engine and input arguments (kwargs).'''
        
        if not entities or 'All' in entities:
            entities = None
    
        results = []
        
        if not nlp_artifacts:
            logger.warning('Skipping SpaCy, nlp artifacts not provided...')
            return results

        ner_entities = nlp_artifacts.entities

        for entity in entities:
            if entity not in self.supported_entities:
                continue
            for ent in ner_entities:
                if not self.__check_label(entity, ent.label_,
                                          self.check_label_groups):
                    continue
                textual_explanation = self.DEFAULT_EXPLANATION.format(
                    ent.label_)
                explanation = self.build_spacy_explanation(
                    self.ner_strength, textual_explanation
                )
                spacy_result = RecognizerResult(
                    entity_type=entity,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=self.ner_strength,
                    analysis_explanation=explanation,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name
                    },
                )
                results.append(spacy_result)

        return results

    @staticmethod
    def __check_label(
        entity: str, label: str, check_label_groups: Tuple[Set, Set]
    ) -> bool:
        return any(
            [entity in egrp and label in lgrp 
             for egrp, lgrp in check_label_groups]
        )

class CustomAnalyzer(AnalyzerEngine):
    '''Custom Analyzer Engine for Presidio.'''
    
    def __init__(self, configuration):
        
        spacy_recognizer = CustomSpacyRecognizer()
        
        # Create NLP engine based on configuration
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # add rule-based recognizers
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)
        registry.add_recognizer(spacy_recognizer)

        # remove the nlp engine we passed, to use custom label mappings
        registry.remove_recognizer('SpacyRecognizer')

        super().__init__(
            nlp_engine=nlp_engine,
            registry=registry,
            supported_languages=['en']
            )