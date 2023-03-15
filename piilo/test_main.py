import logging

logger = logging.getLogger('piilo')
logging.basicConfig(level=logging.INFO)

def test_analyzer(texts):
    return [piilo.analyze(text) for text in texts]

    
def test_anonymizer(texts):
    return [piilo.anonymize(text) for text in texts]
    
if __name__ == "__main__":
    try:
        import piilo
    except ImportError:
        logger.info('Piilo not found in site-packages.')
        logger.info('Temporarily adding parent directory to path...')
        from pathlib import Path
        import sys
        # Add piilo to sys.path so that we can import from piilo
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import piilo

    texts = [
        'test string without identifiers',
        'My name is Antonio. Email: Antonio99@yahoo.com'
        ]

    # To analyze the texts. Returns list of RecognizerResult, defined by presidio_analyzer
    print(test_analyzer(texts))
    
    # To analyze AND anonymize with hiding-in-plain-sight obfuscation. Returns list of texts with identifiers obfuscated.
    print(test_anonymizer(texts))