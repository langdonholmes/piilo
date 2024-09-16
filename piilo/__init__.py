from .main import analyze, anonymize, anonymize_batch, get_anonymize, anonymize_batch_cli
import spacy

# en_core_web_sm as a dependency requires external link
# which does not play nice with CLI
try:
    import en_core_web_sm
except ModuleNotFoundError:
    spacy.cli.download("en_core_web_sm")