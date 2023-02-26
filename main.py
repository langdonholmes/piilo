'''API for PIILO'''

from analyzer import prepare_analyzer
from anonymizer import surrogate_anonymizer
from fastapi import FastAPI
import logging
from models.anonymize import AnonymizeRequest, AnonymizeResponse
from fastapi.middleware.cors import CORSMiddleware

# Define Student Name Detection Model
configuration = {
    'nlp_engine_name': 'spacy',
    'models': [
        {'lang_code': 'en', 'model_name': 'en_student_name_detector'}],
}

# set up logger for this module
logger = logging.getLogger('api')
logging.basicConfig(level=logging.INFO)

# Load Custom Presidio Analyzer and Anonymizer
logger.info("Loading Presidio Analyzer and Anonymizer")
analyzer = prepare_analyzer(configuration)
anonymizer = surrogate_anonymizer()
logger.info("Loaded Presidio Analyzer and Anonymizer")

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define FastAPI routes
@app.get("/")
def hello():
    return {"message": "Hello World"}

@app.post("/anonymize")
def anonymize(anon_req: AnonymizeRequest) -> AnonymizeResponse:
    '''Anonymize PII in text using a custom Presidio Analyzer and Anonymizer
    '''
    analyzer_result = analyzer.analyze(anon_req.raw_text,
                                       entities=anon_req.entities,
                                       language=anon_req.language,
                                       )
    
    anonymizer_result = anonymizer.anonymize(anon_req.raw_text,
                                             analyzer_result)
    
    anonymize_response = AnonymizeResponse(
        anonymized_text=anonymizer_result
        )
    
    return anonymize_response

if __name__ == "__main__":
    import uvicorn
    import os
    
    uvicorn.run(
        "main:app", host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )