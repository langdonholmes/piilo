'''API for PIILO'''

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# assumes piilo is in site-packages
from piilo.engines.analyzer import CustomAnalyzer
from piilo.engines.anonymizer import SurrogateAnonymizer
from piilo.models.anonymize import AnonymizeRequest, AnonymizeResponse

configuration = {
    'nlp_engine_name': 'spacy',
    'models': [
        {'lang_code': 'en', 'model_name': 'en_student_name_detector'}],
}

logger = logging.getLogger('api')

logger.info("Loading Custom Presidio Analyzer and Anonymizer...")
analyzer = CustomAnalyzer(configuration)
anonymizer = SurrogateAnonymizer()
logger.info("Loading Successful!")

app = FastAPI()

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
    import os
    import uvicorn
    
    uvicorn.run(
        "main:app", host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )