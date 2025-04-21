from fastapi import FastAPI


from mderank import MDERank

# This ensures the project root is in the Python path.
# Consider converting your project into a package to avoid this.

from pydantic import BaseModel

# Initialize FastAPI app with metadata.
app: FastAPI = FastAPI(
    title="MDERank Keyphrase Extraction API",
    description="API for extracting keyphrases from text using MDERank.",
    version="1.0.0",
)

# Request and response models
class DocumentRequest(BaseModel):
    text: str
    top_k: int = 20

class KeyphraseResponse(BaseModel):
    keyphrases: list[str]

# Initialize the MDERank model.
mderank = MDERank()
def extract_keyphrases(text: str, top_k: int) -> list[str]:
    """
    Extract keyphrases from the given text using MDERank.
    
    :param text: The input text from which to extract keyphrases.
    :param top_k: The number of top keyphrases to return.
    :return: A list of extracted keyphrases.
    """
    words: list[str] = text.split()
    truncated_text: str = " ".join(words[:400])
    # Extract keyphrases using MDERank.
    keyphrases: list[str] = mderank.extract_keyphrases(document=truncated_text, top_k=top_k)
    return keyphrases

# Endpoint to extract keyphrases.
@app.post(path="/mderank", response_model=KeyphraseResponse)
async def extract_keyphrases_endpoint(request: DocumentRequest) -> KeyphraseResponse:
    keyphrases: list[str] = extract_keyphrases(text=request.text, top_k=request.top_k)
    return KeyphraseResponse(keyphrases=keyphrases)

# Simple root endpoint.
@app.get(path="/")
def root() -> dict[str, str]:
    return {"message": "Welcome to the MDERank Keyphrase Extraction API!"}

# Corrected main guard
if __name__ == "__main__":
    import uvicorn
    # Use "main:app" when running from your project root.
    uvicorn.run(app="main:app", host="0.0.0.0", port=5040, reload=True)
