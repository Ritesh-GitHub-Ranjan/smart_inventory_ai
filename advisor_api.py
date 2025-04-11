from fastapi import FastAPI, Request
from pydantic import BaseModel
from agents.interfaces.advisor_agent import AdvisorAgent

app = FastAPI()
agent = AdvisorAgent()

class Query(BaseModel):
    user_query: str

@app.post("/ask")
def ask(query: Query):
    answer = agent.ask(query.user_query)
    return {"response": answer}
