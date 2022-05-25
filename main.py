from fastapi import FastAPI

app = FastAPI()

# GET endpoint
@app.get("/")
async def say_greeting():
    return {"greeting": "Welcome to the deploy-ml-pipeline project!"}