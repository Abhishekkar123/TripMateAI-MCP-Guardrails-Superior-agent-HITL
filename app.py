from pathlib import Path
import traceback
import uvicorn

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import run_travel_agent
from tools.flight_tool import search_flights, parse_route

# This is to allow nested event loops for async calls in FastAPI
import nest_asyncio
nest_asyncio.apply()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="TripMate AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0"
)


app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)


templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)



class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class FlightRequest(BaseModel):
    query: str
    limit: int = 10



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id
        )

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "weather_results": result.get("weather_results", ""),
                "itinerary": result["itinerary"],
                "llm_calls": result["llm_calls"],
            }
        )

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )



@app.post("/api/flights")
async def flights(request_data: FlightRequest):
    """Direct access to the custom flight tool (no MCP, no LLM)."""
    query = request_data.query.strip()

    if not query:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Query cannot be empty."}
        )

    try:
        dep_iata, arr_iata = parse_route(query)

        # search_flights is blocking (requests), so run it in a thread
        result = await run_in_threadpool(
            search_flights, query, request_data.limit
        )

        return JSONResponse(
            content={
                "success": True,
                "departure_iata": dep_iata,
                "arrival_iata": arr_iata,
                "flights": result,
            }
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running"
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})



if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )