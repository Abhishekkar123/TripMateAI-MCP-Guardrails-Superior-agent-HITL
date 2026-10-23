# import os 
# import asyncio
# import certifi 
# from dotenv import load_dotenv
# from langchain_mcp_adapters.client import MultiServerMCPClient

# os.environ["SSL_CERT_FILE"] = certifi.where()
# os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# load_dotenv()

# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

# client = MultiServerMCPClient(
#     {
#        "tavily": {
#             "transport": "streamable_http",
#             "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
#         },
        
#         "aviationstack": {
#             "transport": "stdio",
#             "command": "uvx",
#             "args": [
#                 "aviationstack-mcp"
#             ],
#             "env": {
#                 "AVIATION_STACK_API_KEY":AVIATION_STACK_API_KEY
#             }
#         },
  
#     }
# )


# async def get_all_tools():
#     tools = await client.get_tools()
#     print("\nAvailable MCP Tools:\n")

#     for tool in tools:
#         print(tool.name)
import asyncio
import os
import sys
from pathlib import Path

import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

# Custom local flight tool (replaces the AviationStack MCP server)
from tools.flight_tool import search_flights


# ==========================================
# Environment configuration
# ==========================================

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Automatically find the current project folder.
# This replaces the hard-coded Windows paths.
PROJECT_DIR = Path(__file__).resolve().parent
WEATHER_SERVER_PATH = PROJECT_DIR / "custom_weather_mcp_server.py"


# Preserve the complete environment when starting local stdio MCP servers.
WEATHER_ENV = os.environ.copy()
WEATHER_ENV["OPENWEATHER_API_KEY"] = (
    OPENWEATHER_API_KEY or ""
)


# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)


# ==========================================
# MCP client configuration
# ==========================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY}"
            )
        },

        "weather": {
            "transport": "stdio",

            # Use the same Python environment that runs app.py.
            "command": sys.executable,

            # Automatically use custom_weather_mcp_server.py
            # from the current project directory.
            "args": [
                str(WEATHER_SERVER_PATH)
            ],

            "env": WEATHER_ENV
        }
    }
)


# ==========================================
# Diagnostic function
# ==========================================

async def get_all_tools():
    """
    Load each MCP server separately.

    A broken server will no longer prevent the other
    working servers from loading.
    """

    all_tools = []

    for server_name in (
        "tavily",
        "weather"
    ):
        try:
            tools = await client.get_tools(
                server_name=server_name
            )

            all_tools.extend(tools)

            print(
                f"\nAvailable tools from "
                f"{server_name} MCP:\n"
            )

            for tool in tools:
                print(tool.name)

        except Exception as error:
            print(
                f"\nCould not connect to "
                f"{server_name} MCP:\n{error}\n"
            )

    # Custom local flight tool (no MCP server involved)
    print("\nAvailable local tools:\n")
    print("flight_tool.search_flights")

    return all_tools


# ==========================================
# Tavily MCP tool
# ==========================================

search_tool = None


async def initialize_mcp():
    """
    Initialize only Tavily.

    Previously this function initialized all MCP servers,
    so an AviationStack or Weather failure also caused
    Tavily hotel search to fail.
    """

    global search_tool

    if search_tool is not None:
        return

    tools = await client.get_tools(
        server_name="tavily"
    )

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    search_tool = tools_by_name.get(
        "tavily_search"
    )

    if search_tool is None:
        available_tools = ", ".join(
            tools_by_name.keys()
        )

        raise RuntimeError(
            "Tavily MCP connected, but the "
            "'tavily_search' tool was not found. "
            f"Available tools: "
            f"{available_tools or 'none'}"
        )


async def tavily_mcp_search(query: str):
    await initialize_mcp()

    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )

    return result


# ==========================================
# Custom Flight Tool (local, no MCP)
# ==========================================

async def flight_tool_search(query: str, limit: int = 10):
    """
    Async wrapper around flight_tool.search_flights.

    search_flights() uses the blocking `requests` library, so it is
    run in a worker thread to avoid blocking the event loop.
    Use this anywhere you previously called aviation_mcp_call().
    """
    return await asyncio.to_thread(search_flights, query, limit)


# ==========================================
# Weather MCP tools
# ==========================================

weather_tool = None
forecast_tool = None


async def initialize_weather_tools():
    global weather_tool
    global forecast_tool

    if (
        weather_tool is not None
        and forecast_tool is not None
    ):
        return

    if not WEATHER_SERVER_PATH.exists():
        raise FileNotFoundError(
            "Weather MCP server file was not found: "
            f"{WEATHER_SERVER_PATH}"
        )

    # Load only Weather.
    # Tavily and AviationStack will not be started.
    tools = await client.get_tools(
        server_name="weather"
    )

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    weather_tool = tools_by_name.get(
        "get_current_weather"
    )

    forecast_tool = tools_by_name.get(
        "get_forecast"
    )

    missing_tools = []

    if weather_tool is None:
        missing_tools.append(
            "get_current_weather"
        )

    if forecast_tool is None:
        missing_tools.append(
            "get_forecast"
        )

    if missing_tools:
        available_tools = ", ".join(
            tools_by_name.keys()
        )

        raise RuntimeError(
            "Missing Weather MCP tools: "
            f"{', '.join(missing_tools)}. "
            f"Available tools: "
            f"{available_tools or 'none'}"
        )


async def weather_mcp_search(city: str):
    await initialize_weather_tools()

    result = await weather_tool.ainvoke(
        {
            "city": city
        }
    )

    return result


async def forecast_mcp_search(city: str):
    await initialize_weather_tools()

    result = await forecast_tool.ainvoke(
        {
            "city": city
        }
    )

    return result


# ==========================================
# Destination extractor
# ==========================================

def extract_destination(query: str):
    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """

    response = llm.invoke(prompt)

    return response.content.strip()