import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from dotenv import load_dotenv
import os

load_dotenv()

model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash-lite",
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        family="unknown",
        structured_output=True
    ),
    api_key=os.getenv("API_KEY"),
)

math_server = StdioServerParams(
    command="python",
    args=["server.py"]
)

async def main() -> None:
    math_tools = await mcp_server_tools(math_server)
    agent = AssistantAgent(
        name="AGENT",
        model_client=model_client,
        tools=math_tools,
        system_message="You are a helpful assistant.",
        reflect_on_tool_use=True,
        model_client_stream=True, 
    )
    await Console(agent.run_stream(task="what is 2 + 3=?"))
    await model_client.close()

asyncio.run(main())
