from mcp.server.fastmcp import FastMCP
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
import asyncio
import re
import os
from dotenv import load_dotenv
load_dotenv()
from autogen_core.models import ModelFamily
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from typing import Literal
from pydantic import BaseModel

class AgentResponse(BaseModel):
    Folders: list[str]
    Files: list[str]


structure_model_client = OpenAIChatCompletionClient(
    model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-120b"),
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.groq.com/openai/v1"),
    api_key=os.environ["GROQ_API_KEY"],
    response_format=AgentResponse,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
)


model_client = OpenAIChatCompletionClient(
    model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-120b"),
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.groq.com/openai/v1"),
    api_key=os.environ["GROQ_API_KEY"],
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": False,
    },
)

mcp_server = StdioServerParams(
    command="python",
    args=["server3.py"],
)

async def main():
    tools = await mcp_server_tools(mcp_server)

    create_folders_tool = next(t for t in tools if t.name == "create_folders")
    generate_files_tool = next(t for t in tools if t.name == "generate_and_write_files")

    # -------- STRUCTURE AGENT --------
    structure_agent = AssistantAgent(
        name="StructureAgent",
        model_client=structure_model_client,
        system_message=(
            "You are a software architect.\n"
            "Return ONLY valid JSON in this format:\n"
            "make sure the paths are relative paths i.e. make sure the path of each folder starts with ./(name of the project root folder example: ./myproject/) or similar\n"
            "{\n"
            '  "folders": ["relative/path/"],\n'
            '  "files": ["relative/path/File.ext"]\n'
            "}\n"
            "No markdown. No explanation."
        ),
    )

    user_prompt = """
For a Java Spring Boot REST API project
with CRUD operations for Product entity Generate the folder structure in requested format only.
"""

    # -------- STEP 1: GET STRUCTURE --------
    result = await structure_agent.run(task=user_prompt)
    print("----- Parsed Response -----")
    print(result.messages[-1].content)
    print("---------agent_state-----------")
    agent_state = await structure_agent.save_state()
    print(agent_state)

    #-------- STEP 2: CREATE FOLDERS --------

    folder_agent = AssistantAgent(
        name="FolderAgent",
        model_client=model_client,
        tools=[create_folders_tool],
        system_message="Create all provided folders using the tool.",
    )

    await folder_agent.load_state(agent_state)

    await folder_agent.run(
        task=f"Create these folders from the list of folders genereted by perevious agent"
    )

    print("---Folders created successfully----")

    #-------- STEP 3: GENERATE AND WRITE FILES --------

    print("---Generating and writing files----")

    file_agent = AssistantAgent(
        name="FileAgent",
        model_client=model_client,
        tools=[generate_files_tool],
        system_message="Generate code and write all files using the tool.",
    )

    await file_agent.load_state(agent_state)
    await file_agent.run(
        task=f"Generate and write these files from the list of files genereted by perevious agent. Use the project description: {user_prompt}"
    )


    print("---Files generated and written successfully----")

asyncio.run(main())
