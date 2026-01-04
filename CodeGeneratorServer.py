import os
import sys
import re
import logging
from typing import List
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-server")

mcp = FastMCP("generic-project-mcp")

# ---------------- MODEL ----------------
model_client = OpenAIChatCompletionClient(
    model=os.environ.get("OPENAI_MODEL_NAME", "llama-3.1-8b-instant"),
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

# ---------------- TOOLS ----------------

@mcp.tool()
def create_folders(paths: List[str]) -> str:
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Folder created: {path}")
    return "Folders created successfully"


def strip_markdown_fences(text: str) -> str:
    # Remove ```lang ... ``` or ``` ... ```
    return re.sub(r"^```[\w+-]*\n|```$", "", text.strip(), flags=re.MULTILINE)

@mcp.tool()
async def generate_and_write_files(
    files: List[str],
    project_prompt: str,
) -> str:
    agent = AssistantAgent(
        name="CodeGenerator",
        model_client=model_client,
        system_message=(
            "You are a senior developer.\n"
            "Output ONLY raw source code.\n"
            "DO NOT use markdown, backticks, or code fences.\n"
            "Return plain text code only."
        ),
    )

    for file in files:
        task = f"""
Project description:
{project_prompt}

Generate code for this file:
{file}
"""
        result = await agent.run(task=task)
        raw = result.messages[-1].content.strip()
        code = strip_markdown_fences(raw)

        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"File written: {file}")

    return "Files generated successfully"


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    try:
        logger.info("Starting MCP Server...")
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(e)
        sys.exit(1)
