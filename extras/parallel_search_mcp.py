"""
search_parallel_mcp.py
Interactive search using the Parallel Search MCP server.
"""
import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient

PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"

def extract_search_results(raw_result):
    """
    Extract search results from the MCP tool output.
    The output is expected to be a list of ToolMessage objects.
    Each ToolMessage has a content attribute that may be:
      - a dict with 'type' and 'text' keys (text contains a JSON string)
      - a list of such dicts
      - a plain string
    We parse the JSON and return a list of result dictionaries.
    """
    results = []
    if not isinstance(raw_result, list):
        return results

    for item in raw_result:
        # Get the content from the ToolMessage
        if hasattr(item, "content"):
            content = item.content
        else:
            content = item  # fallback

        # If content is a list, iterate over its elements
        if isinstance(content, list):
            for sub_item in content:
                if isinstance(sub_item, dict) and "text" in sub_item:
                    text = sub_item["text"]
                else:
                    text = str(sub_item)
                results.extend(parse_text_to_results(text))
        elif isinstance(content, dict) and "text" in content:
            # content is a dict with 'text'
            text = content["text"]
            results.extend(parse_text_to_results(text))
        elif isinstance(content, str):
            # content is a plain string
            results.extend(parse_text_to_results(content))
        else:
            # unknown format, just convert to string and try
            results.extend(parse_text_to_results(str(content)))
    return results

def parse_text_to_results(text):
    """Parse a JSON string and return a list of result dicts."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        elif isinstance(data, list):
            return data
        else:
            return [data]  # single object
    except json.JSONDecodeError:
        return [{"raw": text[:500]}]  # fallback

async def perform_search(query: str):
    """Connect to MCP, perform a web search, and display results."""
    connection = {
        "parallel": {
            "transport": "http",
            "url": PARALLEL_MCP_URL,
        }
    }

    print(f"\n🔍 Searching for: '{query}'\n")

    client = MultiServerMCPClient(connection)
    tools = await client.get_tools()

    search_tool = next((t for t in tools if t.name == "web_search"), None)
    if not search_tool:
        print("❌ web_search tool not found on the server.")
        return

    result = await search_tool.ainvoke({
        "search_queries": [query],
        "objective": "User query from interactive script",
        "mode": "advanced",
        "max_results": 5
    })

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    search_results = extract_search_results(result)

    if not search_results:
        print("No results found.")
    else:
        for idx, r in enumerate(search_results, 1):
            print(f"\n--- Result {idx} ---")
            if isinstance(r, dict):
                title = r.get("title", "No title")
                url = r.get("url", "No URL")
                publish_date = r.get("publish_date", "No date")
                excerpts = r.get("excerpts", [])
                print(f"Title: {title}")
                print(f"URL: {url}")
                print(f"Date: {publish_date}")
                if excerpts:
                    print("Excerpt:")
                    # Join excerpts and truncate for readability
                    excerpt_text = " ".join(excerpts)[:300]
                    print(f"  {excerpt_text}")
            else:
                print(str(r)[:500])
    print("\n" + "=" * 60)

async def main():
    print("Parallel Search MCP – Interactive Search")
    print("Press Ctrl+C to exit.\n")
    while True:
        try:
            query = input("Enter your search query (or 'quit' to exit): ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            await perform_search(query)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())