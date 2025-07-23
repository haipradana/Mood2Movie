import google.generativeai as genai
from src.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

def enhance_prompt(user_input: str) -> tuple[str, list[str]]:
    prompt = f"""
You are a helpful assistant that enhances movie recommendation prompts.
Your task is to do TWO things:

1. Rephrase the user input into a richer, clearer English sentence for semantic search.
2. Extract mood/genre-related tags from the input.

IMPORTANT:
You MUST only choose tags from this whitelist:
["calm", "soothing", "relaxing", "adrenaline", "thrilling", "angry",
 "sad", "emotional", "heartwarming", "funny", "romantic", "family",
 "mystery", "suspense", "violent", "scary", "fantasy", "war", "sport",
 "epic", "historical", "space", "sci-fi", "crime", "documentary", "anime",
 "animated", "action", "drama", "romance", "adventure"]

DO NOT invent new tags.

Format your output as JSON:
{{
  "enhanced": "<better version of input>",
  "tags": ["<chosen tags from list above>"]
}}

User input: "{user_input}"
"""
    import json, re
    response = genai.GenerativeModel("models/gemini-2.5-flash").generate_content(prompt)

    try:
        json_block = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        result = json.loads(json_block)
        return result.get("enhanced", user_input), result.get("tags", [])
    except Exception as e:
        print("⚠️ Gemini parsing failed:", e)
        return user_input, []
