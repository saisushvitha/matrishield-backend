import os
import io
import httpx
import json
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from google import genai 
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERPER_KEY = os.getenv("SERPER_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

async def search_serper(query: str) -> List[Dict]:
    search_url = "https://google.serper.dev/search"
    headers = {'X-API-KEY': SERPER_KEY, 'Content-Type': 'application/json'}
    async with httpx.AsyncClient() as client_http:
        try:
            response = await client_http.post(search_url, headers=headers, json={"q": query, "num": 20}, timeout=15.0)
            return response.json().get("organic", [])
        except Exception:
            return []

@app.post("/verify-bio")
async def verify_bio(
    name: str = Form(...),
    company: str = Form(...),
    college: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None) 
):
    ai_report = {}
    visual_search_query = "" 
    
    if file:
        try:
            image_bytes = await file.read()
            
            prompt = f"""
            Act as a Private Investigator. Analyze this photo.
            1. Describe physical features for a search engine (e.g. 'rimless glasses', 'blue saree').
            2. Detect if the image is AI-generated (check for artifacts).
            3. Provide a 'Visual Fingerprint': A 3-word unique context (e.g. 'convocation ceremony', 'office desk').
            4. Return ONLY a JSON object.
            
            JSON Structure:
            {{
                "is_ai_generated": bool,
                "facial_features": "detailed description",
                "search_keywords": "3-5 descriptive words",
                "photo_type": "Natural/Studio/AI",
                "trust_score": 1-10
            }}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                ]
            )
            
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            ai_report = json.loads(clean_json)
            visual_search_query = f'"{name}" {ai_report.get("search_keywords", "")}'
            
        except Exception as e:
            ai_report = {"error": f"AI Parsing Error: {str(e)}"}

    queries = [
        f'"{name}"',
        f'"{name}" {company}',
        f'site:instagram.com "{name}"',
        f'site:facebook.com "{name}"',
        f'site:github.io "{name}"',
        f'site:linkedin.com in "{name}"'
    ]
    
    if college:
        queries.append(f'"{name}" {college}')
    if visual_search_query:
        queries.append(visual_search_query)

    raw_results = []
    for q in queries:
        res = await search_serper(q)
        raw_results.extend(res)

    seen_links = set()
    final_results = []
    
    social_domains = ["instagram.com", "facebook.com", "github.io", "linkedin.com", "twitter.com"]

    for item in raw_results:
        link = item.get("link", "")
        if link not in seen_links:
            text_to_check = (item.get("snippet", "") + item.get("title", "")).lower()
            
            name_match = name.lower() in text_to_check
            company_match = company.lower() in text_to_check
            college_match = college.lower() in text_to_check if college else False
            is_social = any(domain in link for domain in social_domains)

            if name_match:
                final_results.append({
                    "title": item.get("title"),
                    "link": link,
                    "snippet": item.get("snippet"),
                    "source": link.split('/')[2].replace('www.', ''),
                    "verified_match": company_match or college_match or is_social
                })
                seen_links.add(link)

    return {
        "status": "success",
        "search_results": final_results[:40],
        "photo_analysis": ai_report
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)