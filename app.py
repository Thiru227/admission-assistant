def call_llm(prompt: str, context: str) -> str:
    """Call OpenRouter API with enhanced error handling"""
    
    if not OPENROUTER_API_KEY:
        return "⚠️ Configuration needed. Please set OPENROUTER_API_KEY."
    
    # Check if query is admission-related
    is_admission = is_admission_related(prompt)
    
    if is_admission:
        system_prompt = f"""You are RTC Scholar, a friendly and enthusiastic admission counselor for Rathinam Technical Campus.

CONTEXT:
{context}

YOUR TASK:
1. Answer their admission question clearly using the context
2. Be excited and positive about RTC
3. After answering, NATURALLY suggest: "Would you like me to send you our detailed brochure with complete admission information? Just click the 'Get Brochure' button above! 📚"
4. Keep response under 120 words
5. Sound conversational and helpful

PERSONALITY: Warm, encouraging, student-friendly"""
    else:
        system_prompt = f"""You are RTC Scholar, a helpful AI assistant for Rathinam Technical Campus.

CONTEXT:
{context}

RULES:
1. Answer using ONLY the context provided
2. Be accurate and concise
3. Keep response under 100 words
4. End with: "Anything else you'd like to know? 😊"

PERSONALITY: Professional, friendly, helpful"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://render.com",
        "X-Title": "RTC-Scholar"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }
    
    try:
        print(f"🔄 Calling OpenRouter API...")
        print(f"   Model: {LLM_MODEL}")
        print(f"   API Key: {OPENROUTER_API_KEY[:10]}...{OPENROUTER_API_KEY[-5:]}")
        
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60  # Increased timeout
        )
        
        print(f"   Status Code: {response.status_code}")
        
        # Log response for debugging
        if response.status_code != 200:
            print(f"❌ API Response: {response.text[:500]}")
            
            # Handle specific error codes
            if response.status_code == 401:
                return "⚠️ Invalid API key. Please check your OPENROUTER_API_KEY configuration."
            elif response.status_code == 429:
                return "⚠️ Rate limit exceeded. Please try again in a moment."
            elif response.status_code == 402:
                return "⚠️ Insufficient credits. Please add credits to your OpenRouter account."
            else:
                return f"⚠️ API error (Code {response.status_code}). Please try again later."
        
        response.raise_for_status()
        data = response.json()
        
        # Validate response structure
        if 'choices' not in data or not data['choices']:
            print(f"❌ Unexpected response structure: {data}")
            return "Something went wrong with the response. Please try again! 😅"
        
        result = data['choices'][0]['message']['content']
        print(f"✅ LLM response received: {len(result)} chars")
        return result
    
    except requests.exceptions.Timeout:
        print(f"❌ Timeout error after 60 seconds")
        return "Sorry, response took too long. Please try again! ⏱️"
        
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        return "Cannot connect to AI service. Please check your internet connection! 🌐"
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        return "AI service returned an error. Please try again! 🔄"
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        return "Having trouble connecting. Please try again! 🔄"
        
    except (KeyError, IndexError) as e:
        print(f"❌ Parse Error: {e}")
        print(f"   Response data: {data if 'data' in locals() else 'No data'}")
        return "Something went wrong. Please try again! 😅"
        
    except Exception as e:
        print(f"❌ Unexpected Error: {type(e).__name__}: {e}")
        return "An unexpected error occurred. Please try again! 🔄"
