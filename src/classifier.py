import cv2
import numpy as np
import base64
import os
import json
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return os.getenv("GROQ_API_KEY")

WASTE_CATEGORIES = {
    "Plastic": {
        "color": (0, 165, 255), "emoji": "🟦", "bin": "Blue Bin",
        "tip": "Rinse before recycling. Remove caps and labels if possible.",
        "examples": "Bottles, bags, containers, packaging"
    },
    "Paper": {
        "color": (255, 255, 0), "emoji": "📄", "bin": "Yellow Bin",
        "tip": "Keep dry and clean. Remove plastic windows from envelopes.",
        "examples": "Newspapers, cardboard, magazines, office paper"
    },
    "Metal": {
        "color": (192, 192, 192), "emoji": "🔩", "bin": "Grey Bin",
        "tip": "Rinse cans before recycling. Crush to save space.",
        "examples": "Cans, foil, tins, bottle caps"
    },
    "Glass": {
        "color": (0, 255, 255), "emoji": "🫙", "bin": "Green Bin",
        "tip": "Separate by color if possible. Remove lids and rinse.",
        "examples": "Bottles, jars, containers"
    },
    "Organic": {
        "color": (0, 200, 0), "emoji": "🌱", "bin": "Brown Bin",
        "tip": "Great for composting. Keep separate from other waste.",
        "examples": "Food scraps, garden waste, fruit peels"
    },
    "E-Waste": {
        "color": (0, 0, 255), "emoji": "⚡", "bin": "Special Collection",
        "tip": "Never throw in regular bin. Take to e-waste collection point.",
        "examples": "Phones, batteries, cables, electronics"
    }
}

def classify_waste(frame):
    """Use Groq vision AI to classify waste."""
    try:
        client = Groq(api_key=get_api_key())

        _, buffer = cv2.imencode('.jpg', frame)
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Look at the main object in this image and classify it as waste. Reply with ONLY a JSON object, no other text: {\"type\": \"Plastic\", \"confidence\": 90}. The type must be exactly one of: Plastic, Paper, Metal, Glass, Organic, E-Waste."
                        }
                    ]
                }
            ],
            max_tokens=50
        )

        text = response.choices[0].message.content.strip()
        text = text[text.find('{'):text.rfind('}')+1]
        parsed = json.loads(text)
        waste_type = parsed.get("type", "Organic")
        confidence = parsed.get("confidence", 75)

        if waste_type not in WASTE_CATEGORIES:
            waste_type = "Organic"

        return waste_type, confidence, WASTE_CATEGORIES[waste_type]

    except Exception as e:
        print(f"Groq error: {e}")
        return fallback_classify(frame)

def fallback_classify(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_sat = np.mean(hsv[:, :, 1])
    mean_val = np.mean(hsv[:, :, 2])
    mean_hue = np.mean(hsv[:, :, 0])

    if mean_val > 200 and mean_sat < 30:
        return "Paper", 65, WASTE_CATEGORIES["Paper"]
    elif 35 < mean_hue < 85 and mean_sat > 60:
        return "Organic", 65, WASTE_CATEGORIES["Organic"]
    else:
        return "Plastic", 60, WASTE_CATEGORIES["Plastic"]