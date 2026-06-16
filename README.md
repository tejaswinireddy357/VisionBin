# ♻️ VisionBin — AI-Powered Waste Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![Groq](https://img.shields.io/badge/Groq-Vision_AI-orange)

> AI-powered waste classification app that uses your camera to identify waste and tell you which bin to use instantly.

## 🌐 Live Demo
👉 [Click here to try VisionBin](https://visionbin-tejaswinireddy357.streamlit.app)

## 🚀 Features
- 📸 Take a photo of any waste item using your camera
- 🤖 AI instantly classifies it into the correct waste category
- 🗑️ Tells you which bin to use and disposal tips
- 📊 Tracks your scan history and statistics
- 🎨 Beautiful dark dashboard UI

## 🗂️ Waste Categories
| Category | Bin | Examples |
|----------|-----|---------|
| 🟦 Plastic | Blue Bin | Bottles, bags, containers |
| 📄 Paper | Yellow Bin | Newspapers, cardboard |
| 🔩 Metal | Grey Bin | Cans, foil, tins |
| 🫙 Glass | Green Bin | Bottles, jars |
| 🌱 Organic | Brown Bin | Food scraps, garden waste |
| ⚡ E-Waste | Special Collection | Phones, batteries, cables |

## 🛠 Tech Stack
| Tool | Purpose |
|------|---------|
| Streamlit | Frontend dashboard + camera input |
| Groq (Llama 4 Scout) | Vision AI for waste classification |
| OpenCV | Image processing |
| Pillow | Image conversion |
| Python 3.10+ | Core language |

## ⚙️ Run Locally
```bash
git clone https://github.com/tejaswinireddy357/VisionBin.git
cd VisionBin
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Project Structure