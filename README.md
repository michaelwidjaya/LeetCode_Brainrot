# 🎙️ LeetCode Brainrot

**LeetCode Brainrot** is an AI-powered, TedEd-style podcast designed to help you master LeetCode questions intuitively—no code, just engaging, clear explanations. Now you can sharpen your interview skills on the go!

[![Spotify](https://img.shields.io/badge/Listen%20on-Spotify-1DB954?style=for-the-badge&logo=spotify&logoColor=white)](https://open.spotify.com/show/1h9dQFtmJMttfHJj8xBWqa?si=2fc65e959fde4763)
![Episodes](https://img.shields.io/badge/Episodes-150-blue?style=for-the-badge)
![AI Powered](https://img.shields.io/badge/Powered%20by-OpenAI-purple?style=for-the-badge)

> **“No code. No fluff. Just intuitive, podcast-ready explanations of the NeetCode 150.”**

---

## 🔊 Preview the Podcast

[![Spotify Player Screenshot](https://i.postimg.cc/CK8rKdzB/image.png)](https://open.spotify.com/show/1h9dQFtmJMttfHJj8xBWqa?si=2fc65e959fde4763)

> Click to listen on Spotify

---

## 📖 About

LeetCode Brainrot is a fully automated podcast that narrates AI-generated explanations for every problem in the **NeetCode 150**.

- 🎙️ **Podcast-friendly narration** (TED-Ed style, no code)
- 🤖 **Powered by GPT-4o** for expert-level pedagogy
- 🧠 **Automated pipeline** from markdown to audio to Spotify

---

## 🧰 Features

- ✅ **Intuitive Explanations:** Concepts explained clearly without referencing code.
- ✅ **AI-generated Narration:** Engaging voice-overs powered by OpenAI.
- ✅ **Automated Pipeline:** Python-based pipeline for scraping, narrating, and generating audio.

---

## 🛠️ Repository Structure
- 📂 explanations # AI-generated intuitive explanations
- 📂 audio # AI-generated audio files for podcast (not included in this repo)
- 📂 coverart # AI-generated thumbnails for each episode
- 📜 processProblems.py # Python script to generate scripts and audio
- 📜 README.md # This document

- ---

## ⚙️ Tech Stack

- **Python** for automation
- **OpenAI GPT-4o & GPT-4o-mini-tts** for explanation generation and audio
- **Spotify** for podcast hosting

---

## 📥 Installation

```bash
git clone https://github.com/yourusername/leetcode-brainrot.git
cd leetcode-brainrot
pip install -r requirements.txt
```
Create a .env file:

```bash
OPENAI_API_KEY=your_openai_api_key
```
Run the scraper:
```bash
python scraper.py
```

### This will:

- Scrape local markdown solutions

- Generate TED-style narrated explanations

- Convert narration into WAV audio

- Save files to /explanations and /audio

---

## 🎧 Podcast Details
- **Title**: LeetCode Brainrot

- **Description**: LeetCode Brainrot is an AI-powered TedEd-esque podcast that walks you through every problem in the NeetCode 150. No code, just intuitive explanations to help you practice LeetCode on the go.

- **Listen Now**: Spotify Link

## 📜 License
This project is licensed under the MIT License. 

🌟 **Contributions and improvements are welcome!** Feel free to fork, raise issues, or submit pull requests.

📧 **Contact**: tipaek@syr.edu | tax2farm@gmail.com
