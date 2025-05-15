### --- imports ---
import os
import re
import base64

import time

from dotenv import load_dotenv
load_dotenv()

import openai
from openai import OpenAI, RateLimitError, OpenAIError

## --- Scraping class ---
"""
Gets local Leetcode solutions stored in .md files.
process_problems() prompts 4o for test, and saves files in /explanations.
generate_audio() prompts 4o-mini-tts for audio and saves files in /audio. 
"""
class Scraper:
    def __init__(self):
        self.local_dir = "NeetCode-150"

        # ^ - start of string; \d{2} - 2 decimals; \. - one ./-; \s* - string
        self.directory_pattern = re.compile(r"^\d{2}[\.-]\s*")

        self.category_dict = {} # {category: list of problems}
        self.problem_dict = {} # {problem: problem text}

        self.out_base = "explanations" # directory for outputs

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        
    def discover_problem_files(self):
        """
        Walk through files to instantiate dictionaries with:
        (Category: Problem List)
        (Problem: Problem Markdown)
        """
        # --- traverse files ---
        for root, dirs, files in os.walk(self.local_dir):
            if self.directory_pattern.match(os.path.basename(root)): 
                category = os.path.basename(root)
                self.category_dict.setdefault(category, [])
                for fname in files:
                    if fname == "README.md": continue
                    key = f"{category}/{fname}"
                    self.category_dict[category].append(key)
                    if key not in self.problem_dict:
                        path = os.path.join(root, fname)
                        self.problem_dict[key] = self.load_markdown(path)
        

    def load_markdown(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as file:
            markdown_text = file.read()
            return markdown_text
    
    def print_dictionaries(self):
        ## Verify each
        for category, problem_list in self.category_dict.items():
            print(category)
            for problem in problem_list:
                print(self.problem_dict[problem])

    def process_problems(self, max_retries: int = 5, backoff: float = 2.0):
        # make output dir
        os.makedirs(self.out_base, exist_ok=True)

        for category, keys in self.category_dict.items():
            cat_dir = os.path.join(self.out_base, category)
            os.makedirs(cat_dir, exist_ok = True) # make category dir

            for key in keys:
                problem_name = os.path.splitext(os.path.basename(key))[0]
                out_path = os.path.join(cat_dir, f"{problem_name}.md")

                # Skip already completed
                if os.path.exists(out_path): continue

                markdown_text = self.problem_dict[key]

                prompt = [
                    {"role": "system", "content": (
                        "You are an expert computer science educator trained at MIT and UC Berkeley, with deep experience in algorithms, systems, and pedagogy. You specialize in helping recent CS graduates master technical interviews through clear, intuitive explanations of LeetCode problems."

                        "You do not provide code. You focus on explaining the problem-solving process conceptually, helping listeners understand how to move from brute-force ideas to an optimal solution. Your tone is confident and instructive, like a professor explaining concepts in a high-quality lecture." 

                        "Never reference the original prompt or say, 'the problem says'. You never use markdown, bullet points, or code formatting. Your explanations are in the style of a world-class technical communicator — imagine a TED-Ed lesson for a CS audience. Your explanations are meant to be spoken aloud — they should sound natural, with smooth transitions, no lists or headings, and no overly long or technical sentences. Keep things structured and digestible without sounding robotic."
                    )},
                    {"role": "user", "content": (
                        f"Problem: {problem_name}\n\n"
                        f"Here is a LeetCode problem with examples and correct solution: \n\n{markdown_text}\n\n" 
                        "Your job is to deliver a standalone explanation like a world-class CS educator — restate the problem in your own words which can be technical, introduce helpful conceptual examples to build intuition no code, and walk through the thought process behind solving it, including how to move from naive ideas to the optimal solution -- without using any code or markdown. Emphasize reasoning and intuition. Your explanation should sound like a clean, natural lecture, suitable for audio narration."
                        ""
                    )}
                ]

                # Call API
                for attempt in range(1, max_retries+1):
                    try:
                        resp = self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=prompt
                        )
                        explanation = resp.choices[0].message.content.strip()
                        print(f"PROBLEM: {problem_name}\n{explanation}\n\n\n")
                        # Write to file
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(f"# Explanation for {problem_name}\n\n")
                            f.write(explanation)
                        break  # success, move on
                    except RateLimitError:
                        if attempt == max_retries:
                            print(f"Rate limit, giving up on {problem_name}")
                        else:
                            time.sleep(backoff)
                    except OpenAIError as e:
                        print(f"OpenAIError: {e}. Ending process...")
                        break
                    except Exception as e:
                        if attempt == max_retries:
                            print(f"Error on {problem_name}: {e}")
                        else:
                            time.sleep(backoff)
    
    def generate_audio(
        self,
        explanations_dir: str = "explanations",
        audio_dir: str = "audio",
        model: str = "gpt-4o-mini-tts",
        voice: str = "fable",
        max_retries: int = 3,
        backoff: float = 2.0,
        voice_instructions: str = (
        "Read this explanation as if you are narrating a high-quality educational video, "
        "like TED-Ed. Your tone should be calm, intelligent, and confident. "
        "Speak clearly, with natural pacing and engaging intonation shifts to emphasize key concepts."
            ),
        ):
        """
        Walk explanations_dir/*.md, for each file:
        - Skip if corresponding .mp3 exists in audio_dir
        - Read the full explanation
        - Call OpenAI audio.speech.create with voice="fable"
        - Write out as MP3
        """

        for root, _, files in os.walk(explanations_dir):
            # Determine parallel output directory
            rel = os.path.relpath(root, explanations_dir)
            out_folder = os.path.join(audio_dir, rel)
            os.makedirs(out_folder, exist_ok=True)

            for fname in files:
                if not fname.lower().endswith(".md"):
                    continue
                base = os.path.splitext(fname)[0]
                src_path = os.path.join(root, fname)
                dst_path = os.path.join(out_folder, f"{base}.wav")

                # Skip already-generated audio
                if os.path.exists(dst_path):
                    continue

                # Read the explanation text
                text = open(src_path, "r", encoding="utf-8").read()
                

                # API call
                for attempt in range(1, max_retries+1):
                    try:
                        resp = openai.audio.speech.create(
                            model=model,
                            voice=voice,
                            input=text,
                            instructions = voice_instructions
                        )

                        with open(dst_path, "wb") as out:
                            out.write(resp.read())
                        print("Generated valid wav:", dst_path)
                        break
                    except RateLimitError:
                        if attempt == max_retries:
                            print(f"Rate limit, skipping {base}")
                        else:
                            time.sleep(backoff)
                    except OpenAIError as e:
                        print(f"API error on {base}: {e}")
                        break
                    except Exception as e:
                        if attempt == max_retries:
                            print(f"Failed on {base}: {e}")
                        else:
                            time.sleep(backoff)


def main():
    scraper = Scraper()
    scraper.discover_problem_files()
    scraper.print_dictionaries()
    #scraper.process_problems()
    scraper.generate_audio()
    print("----COMPLETE----")

if __name__ == "__main__":
    main()


        

        # def going fable
        
    




