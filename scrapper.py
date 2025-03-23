import requests
from bs4 import BeautifulSoup
import os
import time
import os
import re
import pandas as pd


# SCRAPPING
# Base URL
base_url = "https://www.slovorod.ru/der-tikhonov/"

# Dictionary mapping Russian letters to their transliterated filenames
alphabet = {
    "а": "tih-a.htm", "б": "tih-b.htm", "в": "tih-v.htm", "г": "tih-g.htm", "д": "tih-d.htm",
    "е": "tih-je.htm", "ж": "tih-zh.htm", "з": "tih-z.htm", "и": "tih-i.htm",
    "й": "tih-j.htm", "к": "tih-k.htm", "л": "tih-l.htm", "м": "tih-m.htm", "н": "tih-n.htm",
    "о": "tih-o.htm", "п": "tih-p.htm", "р": "tih-r.htm", "с": "tih-s.htm", "т": "tih-t.htm",
    "у": "tih-u.htm", "ф": "tih-f.htm", "х": "tih-x.htm", "ц": "tih-c.htm", "ч": "tih-ch.htm",
    "ш": "tih-sh.htm", "щ": "tih-sc.htm", "э": "tih-e.htm", "ю": "tih-ju.htm", "я": "tih-ja.htm"
}

# Headers to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Create a directory to save the scraped data
output_dir = "scraped_tikhonov"
os.makedirs(output_dir, exist_ok=True)

# Function to scrape a single page
def scrape_page(letter, url):
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract all text from <p> tags (adjust based on page structure)
            paragraphs = soup.find_all('p')
            text_content = "\n".join([p.get_text() for p in paragraphs])

            # Save to a file named after the letter
            filename = os.path.join(output_dir, f"{letter}.txt")
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(text_content)
            print(f"Successfully scraped {letter} -> {filename}")
        else:
            print(f"Failed to retrieve {letter}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error scraping {letter}: {e}")

# Loop through all letters and scrape their pages
for letter, filename in alphabet.items():
    full_url = base_url + filename
    print(f"Scraping {letter} from {full_url}...")
    scrape_page(letter, full_url)
    time.sleep(1)  # Be polite: pause between requests to avoid overloading the server

print("Scraping complete!")

# COMBINNING FILES
output_dir = "scraped_tikhonov"
combined_filename = os.path.join(output_dir, "combined.txt")

with open(combined_filename, 'w', encoding='utf-8') as outfile:
  for filename in os.listdir(output_dir):
    if filename.endswith(".txt") and filename != "combined.txt":
      filepath = os.path.join(output_dir, filename)
      with open(filepath, 'r', encoding='utf-8') as infile:
        outfile.write(infile.read())
        outfile.write("\n")  # Add a newline between files

print(f"All files combined into: {combined_filename}")

# OPEN COMBINED FILE
words = []
with open("scraped_tikhonov/combined.txt", 'r', encoding='utf-8') as f:
  for line in f:
    if '|' in line: # if there is a division bar in line, it contains a lemma
      words.append(line.strip())
print(words[:10])


# CREATION OF DATAFRAME
# Create a list to store the data for the DataFrame
data = []
for word in words:
  word = re.sub(r'\([^)]*\)', '', word)
  word = re.sub(r'\'', '', word)
  word = re.sub(r'\[[^)]*\]', '', word)
  word = re.sub(r'[0-9]', '', word)
  word = re.sub(r',.*', '', word)
  parts = word.split('|')
  if len(parts) >= 2:
    lemma = parts[0]
    morphemes = '|'.join(parts[1:])  # Join the parts after the first one
    lemma = lemma.strip()
    morphemes = morphemes.strip()
    morphemes = morphemes.strip("/")
    morphemes = re.sub(r'\s.*', '', morphemes)
    data.append([lemma, morphemes])

# CSV
data = pd.DataFrame(data)
data.columns = ['lemma', 'morphemes']
data.to_csv('data.csv', index=False)  # Save to a CSV file named 'data.csv'