import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.keyphrase_extractor import extract_keyphrases

def main():
    # Đọc file văn bản mẫu từ thư mục data
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    keyphrases = extract_keyphrases(text, top_k=10)
    print("Extracted Keyphrases:")
    for i, kp in enumerate(keyphrases, start=1):
        print(f"{i}. {kp}")

if __name__ == "__main__":
    main()
