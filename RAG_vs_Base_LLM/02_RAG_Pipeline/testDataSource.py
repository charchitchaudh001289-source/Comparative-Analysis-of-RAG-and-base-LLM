import os

kb_dir = r"C:\Users\Nathan\Videos\AI Project\RAG_vs_Base_LLM\01_Knowledge_Base\cleaned"

targets = {
    "Q3 (Character splitter)": "CharacterTextSplitter",
    "Q3 (Recursive splitter)": "RecursiveCharacterTextSplitter",
    "Q4 (headers_to_split_on)": "headers_to_split_on"
}

for label, term in targets.items():
    print(f"\nSearching for: {label} -> '{term}'")
    found = False
    for fname in os.listdir(kb_dir):
        if fname.endswith(".md"):
            path = os.path.join(kb_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if term in content:
                    print(f"  [Found] in {fname} ({content.count(term)} occurrences)")
                    found = True
    if not found:
        print("  [Not Found in any document]")