import json
from normalization import get_bad_word_fixes

def save_dict_to_file(d, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"Dictionary saved to '{output_path}' ({len(d)} entries)")

# Load and save
bad_word_fixes = get_bad_word_fixes()
save_dict_to_file(bad_word_fixes, 'bad_word_fixes_verify.json')

# Verify by reloading
with open('bad_word_fixes_verify.json', 'r', encoding='utf-8') as f:
    reloaded = json.load(f)

print(f"\nDictionary intact: {bad_word_fixes == reloaded}")
