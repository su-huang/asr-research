# Whisper output normalization and updated dtl
import os
import csv 

def replace_question_marks(filename, bad_word_fixes):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.readlines()
       
        # Replace question marks and fix bad words on each line
        updated_lines = [fix_bad_words(line.replace("⁇", "<UNINTELLIGIBLE>"), bad_word_fixes)+"\n" for line in content]
       
        with open(filename + "_update", 'w', encoding='utf-8') as file:
            file.writelines(updated_lines)
       
        print(f"File '{filename}' has been updated successfully.")
   
    except FileNotFoundError:
        print(f"The file '{filename}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_bad_word_fixes():
    bad_words_fixed = {}
    path = '/secure/fs00/afield6/police/shuan148/bad_words_fixed.csv'
    
    with open(path, 'r', encoding='latin1') as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split on first comma only, in case fixed value contains commas
            parts = line.split(',', 1)
            
            if len(parts) == 2:
                orig = parts[0].strip()
                fixed = parts[1].strip()
                
                if orig and fixed:  # only add if both orig and fixed are non-empty
                    bad_words_fixed[orig] = fixed

    # Add known confusions (just to be sure)
    bad_words_fixed['FOURTY'] = 'FORTY'
    bad_words_fixed['OK'] = 'OKAY'
    bad_words_fixed['O'] = 'OH'
   
    # Two and three letter abbreviations were common, let's make absolute certain to fix a few really common ones
    bad_words_fixed['DOB'] = 'D O B'
    bad_words_fixed['EMS'] = 'E M S'
   
    # While we're at it, there are some common contractions without apostrophes
    bad_words_fixed['DONT'] = "DON'T"
   
    # Also, phoenetically identical but different spellings so just pick one
    bad_words_fixed['EDDY'] = 'EDDIE'
   
    # Also, 'ALRIGHT' -> 'ALL RIGHT'
    #bad_words_fixed['ALRIGHT'] = 'ALL RIGHT'
   
    # Also, 'GONNA' often confused with 'TO' but...
    # Investigation shows this is actually 'GONNA' <==> 'GOING TO' (so no change)

    return bad_words_fixed

def fix_bad_words(text, bad_word_fixes):
    # Fix "bad" words using manually generated dictionary
    text = text.strip()
    words = text.split()
    words_fixed = []
    for w in words:
        if str(w.upper()) in bad_word_fixes:
            fixed_word = bad_word_fixes[str(w.upper())]
            if fixed_word!=' ':
                words_fixed.append(fixed_word)
        else:
            words_fixed.append(w)
    return ' '.join(words_fixed).replace('  ', ' ').strip()

# def process_directory(root_directory):
#     for dirpath, dirnames, filenames in os.walk(root_directory):
#         if "all" in dirnames:
#             all_dir = os.path.join(dirpath, "all")
#             dev_file = os.path.join(all_dir, "dev.trn")
#             test_file = os.path.join(all_dir, "test.trn")
           
#             if os.path.exists(dev_file):
#                 replace_question_marks(dev_file)
#             if os.path.exists(test_file):
#                 replace_question_marks(test_file)


# bad_word_fixes = get_bad_word_fixes()
# process_directory("/project/graziul/models/error_logs_new/") # This crawls the directory for .trn files, may need to adjust