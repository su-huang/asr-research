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
    # Load file with "fixed" tokens
    bad_words_fixed = {}
    # with open('/secure/fs00/afield6/police/shuan148/bpc-cpdForAF.csv','r', encoding='latin1') as f:
    #     next(f)
    #     for line in f:
    #         if len(line.split(','))>1:
    #             orig, fixed = line.split(',')
    #             if fixed.replace('\n','') != '':
    #                 bad_words_fixed[orig] = fixed.replace('\n','')

    path = '/secure/fs00/afield6/police/shuan148/bpc-cpdForAF.csv'
    
    with open(path, 'r', encoding='latin1') as f:
        # DictReader uses header names
        reader = csv.DictReader(f)        
        for row in reader:
            orig = row.get('transcription_raw', '').strip().upper()
            fixed = row.get('transcription_final', '').strip()            
            if orig and fixed:
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
    return ' '.join(words_fixed).replace('  ',' ')

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