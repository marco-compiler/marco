import sys
import os

import nltk
from nltk.corpus import wordnet as wn

"""
This python script downloads the wordnet data through the nltk library,
and then creates a custom database for the wordnet data. The custom
database is used by the WordDistanceCalculator class to calculate the
semantic distance between two words. The custom database is a set of
four files: count.txt, hypernyms.csv, senses.csv, and synsets.csv.

### count.txt
The count.txt file contains the total number of synsets in the wordnet
database. This number is used to calculate the probability of a synset
occurring in the database.

### hypernyms.csv
The hypernyms.csv file contains the hypernym relationships between
synsets in the wordnet database. The first column is the hypernym
identifier, the second column is the frequency of the synset, and the
remaining columns are the actual hypernyms of the synset. Note that,
conveniently, the synset identifier is the byte offset of the synset
in the file, so that we can quickly jump to the correct position in
the file.

### senses.csv
The senses.csv file contains the synsets of each word in the wordnet
database. The first column is the word, and the remaining columns are
the byte offsets of the synsets of the word in the hypernyms.csv file.

### synsets.csv
The synsets.csv file contains the string representation of each synset
in the wordnet database. The first column is the byte offset of the
synset in the hypernyms.csv file, and the second column is the string
representation of the synset, which can be useful for debugging purposes.
"""


if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <installl-dir> <nltk-data-dir>")
    print("Warning. This script should be run by CMake.")
    sys.exit(1)

print(f"Called with arguments {sys.argv[1]} and {sys.argv[2]}")

# Check that the install directory exists.
if os.listdir(sys.argv[1]):
    # Check that the install directory is empty.
    print(f"Skipping wordnet installation as {sys.argv[1]} is not empty.")
    sys.exit(0)

nltk.data.path.append(sys.argv[2])
nltk.download('wordnet', sys.argv[2])

# To have a nicely byte-indexed file, we set the size of the
# pointers to 8 characters. This way, we can easily calculate
# the offset of a synset in the file.
CHARS_PER_POINTER = 8

pointer_position = 0
hypernyms_offsets = {}
for synset in wn.all_synsets():
    hypernyms = synset.hypernyms()
    hypernyms_offsets[synset] = pointer_position
    num_hypernyms = len(hypernyms)

    # Add the amount of chars used for this particular line
    # to the pointer position. This way, we get to know the
    # current byte offset for each line.
    pointer_position += (CHARS_PER_POINTER + 1) * (num_hypernyms + 2)

hypernyms_lines = []
total_sysnset_count = 0
for synset in wn.all_synsets():
    hypernyms = synset.hypernyms()
    synset_offset = hypernyms_offsets[synset]
    line = [f"{synset_offset:0>{CHARS_PER_POINTER}}"]

    count = 0  # Count the synset frequency.
    for lemma in synset.lemmas():
        count += lemma.count()
    count = 1 if count == 0 else count
    total_sysnset_count += count

    line.append(f"{count:0>{CHARS_PER_POINTER}}")

    for hypernym in hypernyms:
        hypernym_offset = hypernyms_offsets[hypernym]
        string = f"{hypernym_offset:0>{CHARS_PER_POINTER}}"
        line.append(string)

    hypernyms_lines.append(",".join(line))

count_path = os.path.join(sys.argv[1], "count.txt")
with open(count_path, "w") as file:
    file.write(str(total_sysnset_count))

hypernyms_path = os.path.join(sys.argv[1], "hypernyms.csv")
with open(hypernyms_path, "w") as file:
    file.write("\n".join(hypernyms_lines))

senses_path = os.path.join(sys.argv[1], "senses.csv")
with open(senses_path, "w") as file:
    for word in wn.words():
        offsets = [
            f"{hypernyms_offsets[synset]:0>{CHARS_PER_POINTER}}"
            for synset in wn.synsets(word)
        ]
        file.write(f"{word},{','.join(offsets)}\n")

# For debugging purposes, a string representation of each
# synset can also be added to the custom database.
synsets_path = os.path.join(sys.argv[1], "synsets.csv")
with open(synsets_path, "w") as file:
    for synset in wn.all_synsets():
        string = f"{hypernyms_offsets[synset]:0>{CHARS_PER_POINTER}}"
        string += f",{synset.name()}\n"
        file.write(string)

print("Wordnet Database installation successful.")
print(f"Install directory: {sys.argv[1]}")
