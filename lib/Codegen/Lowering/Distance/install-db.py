import sys
import os

import nltk
from nltk.corpus import wordnet as wn


if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <installl-dir> <nltk-data-dir>")
    print("Warning. This script should be run by CMake.")
    sys.exit(1)

print(f"Called with arguments {sys.argv[1]} and {sys.argv[2]}")

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
