#include <fstream>
#include <string>
#include <vector>

using Synset = std::pair<int, int>;

class DatabaseReader
{
private:
    // This file holds information about senses.
    // We use it to find the byte offset of a word.
    static std::ifstream indexSense;

    // These files hold information relating to
    // hypernyms, hyponyms, synonyms, etc.
    static std::ifstream dataAdjective;
    static std::ifstream dataAdverb;
    static std::ifstream dataNoun;
    static std::ifstream dataVerb;

    // We use this flag to check if the files have been opened.
    // This is to prevent opening the files multiple times, on
    // different instances of the class.
    static bool filesOpened;
    static int openInstances;

    // This pairing maps the file streams to the file names.
    static std::vector<std::pair<std::ifstream *, std::string>> files;

    // This pairing maps the synset type to the file stream.
    static std::vector<std::pair<int, std::ifstream *>> synsetFiles;

public:
    DatabaseReader();
    ~DatabaseReader();

    // Given a generic word, this function returns the synset
    // that the word belongs to. If the word is not found, it
    // returns (-1, -1).
    std::vector<Synset> getSynsets(const std::string &word);

    // Given a synset, this function returns a hypernym of the synset,
    // if it exists. Otherwise, it returns (-1, -1).
    Synset getHypernym(const Synset &synset);

    // Given a synset, this function returns the most
    // common word that represents the synset.
    std::string getWord(const Synset &synset);

    // Get the distance of a synset from the root.
    int getDepth(const Synset &synset);

    // Get the least common ancestor of two synsets.
    std::pair<Synset, int> getLCA(const Synset &synset1, const Synset &synset2);
};