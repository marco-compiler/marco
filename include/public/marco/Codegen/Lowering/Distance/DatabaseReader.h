#ifndef MARCO_CODEGEN_LOWERING_DISTANCE_DATABASEREADER_H
#define MARCO_CODEGEN_LOWERING_DISTANCE_DATABASEREADER_H


#include <fstream>
#include <string>
#include <vector>

namespace marco::codegen::lowering
{
    // We use the synset's hypernym.csv byte offset as the synset ID.
    // Since synsets could also be defined in different ways, and one
    // future implementation might use a different ID system, we use
    // a typedef to make it easier to change the synset ID type.
    using Synset = int;

    class DatabaseReader
    {
    private:
        // File mappings. Note that the hypernyms file has an extra
        // column, which is the count of hypernyms for that synset.
        static std::ifstream senses;     // Words -> Synsets.
        static std::ifstream hypernyms;  // Synsets -> Hypernyms.
        static std::ifstream synsets;    // Synsets -> String names.
        
        // We use this flag to check if the files have been opened.
        // This is to prevent opening the files multiple times, on
        // different instances of the class.
        static bool filesOpened;
        static int openInstances;

        // The total number of synset appearances in the corpus.
        static int totalSynsetCount;

        // This method finds a row in a CSV file, given an int byte
        // offset and returns the columns as a vector of strings.
        // The first column is the offset itself, and is omitted.
        static std::vector<std::string>
        offsetToRow(std::ifstream& file, int offset);

        // This method finds a row in a CSV file, given a string key
        // and returns the columns as a vector of strings. The first
        // column is the key itself, and is omitted.
        static std::vector<std::string>
        keyToRow(std::ifstream& file, const std::string& key);

    public:
        DatabaseReader();
        ~DatabaseReader();

        static std::vector<Synset> getSynsets(const std::string& word);
        static std::vector<Synset> getHypernyms(const Synset& synset);

        // Get a synset's count within the corpus.
        static int getSynsetCount(const Synset& synset);

        // Get the total count of synsets in the corpus.
        static int getTotalSynsetCount();

        // Get a readable name for a synset (e.g. "dog.n.01")
        static std::string getSynsetName(const Synset& synset);
    };
}

#endif // MARCO_CODEGEN_LOWERING_DISTANCE_DATABASEREADER_H