#ifndef MARCO_CODEGEN_LOWERING_DISTANCE_DATABASEREADER_H
#define MARCO_CODEGEN_LOWERING_DISTANCE_DATABASEREADER_H

#include <fstream>
#include <vector>
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"

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
        // We keep the constructor private to prevent multiple instances
        // of the DatabaseReader class.
        DatabaseReader();

        // File mappings. Note that the hypernyms file has an extra
        // column, which is the count of hypernyms for that synset.
        std::ifstream senses;     // Words -> Synsets.
        std::ifstream hypernyms;  // Synsets -> Hypernyms.
        std::ifstream synsets;    // Synsets -> String names.

        // If anything goes wrong, we keep track of the failure
        // state through this boolean.
        bool filesOpened;

        // The total number of synset appearances in the corpus.
        int totalSynsetCount;

        // This method finds a row in a CSV file, given an int byte
        // offset and returns the columns as a vector of strings.
        // The first column is the offset itself, and is omitted.
        std::vector<std::string>
        offsetToRow(std::ifstream& file, int offset);

        // This method finds a row in a CSV file, given a string key
        // and returns the columns as a vector of strings. The first
        // column is the key itself, and is omitted.
        std::vector<std::string>
        keyToRow(std::ifstream& file, llvm::StringRef key);

    public:
        // No copying.
        DatabaseReader(const DatabaseReader&) = delete;

        // Instance getter.
        static DatabaseReader& getInstance();

        std::vector<Synset> getSynsets(llvm::StringRef word);
        std::vector<Synset> getHypernyms(const Synset& synset);

        // Get a synset's count within the corpus.
        int getSynsetCount(const Synset& synset);

        // Get the total count of synsets in the corpus.
        int getTotalSynsetCount() const;

        // Get a readable name for a synset (e.g. "dog.n.01")
        std::string getSynsetName(const Synset& synset);
    };
}

#endif // MARCO_CODEGEN_LOWERING_DISTANCE_DATABASEREADER_H