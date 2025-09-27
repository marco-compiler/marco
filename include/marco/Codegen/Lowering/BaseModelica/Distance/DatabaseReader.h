#ifndef MARCO_CODEGEN_LOWERING_DISTANCE_DATABASEREADER_H
#define MARCO_CODEGEN_LOWERING_DISTANCE_DATABASEREADER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <fstream>
#include <vector>

/// @file DatabaseReader.h
/// @brief Header file for the DatabaseReader class.
///
/// To get information about relationships between words, we use the
/// WordNet database, which was used in the paper "Sentence similarity
/// based on semantic nets and corpus statistics" by Li, McLean, Bandar,
/// O'Shea, Crockett. Since the paper's data was already tuned to the
/// WordNet database, we decided to use the same database to get the
/// information we need, even though the original paper was more focused
/// on natural language sentences, and not on variable names.

namespace marco::codegen::lowering::bmodelica {
// We use the synset's hypernym.csv byte offset as the synset ID.
// Since synsets could also be defined in different ways, and one
// future implementation might use a different ID system, we use
// a typedef to make it easier to change the synset ID type.
using Synset = int;

/// @class marco::codegen::lowering::DatabaseReader
/// @brief A class that provides read access to the WordNet database.
///
/// This class encapsulates the reading of the WordNet database files,
/// which consist of three CSV files: senses.csv, hypernyms.csv, and
/// synsets.csv. These files map words to synsets, synsets to hypernyms,
/// and synsets to string names, respectively. A synset is a set of words
/// that are synonymous, and a hypernym is a word that is more general
/// than another word. For example, "language" is a hypernym of "modelica".
/// These mappings are used by @ref
/// marco::codegen::lowering::WordDistanceCalculator to calculate the distance
/// between two words. The wordnet files also contain a special 'count.txt'
/// file, which contains the total number of synset appearances in the corpus.
/// This is used to calculate the prior probability of a synset's appearance in
/// the corpus, in the
/// @ref marco::codegen::lowering::SentenceDistanceCalculator class.
///
/// All csv files contain rows of columns separated by commas. The first
/// column is always the key. The key is a byte offset in the hypernyms
/// and synsets files, and a word in the senses file, as the senses file
/// maps words to synsets. In the hypernyms file, the second column is
/// the count of hypernyms for that synset. All other columns are the
/// hypernyms themselves.
class DatabaseReader {
private:
  // We keep the constructor private to prevent multiple instances
  // of the DatabaseReader class.
  DatabaseReader();

  // File mappings. Note that the hypernyms file has an extra
  // column, which is the count of hypernyms for that synset.
  std::ifstream senses;    // Words -> Synsets.
  std::ifstream hypernyms; // Synsets -> Hypernyms.
  std::ifstream synsets;   // Synsets -> String names.

  // If anything goes wrong, we keep track of the failure
  // state through this boolean.
  bool filesOpened;

  // The total number of synset appearances in the corpus.
  int totalSynsetCount;

  /// @brief Finds a row in a CSV file, given an int byte offset.
  ///
  /// This method finds a row in a CSV file, given an int byte
  /// offset. The first column is the offset itself, and is omitted.
  /// Information is stored in the file as a series of rows, where
  /// each row is a series of columns separated by commas. Using byte
  /// offsets to find rows is faster than using string keys. More info
  /// on the strategy used to find offsets can be found in the python
  /// script used to generate the database files.
  std::vector<std::string> offsetToRow(std::ifstream &file, int offset);

  /// @brief Finds a row in a CSV file, given a string key.
  ///
  /// In order to find an offset, a binary search is performed
  /// on the file, using the actual word as the key. This method
  /// is used to find a row in a CSV file, given a string key
  /// instead of an int byte offset.
  std::vector<std::string> keyToRow(std::ifstream &file, llvm::StringRef key);

public:
  // No copying.
  DatabaseReader(const DatabaseReader &) = delete;

  // Instance getter.
  static DatabaseReader &getInstance();

  /// @brief Gets the synsets for a given word.
  ///
  /// Pure strings are not enough to represent a word in the
  /// WordNet database, as a word can have multiple meanings,
  /// and multiple words can have the same meaning. Thus, we
  /// use synsets to represent words. A synset is a set of words
  /// that are synonymous. Synsets are represented by an integer
  /// ID, which is also the byte offset of the synset in the
  /// hypernyms file.
  std::vector<Synset> getSynsets(llvm::StringRef word);

  /// @brief Gets the hypernyms for a given synset.
  ///
  /// A hypernym is a word that is more general than another word.
  /// For example, "language" is a hypernym of "modelica". This method
  /// returns the hypernyms of a given synset, which are also synsets.
  /// This is needed to construct a DAG of synsets, which is used to
  /// calculate the semantic distance between two words.
  std::vector<Synset> getHypernyms(const Synset &synset);

  /// @brief Gets the number of occurrences of a synset in the corpus.
  ///
  /// In order to calculate the prior probability of a synset's
  /// appearance in the corpus, we need to know how many times it
  /// has been registered. This method returns the number of
  /// occurrences of a synset in the corpus. Note that the information
  /// is stored in the second column of the hypernyms csv.
  int getSynsetCount(const Synset &synset);

  /// @brief Gets the number of occurrences of all synsets in the corpus.
  ///
  /// In order to calculate the prior probability of a synset's
  /// appearance in the corpus, we need to know how many times it
  /// has been registered. This method returns the total number of
  /// occurrences of all synsets in the corpus, which is used as a
  /// normalization factor in the @ref
  /// marco::codegen::lowering::SentenceDistanceCalculator class.
  int getTotalSynsetCount() const;

  /// @brief Gets a readable name for a synset.
  ///
  /// This method is never used in the current implementation, but
  /// it could be useful for debugging purposes. As synsets are
  /// represented by byte offsets in the hypernyms file, they are
  /// not human-readable. This method returns a human-readable name
  /// for a given synset, which is the name of the first word in
  /// the synset.
  std::string getSynsetName(const Synset &synset);
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_DISTANCE_DATABASEREADER_H