#include "marco/Codegen/Lowering/BaseModelica/Distance/SentenceDistanceCalculator.h"
#include <algorithm>
#include <cmath>

namespace marco::codegen::lowering::bmodelica {
SentenceDistanceCalculator::SentenceDistanceCalculator()
    : wordDistanceCalculator(WordDistanceCalculator()) {}

void SentenceDistanceCalculator::lowerCase(std::string &str) {
  // Convert the string to lowercase.
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
}

std::vector<std::string>
SentenceDistanceCalculator::camelCaseSplit(llvm::StringRef str) {
  bool lastCharLower = true;

  std::vector<std::string> words;
  std::string word;
  for (char c : str) {
    // Check if the character is uppercase.
    if (isupper(c)) {
      // Check if the last character was lowercase.
      if (lastCharLower && !word.empty()) {
        // Add the current word to the vector.
        words.push_back(word);
        // Clear the current word.
        word.clear();
      }
      // Add the uppercase character to the current word.
      word.push_back(c);
      lastCharLower = false;
    } else {
      // Add the lowercase character to the current word.
      word.push_back(c);
      lastCharLower = true;
    }
  }

  if (!word.empty()) {
    words.push_back(word);
  }

  // Remove all empty strings from the vector.
  words.erase(std::remove(words.begin(), words.end(), ""), words.end());

  return words;
}

std::vector<std::string>
SentenceDistanceCalculator::underscoreSplit(llvm::StringRef str) {
  std::vector<std::string> words;
  std::string word;
  for (char c : str) {
    // Check if the character is an underscore.
    if (c == '_' && !word.empty()) {
      words.push_back(word);
      word.clear();
    } else if (c != '_') {
      word.push_back(c);
    }
  }

  if (!word.empty()) {
    words.push_back(word);
  }

  return words;
}

std::vector<std::string>
SentenceDistanceCalculator::split(llvm::StringRef str) {
  // Check if the string contains an underscore.
  std::vector<std::string> words;
  if (str.find('_') != std::string::npos) {
    words = underscoreSplit(str);
  } else {
    words = camelCaseSplit(str);
  }

  // Apply the lower case transformation to all words.
  for (std::string &word : words) {
    lowerCase(word);
  }

  return words;
}

std::vector<std::string> SentenceDistanceCalculator::getJointWordSet(
    llvm::ArrayRef<std::string> sentence1,
    llvm::ArrayRef<std::string> sentence2) {
  // Create a set of strings.
  std::vector<std::string> jointWordSet;

  // Iterate over the first sentence.
  for (std::string word : sentence1) {
    // Check if the word is not in the joint word set.
    if (std::find(jointWordSet.begin(), jointWordSet.end(), word) ==
        jointWordSet.end()) {
      // Add the word to the joint word set.
      jointWordSet.push_back(word);
    }
  }

  // Iterate over the second sentence.
  for (std::string word : sentence2) {
    // Check if the word is not in the joint word set.
    if (std::find(jointWordSet.begin(), jointWordSet.end(), word) ==
        jointWordSet.end()) {
      // Add the word to the joint word set.
      jointWordSet.push_back(word);
    }
  }

  return jointWordSet;
}

float SentenceDistanceCalculator::getLexicalCell(
    llvm::StringRef str, llvm::ArrayRef<std::string> sentence) {
  // Check if the string is in the sentence.
  if (std::find(sentence.begin(), sentence.end(), str) != sentence.end()) {
    return 1.0f;
  }

  float bestSimilarity = 0.0f;
  for (llvm::StringRef word : sentence) {
    wordDistanceCalculator.analyzeByWords(str, word);
    float similarity = wordDistanceCalculator.getNormalizedSimilarity();
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
    }
  }

  return bestSimilarity;
}

float SentenceDistanceCalculator::getWordVecSimilarity(
    llvm::ArrayRef<std::string> sentence1,
    llvm::ArrayRef<std::string> sentence2) {
  // Generate the joint word set.
  std::vector<std::string> jointWordSet = getJointWordSet(sentence1, sentence2);

  // Get the lexical vector for the first sentence.
  std::vector<float> lexicalVector1;
  for (llvm::StringRef word : jointWordSet) {
    float cell = getLexicalCell(word, sentence1);
    lexicalVector1.push_back(cell);
  }

  // Get the lexical vector for the second sentence.
  std::vector<float> lexicalVector2;
  for (llvm::StringRef word : jointWordSet) {
    float cell = getLexicalCell(word, sentence2);
    lexicalVector2.push_back(cell);
  }

  // Compute the cosine coefficient.
  float dotProduct = 0.0f;
  float magnitude1 = 0.0f;
  float magnitude2 = 0.0f;

  for (size_t i = 0; i < jointWordSet.size(); i++) {
    dotProduct += lexicalVector1[i] * lexicalVector2[i];
    magnitude1 += lexicalVector1[i] * lexicalVector1[i];
    magnitude2 += lexicalVector2[i] * lexicalVector2[i];
  }
  magnitude1 = std::sqrt(magnitude1);
  magnitude2 = std::sqrt(magnitude2);

  return dotProduct / (magnitude1 * magnitude2);
}

float SentenceDistanceCalculator::getSimilarity(llvm::StringRef sentence1,
                                                llvm::StringRef sentence2) {
  // Split the sentences into words.
  std::vector<std::string> words1 = split(sentence1);
  std::vector<std::string> words2 = split(sentence2);

  // If one of the sentences is empty, return 0.
  if (words1.empty() || words2.empty()) {
    return 0.0f;
  }

  // Compute the similarity of the word vectors.
  return getWordVecSimilarity(words1, words2);
}
} // namespace marco::codegen::lowering::bmodelica