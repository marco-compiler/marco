#include "marco/Codegen/Lowering/Distance/SentenceDistanceCalculator.h"
#include <algorithm>
#include <cmath>

namespace marco::codegen::lowering
{
    SentenceDistanceCalculator::SentenceDistanceCalculator() :
        wordDistanceCalculator(WordDistanceCalculator()) {}

    void SentenceDistanceCalculator::lowerCase(std::string& str) {
        // Convert the string to lowercase.
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    }

    stringVector SentenceDistanceCalculator::camelCaseSplit(const std::string& str) {
        bool lastCharLower = true;
        
        stringVector words;
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

    stringVector SentenceDistanceCalculator::underscoreSplit(const std::string& str) {
        stringVector words;
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

    stringVector SentenceDistanceCalculator::split(const std::string& str) {
        // Check if the string contains an underscore.
        stringVector words;
        if (str.find('_') != std::string::npos) {
            words =  underscoreSplit(str);
        } else {
            words =  camelCaseSplit(str);
        }

        // Apply the lower case transformation to all words.
        for (std::string& word : words) {
            lowerCase(word);
        }

        return words;
    }

    stringVector SentenceDistanceCalculator::getJointWordSet(const stringVector& sentence1,
                                                            const stringVector& sentence2) {
        // Create a set of strings.
        stringVector jointWordSet;

        // Iterate over the first sentence.
        for (const std::string& word : sentence1) {
            // Check if the word is not in the joint word set.
            if (std::find(jointWordSet.begin(), jointWordSet.end(), word) == jointWordSet.end()) {
                // Add the word to the joint word set.
                jointWordSet.push_back(word);
            }
        }

        // Iterate over the second sentence.
        for (const std::string& word : sentence2) {
            // Check if the word is not in the joint word set.
            if (std::find(jointWordSet.begin(), jointWordSet.end(), word) == jointWordSet.end()) {
                // Add the word to the joint word set.
                jointWordSet.push_back(word);
            }
        }

        return jointWordSet;
    }

    float SentenceDistanceCalculator::getLexicalCell(const std::string& str,
                                                    const stringVector& sentence) {
        // Check if the string is in the sentence.
        if (std::find(sentence.begin(), sentence.end(), str) != sentence.end()) {
            return 1.0f;
        }

        float bestSimilarity = 0.0f;
        for (const std::string& word : sentence) {
            wordDistanceCalculator.analyzeByWords(str, word);
            float similarity = wordDistanceCalculator.getNormalizedSimilarity();
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
            }
        }

        return bestSimilarity;
    }

    float SentenceDistanceCalculator::getWordVecSimilarity(const stringVector& sentence1,
                                                        const stringVector& sentence2) {
        // Generate the joint word set.
        stringVector jointWordSet = getJointWordSet(sentence1, sentence2);

        // Get the lexical vector for the first sentence.
        std::vector<float> lexicalVector1;
        for (const std::string& word : jointWordSet) {
            float cell = getLexicalCell(word, sentence1);
            lexicalVector1.push_back(cell);
        }

        // Get the lexical vector for the second sentence.
        std::vector<float> lexicalVector2;
        for (const std::string& word : jointWordSet) {
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

    float SentenceDistanceCalculator::getSimilarity(const std::string& sentence1,
                                                    const std::string& sentence2) {
        // Split the sentences into words.
        stringVector words1 = split(sentence1);
        stringVector words2 = split(sentence2);

        // If one of the sentences is empty, return 0.
        if (words1.empty() || words2.empty()) {
            return 0.0f;
        }

        // Compute the similarity of the word vectors.
        return getWordVecSimilarity(words1, words2);
    }
}