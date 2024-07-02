#ifndef MARCO_CODEGEN_LOWERING_DISTANCE_SENTENCEDISTANCECALCULATOR_H
#define MARCO_CODEGEN_LOWERING_DISTANCE_SENTENCEDISTANCECALCULATOR_H

#include "marco/Codegen/Lowering/Distance/WordDistanceCalculator.h"

namespace marco::codegen::lowering
{
    using stringVector = std::vector<std::string>;

    class SentenceDistanceCalculator {
    private:
        WordDistanceCalculator wordDistanceCalculator;

        // Lower the case of the string.
        static void lowerCase(std::string& str);

        // Split the variable name into words.
        static stringVector camelCaseSplit(const std::string& str);
        static stringVector underscoreSplit(const std::string& str);
        static stringVector split(const std::string& str);

        // Get the joint word set of two sentences.
        static stringVector getJointWordSet(const stringVector& sentence1,
                                            const stringVector& sentence2);

        // Compute the value of the lexical vector corresponding to
        // an element of the joint word set.
        float getLexicalCell(const std::string& str,
                            const stringVector& sentence);

        // Compute the similarity of two word vectors.
        float getWordVecSimilarity(const stringVector& sentence1,
                                const stringVector& sentence2);

    public:
        SentenceDistanceCalculator();

        // Compute the similarity of two sentences.
        float getSimilarity(const std::string& sentence1,
                            const std::string& sentence2);
    };
}

#endif //MARCO_CODEGEN_LOWERING_DISTANCE_SENTENCEDISTANCECALCULATOR_H