#ifndef MARCO_CODEGEN_LOWERING_DISTANCE_SENTENCEDISTANCECALCULATOR_H
#define MARCO_CODEGEN_LOWERING_DISTANCE_SENTENCEDISTANCECALCULATOR_H

#include "marco/Codegen/Lowering/Distance/WordDistanceCalculator.h"

namespace marco::codegen::lowering
{
    class SentenceDistanceCalculator {
    private:
        WordDistanceCalculator wordDistanceCalculator;

        // Lower the case of the string.
        static void lowerCase(std::string& str);

        // Split the variable name into words.
        static std::vector<std::string> camelCaseSplit(llvm::StringRef str);
        static std::vector<std::string> underscoreSplit(llvm::StringRef str);
        static std::vector<std::string> split(llvm::StringRef str);

        // Get the joint word set of two sentences.
        static std::vector<std::string> getJointWordSet(llvm::ArrayRef<std::string> sentence1,
                                                        llvm::ArrayRef<std::string> sentence2);

        // Compute the value of the lexical vector corresponding to
        // an element of the joint word set.
        float getLexicalCell(llvm::StringRef str, llvm::ArrayRef<std::string> sentence);

        // Compute the similarity of two word vectors.
        float getWordVecSimilarity(llvm::ArrayRef<std::string> sentence1,
                                   llvm::ArrayRef<std::string> sentence2);

    public:
        SentenceDistanceCalculator();

        // Compute the similarity of two sentences.
        float getSimilarity(llvm::StringRef sentence1,
                            llvm::StringRef sentence2);
    };
}

#endif //MARCO_CODEGEN_LOWERING_DISTANCE_SENTENCEDISTANCECALCULATOR_H