#ifndef MARCO_CODEGEN_LOWERING_DISTANCE_WORDDISTANCECALCULATOR_H
#define MARCO_CODEGEN_LOWERING_DISTANCE_WORDDISTANCECALCULATOR_H

#include "marco/Codegen/Lowering/Distance/DatabaseReader.h"

/**
 * @file WordDistanceCalculator.h
 * @brief Header file for the WordDistanceCalculator class.
 * 
 * This is a custom implementation of a semantic distance calculator
 * between two variable names. The idea is to calculate the distance
 * between two variable names by interpreting them as short sentences.
 * The distance is calculated by following the paper "Sentence similarity
 * based on semantic nets and corpus statistics" by Li, McLean, Bandar,
 * O'Shea, Crockett.
 * 
 * In this particular implementation, the l and h parameters, as well as
 * the alpha, beta, and semanticThreshold parameters, are based on the
 * paper's recommendations. The similarity function and the normalization
 * of the similarity are also based on the paper's recommendations.
 * 
 * Here's a brief summary discussion on the differences between the paper
 * and this implementation:
 * 
 * The meaning of least common subsumer (LCS) is poorly defined in the paper.
 * Our process is to get all possible synsets for each word and then calculate
 * the distance between all pairs of synsets, to find the smallest one. Each
 * distance is calculated by checking all hypernyms of the synsets. Since
 * some synsets have multiple hypernyms, the structure of the graph is not
 * a tree, but a directed acyclic graph. This seems to differ from the paper,
 * that provides a tree-like diagram. As a result, the distance calculation
 * is different, and the results are not the same.
 * 
 * The paper assigns a corpus frequency to each word, which is part of the
 * process for calculating the similarity between two words. This is implemented
 * in a weaker form in this project, by counting the word frequency using the
 * synset's granularity, as opposed to the word's itself. This was done because
 * a word-based frequency seemed to produce worse results.
 */

namespace marco::codegen::lowering
{
    /**
     * @class marco::codegen::lowering::WordDistanceCalculator
     * @brief A class that calculates the distance between two words.
     * 
     * Variables and words are two distinct entities in our system. A variable
     * is a composition of words, either by camel case or by underscores. A word
     * is a pure word. The task of comparing compositions of words is done by
     * the @ref marco::codegen::lowering::SentenceDistanceCalculator class,
     * which uses this class to compare individual words.
     * 
     * Comparing two words is done by considering two main factors: the distance
     * between the two words' synsets and the distance between the two words'
     * least common ancestor (LCA) synset and the root synset. These two
     * scores are combined to form a similarity score, which is then normalized
     * by the frequency of the words in the corpus. In order to get information
     * about the synsets, this class uses the @ref marco::codegen::lowering::DatabaseReader
     * class to access the WordNet database.
     */
    class WordDistanceCalculator {
    private:
        // We use the singleton DatabaseReader to access the database.
        DatabaseReader& databaseReader;

        /**
         * @brief A helper struct to store the two closest synsets within
         * the two provided groups of synsets.
         * 
         * During comparisons, two clusters of synsets need to be compared,
         * and the two closest synsets within these clusters need are stored
         * in this struct.
         */
        std::pair<Synset, Synset> closestSynsets;

        /**
         * @brief The least common ancestor (LCA) of the two groups of synsets.
         * 
         * Given two synsets, the LCA is the synset that minimizes the distance
         * to both synsets.
         */
        Synset lca;

        /**
         * @brief The l parameter of the tree, which is the distance between
         * the two synsets, passing through the LCA.
         */
        int combinedDistance;

        /**
         * @brief This method calculates the distance to the root
         * synset of a given synset.
         * 
         * The h parameter of the tree is the distance between the LCA
         * and the root synset, which can be calculated through this method.
         * Note that the distance to the root is a measure of how general
         * a synset is. The higher the distance, the more specific the synset.
         * A word such as "modelica" would have a high distance to the root,
         * whereas a word such as "thing" would have a low distance to the root.
         */
        int distanceToRoot(const Synset& synset) const;

        /**
         * @brief Parameters for the similarity function.
         * 
         * The similarity function is a function that calculates the similarity
         * between two words. The function is a weighted combination of the
         * distance between the two synsets and the distance between the LCA
         * and the root synset. The alpha and beta parameters are used to
         * tune the function.
         */
        const float alpha;

        /**
         * @brief Parameters for the similarity function.
         * @see alpha
         */
        const float beta;

        /**
         * @brief A low threshold for the similarity function.
         * 
         * Given a low similarity value, the value is discarded.
         * This threshold is used to filter out low similarity values.
         */
        const float semanticThreshold;

        /**
         * @brief Analyzes two clusters of synsets and calculates relevant
         * information, such as the LCA, the closest synsets, and the tree
         * parameters. This information is stored in the class' state and can
         * be accessed through the corresponding getter methods.
         */
        void analyzeBySynsetGroups(llvm::ArrayRef<Synset> synsets1,
                                   llvm::ArrayRef<Synset> synsets2);

        /**
         * @brief Calculates the similarity between two words.
         * 
         * @param l The distance between the two synsets, passing through the LCA.
         * @param h The distance between the LCA and the root synset.
         * 
         * The similarity function is a weighted combination of the two parameters,
         * using the alpha and beta parameters. The function is used to calculate
         * the similarity between two words.
         */
        float similarityFunction(int l, int h) const;

        /**
         * @brief Gets the information content of a synset.
         * 
         * The information content of a synset is a measure of how informative
         * the synset is. The information content is calculated by the number
         * of occurrences of the synset in the corpus. The more occurrences,
         * the less informative the synset is. Synsets with a high information
         * are favored in the similarity calculation after normalization.
         */
        float getInformationContent(const Synset& synset) const;

        /**
         * @brief Gets the tree parameters of the analyzed words.
         * Note that this method should be called after the analyzeBySynsetGroups
         * method is run, as the closest synsets are stored in the class' state.
         */
        const std::pair<int, int> getTreeParameters() const;

        /**
         * @brief Gets the two closest synsets within the two groups of synsets.
         * 
         * Note that this method should be called after the analyzeBySynsetGroups
         * method is run, as the closest synsets are stored in the class' state.
         */
        const std::pair<Synset, Synset> getClosestSynsets() const;

        /**
         * @brief Gets the least common ancestor (LCA) of the two groups of synsets.
         */
        Synset getLCA() const;

    public:
        WordDistanceCalculator();

        /**
         * @brief Analyzes two words and calculates relevant information,
         * such as the LCA, the closest synsets, and the tree parameters.
         * This information is stored in the class' state and can be accessed
         * through the corresponding getter methods.
         * 
         * @param word1 The first word to analyze.
         * @param word2 The second word to analyze.
         * 
         * Note that this method compares words, and not synsets. The synsets
         * are retrieved from the WordNet database using the DatabaseReader,
         * so the function is actually comparing clusters of synsets.
         */
        void analyzeByWords(llvm::StringRef word1, llvm::StringRef word2);

        /**
         * @brief Gets the normalized similarity between two words.
         * 
         * Since rarer words are more informative, the similarity between two
         * words is normalized by the frequency of the words in the corpus.
         * It's more common to find the word "thing" than the word "vertebrate",
         * so it's easier to mistakingly identify the word "thing" as a synonym
         * of another word. This method normalizes the similarity by the frequency
         * of the words in the corpus.
         */
        float getNormalizedSimilarity() const;
    };
}

#endif //MARCO_CODEGEN_LOWERING_DISTANCE_WORDDISTANCECALCULATOR_H