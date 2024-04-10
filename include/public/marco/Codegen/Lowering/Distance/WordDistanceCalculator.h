#include "marco/Codegen/Lowering/Distance/DatabaseReader.h"
#include <string>

namespace marco::codegen::lowering
{
    class WordDistanceCalculator {
    private:
        // To get the information about the DAG distance between two words and
        // the LCA distance to the root synset, we use the DatabaseReader.
        const DatabaseReader databaseReader;

        // The two closest synsets to the root synset, within the two provided groups.
        std::pair<Synset, Synset> closestSynsets;

        // The LCA of the two groups of synsets.
        Synset lca;

        // The combined distance to the LCAs of the two groups of synsets.
        int combinedDistance;

        // This method finds the distance to the root synset.
        int distanceToRoot(const Synset& synset) const;

    public:
        WordDistanceCalculator();
        ~WordDistanceCalculator();

        const float alpha;
        const float beta;
        const float semanticThreshold;

        // This method finds the LCA of two words.
        void analyzeByWords(const std::string& word1, const std::string& word2);

        // This method finds the LCA of two groups of synsets.
        void analyzeBySynsetGroups(const std::vector<Synset>& synsets1,
                                const std::vector<Synset>& synsets2);

        // This method calculates the semantic similarity between two words using
        // the synset distance to the LCA and the LCA distance to the root synset.
        float similarityFunction(int l, int h) const;

        // Get the normalized similarity, accounting for the word's frequency,
        // and discarding low similarity values. Analyze the words before calling
        // this method.
        float getNormalizedSimilarity() const;

        // Get the information content of a synset.
        float getInformationContent(const Synset& synset) const;

        // Get the parameters l and h.
        const std::pair<int, int> getTreeParameters() const;

        // Get the closest synsets to the root synset.
        const std::pair<Synset, Synset> getClosestSynsets() const;

        // Get the LCA of the two groups of synsets.
        Synset getLCA() const;
    };
}