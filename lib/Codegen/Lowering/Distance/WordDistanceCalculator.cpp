#include "marco/Codegen/Lowering/Distance/WordDistanceCalculator.h"
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

using namespace std;

namespace marco::codegen::lowering
{

    WordDistanceCalculator::WordDistanceCalculator() :
        databaseReader(DatabaseReader::getInstance()),
        alpha(0.2), beta(0.45),
        semanticThreshold(0.2) {}

    void WordDistanceCalculator::analyzeByWords(llvm::StringRef word1,
                                                llvm::StringRef word2)
    {
        vector<Synset> synsets1 = databaseReader.getSynsets(word1);
        vector<Synset> synsets2 = databaseReader.getSynsets(word2);

        analyzeBySynsetGroups(synsets1, synsets2);
    }

    void WordDistanceCalculator::analyzeBySynsetGroups(llvm::ArrayRef<Synset> group1,
                                                       llvm::ArrayRef<Synset> group2)
    {
        if (group1.empty() || group2.empty())
        {
            closestSynsets = {-1, -1};
            combinedDistance = -1;
            lca = -1;
            return;
        }

        // We use a tuple of three elements to store:
        //  - the current synset
        //  - the distance traveled so far
        //  - the initial value of the synset.
        vector<tuple<Synset, int, Synset>> leaves1;
        vector<tuple<Synset, int, Synset>> nodes1;
        for (const Synset& synset : group1)
        {
            leaves1.push_back(make_tuple(synset, 0, synset));
            nodes1.push_back(make_tuple(synset, 0, synset));
        }

        vector<tuple<Synset, int, Synset>> leaves2;
        vector<tuple<Synset, int, Synset>> nodes2;
        for (const Synset& synset : group2)
        {
            leaves2.push_back(make_tuple(synset, 0, synset));
            nodes2.push_back(make_tuple(synset, 0, synset));
        }

        while (leaves1.size() > 0 || leaves2.size() > 0)
        {
            // Grow the first tree.
            int length = leaves1.size();
            for (int i = 0; i < length; i++)
            {
                vector<Synset> hypernyms = databaseReader.getHypernyms(get<0>(leaves1[i]));
                for (const Synset& hypernym : hypernyms)
                {
                    nodes1.push_back(make_tuple(hypernym, get<1>(leaves1[i]) + 1, get<2>(leaves1[i])));
                    leaves1.push_back(make_tuple(hypernym, get<1>(leaves1[i]) + 1, get<2>(leaves1[i])));
                }
            }

            for (int i = 0; i < length; i++)
            {
                leaves1.erase(leaves1.begin());
            }

            // Grow the second tree.
            int length2 = leaves2.size();
            for (int i = 0; i < length2; i++)
            {
                vector<Synset> hypernyms = databaseReader.getHypernyms(get<0>(leaves2[i]));
                for (const Synset& hypernym : hypernyms)
                {
                    nodes2.push_back(make_tuple(hypernym, get<1>(leaves2[i]) + 1, get<2>(leaves2[i])));
                    leaves2.push_back(make_tuple(hypernym, get<1>(leaves2[i]) + 1, get<2>(leaves2[i])));
                }
            }

            for (int i = 0; i < length2; i++)
            {
                leaves2.erase(leaves2.begin());
            }

            // Check if leaves1 and nodes2 have any common synsets.
            for (tuple<Synset, int, Synset>& node1 : leaves1)
            {
                for (tuple<Synset, int, Synset>& node2 : nodes2)
                {
                    if (get<0>(node1) == get<0>(node2))
                    {
                        lca = get<0>(node1);
                        combinedDistance = get<1>(node1) + get<1>(node2);
                        closestSynsets = {get<2>(node1), get<2>(node2)};
                        return;
                    }
                }
            }

            // Check if leaves2 and nodes1 have any common synsets.
            for (tuple<Synset, int, Synset>& node1 : nodes1)
            {
                for (tuple<Synset, int, Synset>& node2 : leaves2)
                {
                    if (get<0>(node1) == get<0>(node2))
                    {
                        lca = get<0>(node1);
                        combinedDistance = get<1>(node1) + get<1>(node2);
                        closestSynsets = {get<2>(node1), get<2>(node2)};
                        return;
                    }
                }
            }
        }

        // If no common synsets were found...
        closestSynsets = {-1, -1};
        lca = -1;
    }

    float WordDistanceCalculator::similarityFunction(int l, int h) const
    {
        if (l == -1 || h == -1) { return 0; }

        // Calculate the semantic similarity between two words.
        float numerator = exp(beta * h) - exp(- beta * h);
        float denominator = exp(beta * h) + exp(- beta * h);

        return exp(- alpha * l) * (numerator / denominator);
    }

    float WordDistanceCalculator::getNormalizedSimilarity() const
    {
        // Get the product of the information content of the closest synsets.
        float ic1 = getInformationContent(closestSynsets.first);
        float ic2 = getInformationContent(closestSynsets.second);

        // Get the semantic similarity between the two words.
        float similarity = similarityFunction(combinedDistance, distanceToRoot(lca));
        if (similarity < semanticThreshold) { return 0; }
        similarity = similarity * ic1 * ic2;
        
        return similarity;
    }

    float WordDistanceCalculator::getInformationContent(const Synset& synset) const
    {
        if (synset == -1) { return 0; }

        int count = databaseReader.getSynsetCount(synset);
        int total = databaseReader.getTotalSynsetCount();

        return 1 - log((float) count + 1) / log((float) total + 1);
    }

    int WordDistanceCalculator::distanceToRoot(const Synset& synset) const
    {
        if (synset == -1) { return -1; }

        // Create a vector of nodes to check.
        vector<pair<Synset, int>> nodes = {{synset, 0}};

        pair<Synset, int> node;
        while (!nodes.empty())
        {
            // Get the first node in the vector.
            node = nodes.front();
            nodes.erase(nodes.begin());

            // Get the hypernyms of the node.
            vector<Synset> hypernyms = databaseReader.getHypernyms(node.first);

            // If no hypernyms were found, return the distance.
            if (hypernyms.empty()) { break; }

            // Add the hypernyms to the vector.
            for (const Synset& hypernym : hypernyms)
            {
                nodes.push_back({hypernym, node.second + 1});
            }
        }

        return node.second;
    }

    const pair<int, int> WordDistanceCalculator::getTreeParameters() const
    {
        return {combinedDistance, distanceToRoot(lca)};
    }

    const pair<Synset, Synset> WordDistanceCalculator::getClosestSynsets() const
    {
        return closestSynsets;
    }

    Synset WordDistanceCalculator::getLCA() const
    {
        return lca;
    }
}