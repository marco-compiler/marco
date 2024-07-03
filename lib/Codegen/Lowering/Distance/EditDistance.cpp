#include "marco/Codegen/Lowering/Distance/EditDistance.h"
#include "marco/Codegen/Lowering/Distance/EditDistanceParameters.h"
#include <vector>

using namespace ::marco;
using namespace ::marco::codegen::lowering;

namespace marco::codegen::lowering
{
  unsigned int editDistance(const std::string &actual, const std::string &expected) {
    unsigned int m = actual.length();
    unsigned int n = expected.length();

    constexpr unsigned int maxCost = 65535;

    // Create a matrix to store partial distances for the dynamic programming algorithm.
    std::vector<unsigned int> distances((n+1)*(m+1), 0);

    // Initialize the first row of the matrix (all deletions).
    for (unsigned int j=0; j<=n; ++j) {
      distances[j] = j*editDistanceDeletionCost;
    }

    // Initialize the first column of the matrix (all insertions).
    unsigned int first_char_idx = (n > 0) ? charToSubstitutionCostsIndex(expected[0]) : 0;
    for (unsigned int i=1; i<=m; ++i) {
      unsigned int actual_char_idx = charToSubstitutionCostsIndex(actual[i-1]);
      unsigned int char_insertion_cost = (n > 0) ?
                                          editDistanceSubsitutionCosts[actual_char_idx][first_char_idx] :
                                          editDistanceLargestSubstitutionCost;
      distances[i*(n+1)] = distances[(i-1)*(n+1)] + editDistanceBaseInsertionCost + char_insertion_cost;
    } 

    // Compute the distance matrix.
    for (unsigned int i=1; i<=m; ++i) {
      for (unsigned int j=1; j<=n; ++j) {
        unsigned int index = i*(n+1) + j;
        unsigned int actual_char_idx = charToSubstitutionCostsIndex(actual[i-1]);
        unsigned int expected_char_idx = charToSubstitutionCostsIndex(expected[j-1]);

        // Calculate the distance in case of deletion.
        unsigned int deletion_result = distances[index - 1] + editDistanceDeletionCost;

        // Calculate the distance in case of insertion.
        unsigned int current_char_cost = editDistanceSubsitutionCosts[actual_char_idx][expected_char_idx];
        unsigned int next_char_cost = (j < n) ?
              editDistanceSubsitutionCosts[actual_char_idx][charToSubstitutionCostsIndex(expected[j])] :
              editDistanceLargestSubstitutionCost;
        unsigned int insertion_result = distances[index - (n+1)] + editDistanceBaseInsertionCost + 
                                              std::min(current_char_cost, next_char_cost);

        // Calculate the distance in case of substitution.
        unsigned int substitution_result = distances[index - (n+1) - 1] + 
                                                 editDistanceSubsitutionCosts[actual_char_idx][expected_char_idx];

        // Calculate the distance in case of transposition.
        unsigned int transposition_result = 
                           (i >= 2 && j >= 2 && actual[i-1] == expected[j-2] && actual[i-2] == expected[j-1]) ?
                           distances[index - 2*(n+1) - 2] + editDistanceTranspositionCost :
                           maxCost;

        // Select the operation with the lowest cost.
        distances[index] = std::min(std::min(deletion_result, insertion_result), 
                                    std::min(substitution_result, transposition_result));
      }
    }

    return distances[(n+1)*(m+1)-1];
  }
}