#include "marco/Codegen/Lowering/Distance/EditDistance.h"
#include <vector>

using namespace ::marco;
using namespace ::marco::codegen::lowering;

namespace marco::codegen::lowering
{
  unsigned int EditDistance::editDistance(const std::string &actual, const std::string &expected) {
    const unsigned int m = actual.length();
    const unsigned int n = expected.length();

    // Create a matrix to store partial distances for the dynamic programming algorithm.
    std::vector<unsigned int> distances((n+1)*(m+1), 0);

    // Initialize the first row of the matrix (all deletions).
    for (unsigned int j=0; j<=n; ++j) {
      distances[j] = j*deletion_cost;
    }

    // Initialize the first column of the matrix (all insertions).
    const unsigned int first_char_idx = (n > 0) ? charToIndex(expected[0]) : 0;
    for (unsigned int i=1; i<=m; ++i) {
      const unsigned int actual_char_idx = charToIndex(actual[i-1]);
      const unsigned int char_insertion_cost = (n > 0) ?
                                          substitution_costs[actual_char_idx][first_char_idx] :
                                          largest_substitution_cost;
      distances[i*(n+1)] = distances[(i-1)*(n+1)] + base_insertion_cost + char_insertion_cost;
    } 

    // Compute the distance matrix.
    for (unsigned int i=1; i<=m; ++i) {
      for (unsigned int j=1; j<=n; ++j) {
        const unsigned int index = i*(n+1) + j;
        const unsigned int actual_char_idx = charToIndex(actual[i-1]);
        const unsigned int expected_char_idx = charToIndex(expected[j-1]);

        // Calculate the distance in case of deletion.
        const unsigned int deletion_result = distances[index - 1] + deletion_cost;

        // Calculate the distance in case of insertion.
        const unsigned int current_char_cost = substitution_costs[actual_char_idx][expected_char_idx];
        const unsigned int next_char_cost = (j < n) ?
              substitution_costs[actual_char_idx][charToIndex(expected[j])] :
              largest_substitution_cost;
        const unsigned int insertion_result = distances[index - (n+1)] + base_insertion_cost + 
                                              std::min(current_char_cost, next_char_cost);

        // Calculate the distance in case of substitution.
        const unsigned int substitution_result = distances[index - (n+1) - 1] + 
                                                 substitution_costs[actual_char_idx][expected_char_idx];

        // Calculate the distance in case of transposition.
        const unsigned int transposition_result = 
                           (i >= 2 && j >= 2 && actual[i-1] == expected[j-2] && actual[i-2] == expected[j-1]) ?
                           distances[index - 2*(n+1) - 2] + transposition_cost :
                           max_cost;

        // Select the operation with the lowest cost.
        distances[index] = std::min(std::min(deletion_result, insertion_result), 
                                    std::min(substitution_result, transposition_result));
      }
    }

    return distances[(n+1)*(m+1)-1];
  }

  const unsigned int EditDistance::largest_substitution_cost = 21;
  // The costs are proportional to the physical distances between keys in a keyboard with a QWERTY layout.
  // In case of an upper/lower case mismatch, the distance is increased by 2, as this value
  // was found to work well in the performed tests.
  const std::array<std::array<unsigned int, 63>, 63> EditDistance::substitution_costs =
        {0, 18, 16, 14, 12, 10, 8, 6, 4, 2, 19, 12, 15, 16, 15, 14, 12, 10, 6, 8, 7, 6, 9, 10, 4, 4, 19, 13, 18, 11, 7, 14, 17, 17, 9, 19, 6, 17, 10, 13, 14, 13, 12, 10, 8, 4, 6, 5, 4, 7, 8, 2, 2, 17, 11, 16, 9, 5, 12, 15, 15, 7, 17, 18, 0, 2, 4, 6, 8, 10, 12, 14, 16, 6, 14, 10, 8, 7, 10, 12, 14, 17, 16, 18, 19, 17, 15, 19, 21, 4, 9, 7, 11, 15, 12, 6, 9, 13, 8, 21, 4, 12, 8, 6, 5, 8, 10, 12, 15, 14, 16, 17, 15, 13, 17, 19, 2, 7, 5, 9, 13, 10, 4, 7, 11, 6, 16, 2, 0, 2, 4, 6, 8, 10, 12, 14, 6, 12, 9, 7, 6, 8, 10, 12, 15, 14, 16, 18, 15, 14, 17, 19, 4, 7, 6, 9, 13, 10, 4, 8, 11, 8, 19, 4, 10, 7, 5, 4, 6, 8, 10, 13, 12, 14, 16, 13, 12, 15, 17, 2, 5, 4, 7, 11, 8, 2, 6, 9, 6, 14, 4, 2, 0, 2, 4, 6, 8, 10, 12, 7, 10, 8, 6, 4, 7, 8, 10, 13, 12, 14, 16, 14, 12, 15, 17, 6, 6, 6, 7, 11, 9, 4, 8, 9, 8, 17, 5, 8, 6, 4, 2, 5, 6, 8, 11, 10, 12, 14, 12, 10, 13, 15, 4, 4, 4, 5, 9, 7, 2, 6, 7, 6, 12, 6, 4, 2, 0, 2, 4, 6, 8, 10, 8, 9, 8, 6, 4, 6, 7, 8, 11, 10, 12, 14, 12, 10, 13, 15, 7, 4, 7, 6, 9, 8, 6, 8, 7, 9, 15, 6, 7, 6, 4, 2, 4, 5, 6, 9, 8, 10, 12, 10, 8, 11, 13, 5, 2, 5, 4, 7, 6, 4, 6, 5, 7, 10, 8, 6, 4, 2, 0, 2, 4, 6, 8, 10, 8, 8, 7, 6, 6, 6, 7, 9, 8, 10, 12, 10, 9, 11, 13, 9, 4, 8, 4, 7, 8, 7, 9, 6, 10, 13, 8, 6, 6, 5, 4, 4, 4, 5, 7, 6, 8, 10, 8, 7, 9, 11, 7, 2, 6, 2, 5, 6, 5, 7, 4, 8, 8, 10, 8, 6, 4, 2, 0, 2, 4, 6, 12, 8, 9, 8, 7, 7, 6, 6, 7, 7, 8, 10, 9, 8, 9, 11, 11, 6, 10, 4, 6, 8, 9, 10, 4, 12, 12, 10, 6, 7, 6, 5, 5, 4, 4, 5, 5, 6, 8, 7, 6, 7, 9, 9, 4, 8, 2, 4, 6, 7, 8, 2, 10, 6, 12, 10, 8, 6, 4, 2, 0, 2, 4, 14, 8, 10, 10, 9, 8, 7, 6, 6, 6, 7, 8, 8, 8, 7, 9, 13, 7, 12, 6, 4, 9, 11, 12, 4, 14, 10, 12, 6, 8, 8, 7, 6, 5, 4, 4, 4, 5, 6, 6, 6, 5, 7, 11, 5, 10, 4, 2, 7, 9, 10, 2, 12, 4, 14, 12, 10, 8, 6, 4, 2, 0, 2, 16, 9, 12, 12, 11, 10, 8, 7, 4, 6, 6, 7, 8, 8, 6, 7, 15, 9, 14, 7, 4, 10, 13, 14, 6, 15, 8, 14, 7, 10, 10, 9, 8, 6, 5, 2, 4, 4, 5, 6, 6, 4, 5, 13, 7, 12, 5, 2, 8, 11, 12, 4, 13, 2, 16, 14, 12, 10, 8, 6, 4, 2, 0, 18, 10, 14, 14, 13, 12, 10, 8, 4, 7, 6, 6, 8, 9, 4, 6, 17, 11, 16, 9, 6, 12, 15, 15, 7, 17, 7, 16, 8, 12, 12, 11, 10, 8, 6, 2, 5, 4, 4, 6, 7, 2, 4, 15, 9, 14, 7, 4, 10, 13, 13, 5, 15, 19, 6, 6, 7, 8, 10, 12, 14, 16, 18, 0, 9, 5, 4, 4, 6, 8, 10, 14, 12, 14, 16, 13, 11, 16, 18, 2, 6, 2, 8, 12, 7, 3, 4, 10, 2, 21, 2, 11, 7, 6, 6, 8, 10, 12, 16, 14, 16, 18, 15, 13, 18, 20, 4, 8, 4, 10, 14, 9, 5, 6, 12, 4, 12, 14, 12, 10, 9, 8, 8, 8, 9, 10, 9, 0, 4, 5, 6, 4, 2, 2, 6, 4, 5, 7, 4, 2, 8, 10, 10, 5, 7, 4, 5, 2, 8, 6, 4, 8, 12, 11, 2, 6, 7, 8, 6, 4, 4, 8, 6, 7, 9, 6, 4, 10, 12, 12, 7, 9, 6, 7, 4, 10, 8, 6, 10, 15, 10, 9, 8, 8, 8, 9, 10, 12, 14, 5, 4, 0, 2, 4, 2, 4, 5, 10, 7, 9, 11, 8, 6, 12, 14, 6, 4, 4, 5, 8, 2, 5, 2, 6, 4, 16, 7, 6, 2, 4, 6, 4, 6, 7, 12, 9, 11, 13, 10, 8, 14, 16, 8, 6, 6, 7, 10, 4, 7, 4, 8, 6, 16, 8, 7, 6, 6, 7, 8, 10, 12, 14, 4, 5, 2, 0, 2, 2, 4, 6, 10, 8, 10, 12, 9, 7, 12, 14, 4, 3, 2, 4, 8, 4, 3, 2, 6, 4, 17, 6, 7, 4, 2, 4, 4, 6, 8, 12, 10, 12, 14, 11, 9, 14, 16, 6, 5, 4, 6, 10, 6, 5, 4, 8, 6, 15, 7, 6, 4, 4, 6, 7, 9, 11, 13, 4, 6, 4, 2, 0, 3, 4, 6, 10, 8, 10, 12, 10, 8, 12, 14, 4, 2, 3, 4, 8, 5, 2, 4, 6, 5, 18, 6, 8, 6, 4, 2, 5, 6, 8, 12, 10, 12, 14, 12, 10, 14, 16, 6, 4, 5, 6, 10, 7, 4, 6, 8, 7, 14, 10, 8, 7, 6, 6, 7, 8, 10, 12, 6, 4, 2, 2, 3, 0, 2, 4, 8, 6, 8, 10, 7, 5, 10, 12, 6, 2, 4, 3, 6, 2, 4, 4, 4, 5, 15, 8, 6, 4, 4, 5, 2, 4, 6, 10, 8, 10, 12, 9, 7, 12, 14, 8, 4, 6, 5, 8, 4, 6, 6, 6, 7, 12, 12, 10, 8, 7, 6, 6, 7, 8, 10, 8, 2, 4, 4, 4, 2, 0, 2, 6, 4, 6, 8, 5, 4, 8, 10, 8, 3, 6, 2, 4, 2, 6, 5, 3, 7, 13, 10, 4, 6, 6, 6, 4, 2, 4, 8, 6, 8, 10, 7, 6, 10, 12, 10, 5, 8, 4, 6, 4, 8, 7, 5, 9, 10, 14, 12, 10, 8, 7, 6, 6, 7, 8, 10, 2, 5, 6, 6, 4, 2, 0, 4, 2, 4, 6, 4, 2, 6, 8, 10, 4, 8, 3, 3, 4, 8, 7, 2, 9, 11, 12, 4, 7, 8, 8, 6, 4, 2, 6, 4, 6, 8, 6, 4, 8, 10, 12, 6, 10, 5, 5, 6, 10, 9, 4, 11, 6, 17, 15, 13, 11, 9, 7, 6, 4, 4, 14, 6, 10, 10, 10, 8, 6, 4, 0, 3, 2, 3, 4, 5, 2, 4, 14, 8, 12, 6, 2, 8, 12, 12, 4, 14, 8, 16, 8, 12, 12, 12, 10, 8, 6, 2, 5, 4, 5, 6, 7, 4, 6, 16, 10, 14, 8, 4, 10, 14, 14, 6, 16, 8, 16, 14, 12, 10, 8, 7, 6, 6, 7, 12, 4, 7, 8, 8, 6, 4, 2, 3, 0, 2, 4, 2, 2, 4, 6, 12, 6, 10, 4, 2, 5, 10, 9, 3, 11, 9, 14, 6, 9, 10, 10, 8, 6, 4, 5, 2, 4, 6, 4, 4, 6, 8, 14, 8, 12, 6, 4, 7, 12, 11, 5, 13, 7, 18, 16, 14, 12, 10, 8, 7, 6, 6, 14, 5, 9, 10, 10, 8, 6, 4, 2, 2, 0, 2, 2, 4, 3, 4, 14, 8, 12, 6, 3, 7, 12, 11, 4, 13, 7, 16, 7, 11, 12, 12, 10, 8, 6, 4, 4, 2, 4, 4, 6, 5, 6, 16, 10, 14, 8, 5, 9, 14, 13, 6, 15, 6, 19, 18, 16, 14, 12, 10, 8, 7, 6, 16, 7, 11, 12, 12, 10, 8, 6, 3, 4, 2, 0, 4, 5, 2, 3, 16, 10, 14, 8, 4, 9, 14, 13, 6, 15, 6, 18, 9, 13, 14, 14, 12, 10, 8, 5, 6, 4, 2, 6, 7, 4, 5, 18, 12, 16, 10, 6, 11, 16, 15, 8, 17, 9, 17, 15, 14, 12, 10, 9, 8, 8, 8, 13, 4, 8, 9, 10, 7, 5, 4, 4, 2, 2, 4, 0, 2, 5, 6, 14, 8, 11, 6, 4, 6, 12, 10, 5, 12, 8, 15, 6, 10, 11, 12, 9, 7, 6, 6, 4, 4, 6, 2, 4, 7, 8, 16, 10, 13, 8, 6, 8, 14, 12, 7, 14, 10, 15, 14, 12, 10, 9, 8, 8, 8, 9, 11, 2, 6, 7, 8, 5, 4, 2, 5, 2, 4, 5, 2, 0, 6, 8, 12, 6, 9, 5, 4, 4, 10, 8, 4, 10, 10, 13, 4, 8, 9, 10, 7, 6, 4, 7, 4, 6, 7, 4, 2, 8, 10, 14, 8, 11, 7, 6, 6, 12, 10, 6, 12, 4, 19, 17, 15, 13, 11, 9, 7, 6, 4, 16, 8, 12, 12, 12, 10, 8, 6, 2, 4, 3, 2, 5, 6, 0, 2, 16, 10, 14, 8, 4, 10, 14, 14, 6, 16, 7, 18, 10, 14, 14, 14, 12, 10, 8, 4, 6, 5, 4, 7, 8, 2, 4, 18, 12, 16, 10, 6, 12, 16, 16, 8, 18, 4, 21, 19, 17, 15, 13, 11, 9, 7, 6, 18, 10, 14, 14, 14, 12, 10, 8, 4, 6, 4, 3, 6, 8, 2, 0, 18, 12, 16, 10, 6, 12, 16, 16, 8, 17, 6, 20, 12, 16, 16, 16, 14, 12, 10, 6, 8, 6, 5, 8, 10, 4, 2, 20, 14, 18, 12, 8, 14, 18, 18, 10, 19, 19, 4, 4, 6, 7, 9, 11, 13, 15, 17, 2, 10, 6, 4, 4, 6, 8, 10, 14, 12, 14, 16, 14, 12, 16, 18, 0, 6, 3, 8, 12, 8, 2, 5, 10, 4, 21, 4, 12, 8, 6, 6, 8, 10, 12, 16, 14, 16, 18, 16, 14, 18, 20, 2, 8, 5, 10, 14, 10, 4, 7, 12, 6, 13, 9, 7, 6, 4, 4, 6, 7, 9, 11, 6, 5, 4, 3, 2, 2, 3, 4, 8, 6, 8, 10, 8, 6, 10, 12, 6, 0, 4, 2, 6, 4, 4, 5, 4, 6, 16, 8, 7, 6, 5, 4, 4, 5, 6, 10, 8, 10, 12, 10, 8, 12, 14, 8, 2, 6, 4, 8, 6, 6, 7, 6, 8, 18, 7, 6, 6, 7, 8, 10, 12, 14, 16, 2, 7, 4, 2, 3, 4, 6, 8, 12, 10, 12, 14, 11, 9, 14, 16, 3, 4, 0, 6, 10, 5, 2, 2, 8, 2, 19, 4, 9, 6, 4, 5, 6, 8, 10, 14, 12, 14, 16, 13, 11, 16, 18, 5, 6, 2, 8, 12, 7, 4, 4, 10, 4, 11, 11, 9, 7, 6, 4, 4, 6, 7, 9, 8, 4, 5, 4, 4, 3, 2, 3, 6, 4, 6, 8, 6, 5, 8, 10, 8, 2, 6, 0, 4, 4, 6, 6, 2, 8, 14, 10, 6, 7, 6, 6, 5, 4, 5, 8, 6, 8, 10, 8, 7, 10, 12, 10, 4, 8, 2, 6, 6, 8, 8, 4, 10, 7, 15, 13, 11, 9, 7, 6, 4, 4, 6, 12, 5, 8, 8, 8, 6, 4, 3, 2, 2, 3, 4, 4, 4, 4, 6, 12, 6, 10, 4, 0, 6, 10, 10, 2, 12, 10, 14, 7, 10, 10, 10, 8, 6, 5, 4, 4, 5, 6, 6, 6, 6, 8, 14, 8, 12, 6, 2, 8, 12, 12, 4, 14, 14, 12, 10, 9, 8, 8, 8, 9, 10, 12, 7, 2, 2, 4, 5, 2, 2, 4, 8, 5, 7, 9, 6, 4, 10, 12, 8, 4, 5, 4, 6, 0, 6, 4, 5, 6, 14, 9, 4, 4, 6, 7, 4, 4, 6, 10, 7, 9, 11, 8, 6, 12, 14, 10, 6, 7, 6, 8, 2, 8, 6, 7, 8, 17, 6, 4, 4, 6, 7, 9, 11, 13, 15, 3, 8, 5, 3, 2, 4, 6, 8, 12, 10, 12, 14, 12, 10, 14, 16, 2, 4, 2, 6, 10, 6, 0, 4, 8, 4, 19, 5, 10, 7, 5, 4, 6, 8, 10, 14, 12, 14, 16, 14, 12, 16, 18, 4, 6, 4, 8, 12, 8, 2, 6, 10, 6, 17, 9, 8, 8, 8, 9, 10, 12, 14, 15, 4, 6, 2, 2, 4, 4, 5, 7, 12, 9, 11, 13, 10, 8, 14, 16, 5, 5, 2, 6, 10, 4, 4, 0, 8, 2, 18, 6, 8, 4, 4, 6, 6, 7, 9, 14, 11, 13, 15, 12, 10, 16, 18, 7, 7, 4, 8, 12, 6, 6, 2, 10, 4, 9, 13, 11, 9, 7, 6, 4, 4, 6, 7, 10, 4, 6, 6, 6, 4, 3, 2, 4, 3, 4, 6, 5, 4, 6, 8, 10, 4, 8, 2, 2, 5, 8, 8, 0, 10, 12, 12, 6, 8, 8, 8, 6, 5, 4, 6, 5, 6, 8, 7, 6, 8, 10, 12, 6, 10, 4, 4, 7, 10, 10, 2, 12, 19, 8, 8, 8, 9, 10, 12, 14, 15, 17, 2, 8, 4, 4, 5, 5, 7, 9, 14, 11, 13, 15, 12, 10, 16, 17, 4, 6, 2, 8, 12, 6, 4, 2, 10, 0, 20, 4, 10, 6, 6, 7, 7, 9, 11, 16, 13, 15, 17, 14, 12, 18, 19, 6, 8, 4, 10, 14, 8, 6, 4, 12, 2, 6, 21, 19, 17, 15, 13, 12, 10, 8, 7, 21, 12, 16, 17, 18, 15, 13, 11, 8, 9, 7, 6, 8, 10, 7, 6, 21, 16, 19, 14, 10, 14, 19, 18, 12, 20, 0, 19, 10, 14, 15, 16, 13, 11, 9, 6, 7, 5, 4, 6, 8, 5, 4, 19, 14, 17, 12, 8, 12, 17, 16, 10, 18, 17, 4, 4, 5, 6, 8, 10, 12, 14, 16, 2, 11, 7, 6, 6, 8, 10, 12, 16, 14, 16, 18, 15, 13, 18, 20, 4, 8, 4, 10, 14, 9, 5, 6, 12, 4, 19, 0, 9, 5, 4, 4, 6, 8, 10, 14, 12, 14, 16, 13, 11, 16, 18, 2, 6, 2, 8, 12, 7, 3, 4, 10, 2, 10, 12, 10, 8, 7, 6, 6, 6, 7, 8, 11, 2, 6, 7, 8, 6, 4, 4, 8, 6, 7, 9, 6, 4, 10, 12, 12, 7, 9, 6, 7, 4, 10, 8, 6, 10, 10, 9, 0, 4, 5, 6, 4, 2, 2, 6, 4, 5, 7, 4, 2, 8, 10, 10, 5, 7, 4, 5, 2, 8, 6, 4, 8, 13, 8, 7, 6, 6, 6, 7, 8, 10, 12, 7, 6, 2, 4, 6, 4, 6, 7, 12, 9, 11, 13, 10, 8, 14, 16, 8, 6, 6, 7, 10, 4, 7, 4, 8, 6, 14, 5, 4, 0, 2, 4, 2, 4, 5, 10, 7, 9, 11, 8, 6, 12, 14, 6, 4, 4, 5, 8, 2, 5, 2, 6, 4, 14, 6, 5, 4, 4, 5, 6, 8, 10, 12, 6, 7, 4, 2, 4, 4, 6, 8, 12, 10, 12, 14, 11, 9, 14, 16, 6, 5, 4, 6, 10, 6, 5, 4, 8, 6, 15, 4, 5, 2, 0, 2, 2, 4, 6, 10, 8, 10, 12, 9, 7, 12, 14, 4, 3, 2, 4, 8, 4, 3, 2, 6, 4, 13, 5, 4, 2, 2, 4, 5, 7, 9, 11, 6, 8, 6, 4, 2, 5, 6, 8, 12, 10, 12, 14, 12, 10, 14, 16, 6, 4, 5, 6, 10, 7, 4, 6, 8, 7, 16, 4, 6, 4, 2, 0, 3, 4, 6, 10, 8, 10, 12, 10, 8, 12, 14, 4, 2, 3, 4, 8, 5, 2, 4, 6, 5, 12, 8, 6, 5, 4, 4, 5, 6, 8, 10, 8, 6, 4, 4, 5, 2, 4, 6, 10, 8, 10, 12, 9, 7, 12, 14, 8, 4, 6, 5, 8, 4, 6, 6, 6, 7, 13, 6, 4, 2, 2, 3, 0, 2, 4, 8, 6, 8, 10, 7, 5, 10, 12, 6, 2, 4, 3, 6, 2, 4, 4, 4, 5, 10, 10, 8, 6, 5, 4, 4, 5, 6, 8, 10, 4, 6, 6, 6, 4, 2, 4, 8, 6, 8, 10, 7, 6, 10, 12, 10, 5, 8, 4, 6, 4, 8, 7, 5, 9, 11, 8, 2, 4, 4, 4, 2, 0, 2, 6, 4, 6, 8, 5, 4, 8, 10, 8, 3, 6, 2, 4, 2, 6, 5, 3, 7, 8, 12, 10, 8, 6, 5, 4, 4, 5, 6, 12, 4, 7, 8, 8, 6, 4, 2, 6, 4, 6, 8, 6, 4, 8, 10, 12, 6, 10, 5, 5, 6, 10, 9, 4, 11, 9, 10, 2, 5, 6, 6, 4, 2, 0, 4, 2, 4, 6, 4, 2, 6, 8, 10, 4, 8, 3, 3, 4, 8, 7, 2, 9, 4, 15, 13, 11, 9, 7, 5, 4, 2, 2, 16, 8, 12, 12, 12, 10, 8, 6, 2, 5, 4, 5, 6, 7, 4, 6, 16, 10, 14, 8, 4, 10, 14, 14, 6, 16, 6, 14, 6, 10, 10, 10, 8, 6, 4, 0, 3, 2, 3, 4, 5, 2, 4, 14, 8, 12, 6, 2, 8, 12, 12, 4, 14, 6, 14, 12, 10, 8, 6, 5, 4, 4, 5, 14, 6, 9, 10, 10, 8, 6, 4, 5, 2, 4, 6, 4, 4, 6, 8, 14, 8, 12, 6, 4, 7, 12, 11, 5, 13, 7, 12, 4, 7, 8, 8, 6, 4, 2, 3, 0, 2, 4, 2, 2, 4, 6, 12, 6, 10, 4, 2, 5, 10, 9, 3, 11, 5, 16, 14, 12, 10, 8, 6, 5, 4, 4, 16, 7, 11, 12, 12, 10, 8, 6, 4, 4, 2, 4, 4, 6, 5, 6, 16, 10, 14, 8, 5, 9, 14, 13, 6, 15, 5, 14, 5, 9, 10, 10, 8, 6, 4, 2, 2, 0, 2, 2, 4, 3, 4, 14, 8, 12, 6, 3, 7, 12, 11, 4, 13, 4, 17, 16, 14, 12, 10, 8, 6, 5, 4, 18, 9, 13, 14, 14, 12, 10, 8, 5, 6, 4, 2, 6, 7, 4, 5, 18, 12, 16, 10, 6, 11, 16, 15, 8, 17, 4, 16, 7, 11, 12, 12, 10, 8, 6, 3, 4, 2, 0, 4, 5, 2, 3, 16, 10, 14, 8, 4, 9, 14, 13, 6, 15, 7, 15, 13, 12, 10, 8, 7, 6, 6, 6, 15, 6, 10, 11, 12, 9, 7, 6, 6, 4, 4, 6, 2, 4, 7, 8, 16, 10, 13, 8, 6, 8, 14, 12, 7, 14, 6, 13, 4, 8, 9, 10, 7, 5, 4, 4, 2, 2, 4, 0, 2, 5, 6, 14, 8, 11, 6, 4, 6, 12, 10, 5, 12, 8, 13, 12, 10, 8, 7, 6, 6, 6, 7, 13, 4, 8, 9, 10, 7, 6, 4, 7, 4, 6, 7, 4, 2, 8, 10, 14, 8, 11, 7, 6, 6, 12, 10, 6, 12, 8, 11, 2, 6, 7, 8, 5, 4, 2, 5, 2, 4, 5, 2, 0, 6, 8, 12, 6, 9, 5, 4, 4, 10, 8, 4, 10, 2, 17, 15, 13, 11, 9, 7, 5, 4, 2, 18, 10, 14, 14, 14, 12, 10, 8, 4, 6, 5, 4, 7, 8, 2, 4, 18, 12, 16, 10, 6, 12, 16, 16, 8, 18, 5, 16, 8, 12, 12, 12, 10, 8, 6, 2, 4, 3, 2, 5, 6, 0, 2, 16, 10, 14, 8, 4, 10, 14, 14, 6, 16, 2, 19, 17, 15, 13, 11, 9, 7, 5, 4, 20, 12, 16, 16, 16, 14, 12, 10, 6, 8, 6, 5, 8, 10, 4, 2, 20, 14, 18, 12, 8, 14, 18, 18, 10, 19, 4, 18, 10, 14, 14, 14, 12, 10, 8, 4, 6, 4, 3, 6, 8, 2, 0, 18, 12, 16, 10, 6, 12, 16, 16, 8, 17, 17, 2, 2, 4, 5, 7, 9, 11, 13, 15, 4, 12, 8, 6, 6, 8, 10, 12, 16, 14, 16, 18, 16, 14, 18, 20, 2, 8, 5, 10, 14, 10, 4, 7, 12, 6, 19, 2, 10, 6, 4, 4, 6, 8, 10, 14, 12, 14, 16, 14, 12, 16, 18, 0, 6, 3, 8, 12, 8, 2, 5, 10, 4, 11, 7, 5, 4, 2, 2, 4, 5, 7, 9, 8, 7, 6, 5, 4, 4, 5, 6, 10, 8, 10, 12, 10, 8, 12, 14, 8, 2, 6, 4, 8, 6, 6, 7, 6, 8, 14, 6, 5, 4, 3, 2, 2, 3, 4, 8, 6, 8, 10, 8, 6, 10, 12, 6, 0, 4, 2, 6, 4, 4, 5, 4, 6, 16, 5, 4, 4, 5, 6, 8, 10, 12, 14, 4, 9, 6, 4, 5, 6, 8, 10, 14, 12, 14, 16, 13, 11, 16, 18, 5, 6, 2, 8, 12, 7, 4, 4, 10, 4, 17, 2, 7, 4, 2, 3, 4, 6, 8, 12, 10, 12, 14, 11, 9, 14, 16, 3, 4, 0, 6, 10, 5, 2, 2, 8, 2, 9, 9, 7, 5, 4, 2, 2, 4, 5, 7, 10, 6, 7, 6, 6, 5, 4, 5, 8, 6, 8, 10, 8, 7, 10, 12, 10, 4, 8, 2, 6, 6, 8, 8, 4, 10, 12, 8, 4, 5, 4, 4, 3, 2, 3, 6, 4, 6, 8, 6, 5, 8, 10, 8, 2, 6, 0, 4, 4, 6, 6, 2, 8, 5, 13, 11, 9, 7, 5, 4, 2, 2, 4, 14, 7, 10, 10, 10, 8, 6, 5, 4, 4, 5, 6, 6, 6, 6, 8, 14, 8, 12, 6, 2, 8, 12, 12, 4, 14, 8, 12, 5, 8, 8, 8, 6, 4, 3, 2, 2, 3, 4, 4, 4, 4, 6, 12, 6, 10, 4, 0, 6, 10, 10, 2, 12, 12, 10, 8, 7, 6, 6, 6, 7, 8, 10, 9, 4, 4, 6, 7, 4, 4, 6, 10, 7, 9, 11, 8, 6, 12, 14, 10, 6, 7, 6, 8, 2, 8, 6, 7, 8, 12, 7, 2, 2, 4, 5, 2, 2, 4, 8, 5, 7, 9, 6, 4, 10, 12, 8, 4, 5, 4, 6, 0, 6, 4, 5, 6, 15, 4, 2, 2, 4, 5, 7, 9, 11, 13, 5, 10, 7, 5, 4, 6, 8, 10, 14, 12, 14, 16, 14, 12, 16, 18, 4, 6, 4, 8, 12, 8, 2, 6, 10, 6, 17, 3, 8, 5, 3, 2, 4, 6, 8, 12, 10, 12, 14, 12, 10, 14, 16, 2, 4, 2, 6, 10, 6, 0, 4, 8, 4, 15, 7, 6, 6, 6, 7, 8, 10, 12, 13, 6, 8, 4, 4, 6, 6, 7, 9, 14, 11, 13, 15, 12, 10, 16, 18, 7, 7, 4, 8, 12, 6, 6, 2, 10, 4, 16, 4, 6, 2, 2, 4, 4, 5, 7, 12, 9, 11, 13, 10, 8, 14, 16, 5, 5, 2, 6, 10, 4, 4, 0, 8, 2, 7, 11, 9, 7, 5, 4, 2, 2, 4, 5, 12, 6, 8, 8, 8, 6, 5, 4, 6, 5, 6, 8, 7, 6, 8, 10, 12, 6, 10, 4, 4, 7, 10, 10, 2, 12, 10, 10, 4, 6, 6, 6, 4, 3, 2, 4, 3, 4, 6, 5, 4, 6, 8, 10, 4, 8, 2, 2, 5, 8, 8, 0, 10, 17, 6, 6, 6, 7, 8, 10, 12, 13, 15, 4, 10, 6, 6, 7, 7, 9, 11, 16, 13, 15, 17, 14, 12, 18, 19, 6, 8, 4, 10, 14, 8, 6, 4, 12, 2, 18, 2, 8, 4, 4, 5, 5, 7, 9, 14, 11, 13, 15, 12, 10, 16, 17, 4, 6, 2, 8, 12, 6, 4, 2, 10, 0}; 
}