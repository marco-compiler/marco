#include "marco/Codegen/Lowering/Distance/EditDistance.h"
#include "marco/Codegen/Lowering/Distance/EditDistanceParameters.h"
#include <string>
#include <vector>

using namespace ::marco;
using namespace ::marco::codegen::lowering;

namespace marco::codegen::lowering {
unsigned int editDistance(llvm::StringRef actual, llvm::StringRef expected) {
  std::string actualName(actual);
  std::string expectedName(expected);
  unsigned int m = actualName.length();
  unsigned int n = expectedName.length();

  constexpr unsigned int maxCost = 65535;

  // Create a matrix to store partial distances for the dynamic programming
  // algorithm.
  std::vector<unsigned int> distances((n + 1) * (m + 1), 0);

  // Initialize the first row of the matrix (all deletions).
  for (unsigned int j = 0; j <= n; ++j) {
    distances[j] = j * editDistanceDeletionCost;
  }

  // Initialize the first column of the matrix (all insertions).
  unsigned int firstCharIdx =
      (n > 0) ? charToSubstitutionCostsIndex(expectedName[0]) : 0;
  for (unsigned int i = 1; i <= m; ++i) {
    unsigned int actualNameCharIdx =
        charToSubstitutionCostsIndex(actualName[i - 1]);
    unsigned int charInsertionCost =
        (n > 0) ? editDistanceSubsitutionCosts[actualNameCharIdx][firstCharIdx]
                : editDistanceLargestSubstitutionCost;
    distances[i * (n + 1)] = distances[(i - 1) * (n + 1)] +
                             editDistanceBaseInsertionCost + charInsertionCost;
  }

  // Compute the distance matrix.
  for (unsigned int i = 1; i <= m; ++i) {
    for (unsigned int j = 1; j <= n; ++j) {
      unsigned int index = i * (n + 1) + j;
      unsigned int actualNameCharIdx =
          charToSubstitutionCostsIndex(actualName[i - 1]);
      unsigned int expectedNameCharIdx =
          charToSubstitutionCostsIndex(expectedName[j - 1]);

      // Calculate the distance in case of deletion.
      unsigned int deletionResult =
          distances[index - 1] + editDistanceDeletionCost;

      // Calculate the distance in case of insertion.
      unsigned int currentCharCost =
          editDistanceSubsitutionCosts[actualNameCharIdx][expectedNameCharIdx];
      unsigned int nextCharCost =
          (j < n) ? editDistanceSubsitutionCosts[actualNameCharIdx]
                                                [charToSubstitutionCostsIndex(
                                                    expectedName[j])]
                  : editDistanceLargestSubstitutionCost;
      unsigned int insertionResult = distances[index - (n + 1)] +
                                     editDistanceBaseInsertionCost +
                                     std::min(currentCharCost, nextCharCost);

      // Calculate the distance in case of substitution.
      // This includes the case when there is a match (0 substitution cost).
      unsigned int substitutionResult =
          distances[index - (n + 1) - 1] +
          editDistanceSubsitutionCosts[actualNameCharIdx][expectedNameCharIdx];

      // Calculate the distance in case of transposition.
      unsigned int transpositionResult =
          (i >= 2 && j >= 2 && actualName[i - 1] == expectedName[j - 2] &&
           actualName[i - 2] == expectedName[j - 1])
              ? distances[index - 2 * (n + 1) - 2] +
                    editDistanceTranspositionCost
              : maxCost;

      // Select the operation with the lowest cost.
      distances[index] =
          std::min(std::min(deletionResult, insertionResult),
                   std::min(substitutionResult, transpositionResult));
    }
  }

  return distances[(n + 1) * (m + 1) - 1];
}
} // namespace marco::codegen::lowering