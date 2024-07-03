#ifndef MARCO_CODEGEN_LOWERING_DISTANCE_EDITDISTANCE_H
#define MARCO_CODEGEN_LOWERING_DISTANCE_EDITDISTANCE_H

#include <string>

namespace marco::codegen::lowering
{
  // Calculate and return the optimal string alignment distance of the two words. 
  // The distance is based on the Damerau–Levenshtein distance, which is similar 
  // to the well-known Levenshtein distance, but has a special case for transpositions.
  // The optimal string alignment difference differs from the Damerau–Levenshtein
  // distance in the fact that it does not allow for a substring to be edited
  // more than once.
  // This implementation uses custom weights for edit operations. 
  // The distance is not symmetric.
  // The words can only contain digits, letters and '_', which are the characters
  // allowed for Modelica for identifiers.
  unsigned int editDistance(const std::string &actual, const std::string &expected);
}

#endif // MARCO_CODEGEN_LOWERING_DISTANCE_EDITDISTANCE_H