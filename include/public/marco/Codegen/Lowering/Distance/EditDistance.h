#ifndef MARCO_EDIT_DISTANCE_H
#define MARCO_EDIT_DISTANCE_H

#include <string>
#include <array>
#include <cassert>
#include <optional>

namespace marco::codegen::lowering
{
  class EditDistance
  {
    public:
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
      static unsigned int editDistance(const std::string &actual, const std::string &expected);

      // Costs for different edit operations.
      static constexpr unsigned int base_insertion_cost = 2;
      static constexpr unsigned int deletion_cost = 2;
      static constexpr unsigned int transposition_cost = 3;
      static const std::array<std::array<unsigned int, 63>, 63> substitution_costs;
      static const unsigned int largest_substitution_cost;

      // Largest value in an unsigned int.
      static constexpr unsigned int max_cost = 65535;

      // Given a character, calculate its index in the substitution_costs matrix.
      // The only allowed characters are digits, letters and '_'.
      inline static unsigned int charToIndex(const char &character) {
        assert(character >= '0');
        if (character <= '9') {
          return character - '0';
        }
        assert(character >= 'A');
        if (character <= 'Z') {
          return character - 'A' + 10;
        }
        assert(character >= '_');
        if (character == '_') {
          return 36;
        }
        assert(character >= 'a' && character <= 'z');
        return character - 'a' + 37;
      }
  };
}

#endif // MARCO_EDIT_DISTANCE_H