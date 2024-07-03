#ifndef MARCO_CODEGEN_LOWERING_IDENTIFIERERROR_H
#define MARCO_CODEGEN_LOWERING_IDENTIFIERERROR_H

#include <array>
#include <set>
#include <string>
#include <tuple>

namespace marco::codegen::lowering
{
  class IdentifierError
  {
    public:
      // Types of identifiers that may generate a lowering error.
      enum class IdentifierType {
        VARIABLE,
        FUNCTION,
        TYPE,
        FIELD
      };

      // Create an IdentifierError object, calculating the most similar identifier to identifierName 
      // among those in declaredIdentifiers and the built-in ones (if any).
      IdentifierError(const IdentifierType &errorType, const std::string &identifierName, 
                      const std::set<std::string> &declaredIdentifiers);

      IdentifierType getIdentifierType() const;
      std::string getActual() const;
      std::string getPredicted() const;

    private:
      // Type of identifier that generated the error.
      const IdentifierType identifierType;
      // Actual identifier that generated the error.
      const std::string actualName;
      // Most similar identifier or keyword to the actual one.
      std::string predictedName;

      // Threshold to avoid using semantic distance.
      // This will be multiplied by the length of the actual identifier,
      // in order to account for the higher likelihood of errors in longer
      // identifiers.
      constexpr static unsigned int threshold = 1;

      // List of built-in functions in the language.
      static const std::set<std::string> builtInFunctions;

      // List of built-in types in the language.
      static const std::set<std::string> builtInTypes;

      // Compute the edit distance between actual and all elements of possibleIdentifiers.
      // Return a tuple with the lowest distance and the identifier with
      // said distance.
      std::tuple<unsigned int, std::string> computeLowestEditDistance(
            const std::string &actual, const std::set<std::string> &possibleIdentifiers) const;
      
      // Compute the semantic distance between actual and all elements of possibleIdentifiers.
      // Return a tuple with the highest similarity and the identifier with
      // said similarity.
      std::tuple<float, std::string> computeHighestSemanticSimilarity(
            const std::string &actual, const std::set<std::string> &possibleIdentifiers) const;
  };
}

#endif //MARCO_CODEGEN_LOWERING_IDENTIFIERERROR_H