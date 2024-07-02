#include "marco/Codegen/Lowering/IdentifierError.h"
#include "marco/Codegen/Lowering/Distance/EditDistance.h"
#include "marco/Codegen/Lowering/Distance/SentenceDistanceCalculator.h"

namespace marco::codegen::lowering
{
  IdentifierError::IdentifierError(const IdentifierType &identifierType, std::string identifierName, 
        const std::set<std::string> &declaredIdentifiers, unsigned int line, unsigned int column): 
        identifierType(identifierType), actualName(identifierName), predictedName(""),
        line(line), column(column) {
    unsigned int lowestDistance = UINT32_MAX;

    // Choose which built in identifiers to check.
    const std::set<std::string> emptySet = {};

    auto builtInBegin = emptySet.cbegin();
    auto builtInEnd = emptySet.cend();

    switch (identifierType) {
      case IdentifierType::FUNCTION: {
        builtInBegin = builtInFunctions.cbegin();
        builtInEnd = builtInFunctions.cend();
        break;
      }

      case IdentifierType::TYPE: {
        builtInBegin = builtInTypes.cbegin();
        builtInEnd = builtInTypes.cend();
        break;
      }

      default: {}
    }

    // Iterate over all declared or built-in identifiers and compare them with the actual identifier name.
    for (auto pIdent = declaredIdentifiers.cbegin(); pIdent != builtInEnd; ++pIdent) {
      // Start iterating on the built-in identifiers.
      if (pIdent == declaredIdentifiers.cend()) {
        pIdent = builtInBegin;
        if (builtInBegin == builtInEnd) {
          break;
        }
      }
      
      // Calculate the edit distance.
      const std::string declaredId = *pIdent;
      const unsigned int distance = EditDistance::editDistance(actualName, declaredId);

      if (distance < lowestDistance) {
        predictedName = declaredId;
        lowestDistance = distance;
      }
    }

    // If the distance is above the threshold, use semantic distance.
    std::string predictedNameSemantic = "";
    if (lowestDistance >= threshold * identifierName.size()) {
      SentenceDistanceCalculator calculator;
      float highestSimilarity = 0.f;
      
      // Iterate over all declared or built-in identifiers and compare them with the actual identifier name.
      for (auto pIdent = declaredIdentifiers.cbegin(); pIdent != builtInEnd; ++pIdent) {
        // Start iterating on the built-in identifiers.
        if (pIdent == declaredIdentifiers.cend()) {
          pIdent = builtInBegin;
          if (builtInBegin == builtInEnd) {
            break;
          }
        }
        
        // Calculate the semantic distance.
        const std::string declaredId = *pIdent;
        const float similarity = calculator.getSimilarity(actualName, declaredId);

        if (similarity > highestSimilarity) {
          predictedNameSemantic = declaredId;
          highestSimilarity = similarity;
        }
      }

      if (highestSimilarity != 0.f) {
        predictedName = predictedNameSemantic;
      }
    }
  }

  IdentifierError::IdentifierType IdentifierError::getIdentifierType() const {
    return identifierType;
  }

  std::string IdentifierError::getActual() const {
    return actualName;
  }

  std::string IdentifierError::getPredicted() const {
    return predictedName;
  }

  unsigned int IdentifierError::getLine() const {
    return line;
  }

  unsigned int IdentifierError::getColumn() const {
    return column;
  }

  const std::set<std::string> IdentifierError::builtInFunctions = {"abs", "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh", "der", "diagonal", "div", "exp", "fill", "floor", "identity", "integer", "linspace", "log", "log10", "max", "min", "mod", "ndims", "ones", "product", "rem", "sign", "sin", "sinh", "size", "sqrt", "sum", "symmetric", "tan", "tanh", "transpose", "zeros"};

  const std::set<std::string> IdentifierError::builtInTypes = {"Real", "Integer", "String", "Boolean"};
}