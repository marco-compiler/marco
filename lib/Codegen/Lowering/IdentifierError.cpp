#include "marco/Codegen/Lowering/IdentifierError.h"
#include "marco/Codegen/Lowering/Distance/EditDistance.h"
#include "marco/Codegen/Lowering/Distance/SentenceDistanceCalculator.h"

namespace marco::codegen::lowering
{
  IdentifierError::IdentifierError(const IdentifierType &identifierType, const std::string &identifierName, 
        const std::set<std::string> &declaredIdentifiers): 
        identifierType(identifierType), actualName(identifierName), predictedName("") {
    // Check the declared identifiers using edit distance.
    std::tuple<unsigned int, std::string> declaredResult = 
        computeLowestEditDistance(actualName, declaredIdentifiers);
    unsigned int declaredDistance = std::get<0>(declaredResult);
    std::string declaredName = std::get<1>(declaredResult);

    // Check the built-in identifiers using edit distance.
    unsigned int builtInDistance = 65535U;
    std::string builtInName = "";

    switch (identifierType) {
      case IdentifierType::FUNCTION: {
        std::tuple<unsigned int, std::string> builtInResult = 
            computeLowestEditDistance(actualName, builtInFunctions);
        builtInDistance = std::get<0>(builtInResult);
        builtInName = std::get<1>(builtInResult);
        break;
      }

      case IdentifierType::TYPE: {
        std::tuple<unsigned int, std::string> builtInResult = 
            computeLowestEditDistance(actualName, builtInTypes);
        builtInDistance = std::get<0>(builtInResult);
        builtInName = std::get<1>(builtInResult);
        break;
      }

      default: {}
    }

    // Choose the identifier with the lowest distance.
    unsigned int lowestDistance;
    if (builtInDistance <= declaredDistance) {
      lowestDistance = builtInDistance;
      predictedName = builtInName;
    } else {
      lowestDistance = declaredDistance;
      predictedName = declaredName;
    }

    // Use semantic distance only if the distance is above the threshold.
    if (lowestDistance <= threshold * identifierName.size()) {
      return;
    }

    // Check the declared identifiers using semantic distance.
    std::tuple<float, std::string> declaredResultSemantic = 
        computeHighestSemanticSimilarity(actualName, declaredIdentifiers);
    float declaredSimilarity = std::get<0>(declaredResultSemantic);
    declaredName = std::get<1>(declaredResultSemantic);

    // Check the built-in identifiers using semantic distance.
    float builtInSimilarity = 65535U;

    switch (identifierType) {
      case IdentifierType::FUNCTION: {
        std::tuple<float, std::string> builtInResult = 
            computeHighestSemanticSimilarity(actualName, builtInFunctions);
        builtInSimilarity = std::get<0>(builtInResult);
        builtInName = std::get<1>(builtInResult);
        break;
      }

      case IdentifierType::TYPE: {
        std::tuple<float, std::string> builtInResult = 
            computeHighestSemanticSimilarity(actualName, builtInTypes);
        builtInSimilarity = std::get<0>(builtInResult);
        builtInName = std::get<1>(builtInResult);
        break;
      }

      default: {}
    }

    // Choose the identifier with the highest similarity, if any.
    // If semantic distance is not supported, the highest similarity will be 0.
    float highestSimilarity;
    std::string predictedNameSemantic;
    if (builtInSimilarity >= builtInSimilarity) {
      highestSimilarity = builtInSimilarity;
      predictedNameSemantic = builtInName;
    } else {
      highestSimilarity = declaredSimilarity;
      predictedNameSemantic = declaredName;
    }

    if (highestSimilarity != 0.0f) {
      predictedName = predictedNameSemantic;
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

  std::tuple<unsigned int, std::string> 
  IdentifierError::computeLowestEditDistance(const std::string &actual, 
                   const std::set<std::string> &possibleIdentifiers) const {
    unsigned int lowestDistanceYet = 65535U;
    std::string predictedName = "";

    for (auto pIdent = possibleIdentifiers.cbegin(); pIdent != possibleIdentifiers.cend(); ++pIdent) {
      const std::string possibleMatch = *pIdent;
      const unsigned int distance = EditDistance::editDistance(actualName, possibleMatch);

      if (distance < lowestDistanceYet) {
        predictedName = possibleMatch;
        lowestDistanceYet = distance;
      }
    }

    return std::tuple(lowestDistanceYet, predictedName);
  }

  std::tuple<float, std::string> 
  IdentifierError::computeHighestSemanticSimilarity(const std::string &actual, 
                   const std::set<std::string> &possibleIdentifiers) const {
    SentenceDistanceCalculator calculator;
    float highestSimilarityYet = 0.0f;
    std::string predictedName = "";

    for (auto pIdent = possibleIdentifiers.cbegin(); pIdent != possibleIdentifiers.cend(); ++pIdent) {
      const std::string possibleMatch = *pIdent;
      const float similarity = calculator.getSimilarity(actualName, possibleMatch);

      if (similarity > highestSimilarityYet) {
        predictedName = possibleMatch;
        highestSimilarityYet = similarity;
      }
    }

    return std::tuple(highestSimilarityYet, predictedName);
  }

  const std::set<std::string> IdentifierError::builtInFunctions = {"abs", "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh", "der", "diagonal", "div", "exp", "fill", "floor", "identity", "integer", "linspace", "log", "log10", "max", "min", "mod", "ndims", "ones", "product", "rem", "sign", "sin", "sinh", "size", "sqrt", "sum", "symmetric", "tan", "tanh", "transpose", "zeros"};

  const std::set<std::string> IdentifierError::builtInTypes = {"Real", "Integer", "String", "Boolean"};
}