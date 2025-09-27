#include "marco/Codegen/Lowering/BaseModelica/IdentifierError.h"
#include "marco/Codegen/Lowering/BaseModelica/Distance/EditDistance.h"
#include "marco/Codegen/Lowering/BaseModelica/Distance/SentenceDistanceCalculator.h"

namespace marco::codegen::lowering::bmodelica {
IdentifierError::IdentifierError(
    const IdentifierType &identifierType, llvm::StringRef actual,
    const std::set<std::string> &declaredIdentifiers)
    : identifierType(identifierType), actualName(actual), predictedName("") {
  // Check the declared identifiers using edit distance.
  std::pair<unsigned int, std::string> declaredResult =
      computeLowestEditDistance(actual, declaredIdentifiers);
  unsigned int declaredDistance = std::get<0>(declaredResult);
  std::string declaredName = std::get<1>(declaredResult);

  // Check the built-in identifiers using edit distance.
  unsigned int builtInDistance = 65535U;
  std::string builtInName = "";

  switch (identifierType) {
  case IdentifierType::FUNCTION: {
    std::pair<unsigned int, std::string> builtInResult =
        computeLowestEditDistance(actual, builtInFunctions);
    builtInDistance = std::get<0>(builtInResult);
    builtInName = std::get<1>(builtInResult);
    break;
  }

  case IdentifierType::TYPE: {
    std::pair<unsigned int, std::string> builtInResult =
        computeLowestEditDistance(actual, builtInTypes);
    builtInDistance = std::get<0>(builtInResult);
    builtInName = std::get<1>(builtInResult);
    break;
  }

  default: {
  }
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
  if (lowestDistance <= threshold * actualName.size()) {
    return;
  }

  // Check the declared identifiers using semantic distance.
  std::pair<float, std::string> declaredResultSemantic =
      computeHighestSemanticSimilarity(actual, declaredIdentifiers);
  float declaredSimilarity = std::get<0>(declaredResultSemantic);
  declaredName = std::get<1>(declaredResultSemantic);

  // Check the built-in identifiers using semantic distance.
  float builtInSimilarity = 0.f;

  switch (identifierType) {
  case IdentifierType::FUNCTION: {
    std::pair<float, std::string> builtInResult =
        computeHighestSemanticSimilarity(actual, builtInFunctions);
    builtInSimilarity = std::get<0>(builtInResult);
    builtInName = std::get<1>(builtInResult);
    break;
  }

  case IdentifierType::TYPE: {
    std::pair<float, std::string> builtInResult =
        computeHighestSemanticSimilarity(actual, builtInTypes);
    builtInSimilarity = std::get<0>(builtInResult);
    builtInName = std::get<1>(builtInResult);
    break;
  }

  default: {
  }
  }

  // Choose the identifier with the highest similarity, if any.
  // If semantic distance is not supported, the highest similarity will be 0.
  float highestSimilarity;
  std::string predictedNameSemantic;
  if (builtInSimilarity >= declaredSimilarity) {
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

std::string IdentifierError::getActual() const { return actualName; }

std::string IdentifierError::getPredicted() const { return predictedName; }

std::pair<unsigned int, std::string> IdentifierError::computeLowestEditDistance(
    llvm::StringRef actual,
    const std::set<std::string> &possibleIdentifiers) const {
  unsigned int lowestDistanceYet = 65535U;
  std::string predictedName = "";

  for (auto pIdent = possibleIdentifiers.cbegin();
       pIdent != possibleIdentifiers.cend(); ++pIdent) {
    std::string possibleMatch = *pIdent;
    unsigned int distance =
        editDistance(actual, llvm::StringRef(possibleMatch));

    if (distance < lowestDistanceYet) {
      predictedName = possibleMatch;
      lowestDistanceYet = distance;
    }
  }

  return std::pair(lowestDistanceYet, predictedName);
}

std::pair<float, std::string> IdentifierError::computeHighestSemanticSimilarity(
    llvm::StringRef actual,
    const std::set<std::string> &possibleIdentifiers) const {
  SentenceDistanceCalculator calculator;
  float highestSimilarityYet = 0.0f;
  std::string predictedName = "";

  for (auto pIdent = possibleIdentifiers.cbegin();
       pIdent != possibleIdentifiers.cend(); ++pIdent) {
    std::string possibleMatch = *pIdent;
    float similarity =
        calculator.getSimilarity(actual, llvm::StringRef(possibleMatch));

    if (similarity > highestSimilarityYet) {
      predictedName = possibleMatch;
      highestSimilarityYet = similarity;
    }
  }

  return std::pair(highestSimilarityYet, predictedName);
}

const std::set<std::string> IdentifierError::builtInFunctions = {
    "abs",       "acos",    "asin",     "atan",    "atan2",     "ceil", "cos",
    "cosh",      "der",     "diagonal", "div",     "exp",       "fill", "floor",
    "identity",  "integer", "linspace", "log",     "log10",     "max",  "min",
    "mod",       "ndims",   "ones",     "product", "rem",       "sign", "sin",
    "sinh",      "size",    "sqrt",     "sum",     "symmetric", "tan",  "tanh",
    "transpose", "zeros"};

const std::set<std::string> IdentifierError::builtInTypes = {
    "Real", "Integer", "String", "Boolean"};
} // namespace marco::codegen::lowering::bmodelica