#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_IDENTIFIERERROR_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_IDENTIFIERERROR_H

#include "llvm/ADT/StringRef.h"
#include <set>
#include <string>
#include <utility>

/// @file IdentifierError.h
/// @brief Header file for the IdentifierError class.
///
/// The IdentifierError class contains the information relative to an error
/// caused by an undefined identifier and is used to compute the identifier
/// that was most likely intended to be written instead.

namespace marco::codegen::lowering::bmodelica {
/// @class marco::codegen::lowering::IdentifierError
/// @brief A class that represents an error due to an undefined identifier.
///
/// This class contains the information relative to an error caused by an
/// undefined identifier.
/// It stores information about which type of identifier caused the error
/// (variable, function, class/type, field), as well as the name of the
/// identifier. It also computes the identifier that is most likely to
/// be the one the user intended to write among the provided ones and
/// the built-in ones.
///
/// The most likely identifier is chosen in two steps:
/// - syntactic distance: the editDistance function is applied to all
///   provided and built-in identifiers, and the identifier with the lowest
///   distance is chosen @ref marco::codegen::lowering::EditDistance.
/// - semantic distance: the getSimilarity method of
///   SentenceDistanceCalculator is applied to the same identifiers,
///   and the identifier with the highest similarity is chosen
///   @ref marco::codegen::lowering::SentenceDistanceCalculator.
///
/// The final identifier is chosen in the following way:
/// - if the identifier chosen in the syntactic step has a distance
///   within a certain threshold, than it is used and the semantic step
///   is skipped. The reason is that syntactic errors are more common than
///   semantic ones.
/// - if instead the distance is above the threshold, and the semantic step
///   provided an identifier with a similarity different from 0, than this
///   identifier is used.
/// - if the distance is above the threshold, but the semantic step resulted
///   in no identifier with a similarity different from 0, than the syntactic
///   result is used. This may happen if the user did not install the wordnet
///   database, or if the words that form an identifier name are not present
///   in the database.
class IdentifierError {
public:
  /// @class marco::codegen::lowering::IdentifierError::IdentifierType
  /// @brief An enumeration of the types of identifier that can cause an
  /// identifier error.
  ///
  /// This class is an enumeration of the different types of identifier that
  /// can cause an IdentifierError.
  /// In particular, TYPE refers to named models, records or base types, and
  /// FIELD refers to fields of models and records.
  enum class IdentifierType { VARIABLE, FUNCTION, TYPE, FIELD };

  /// @brief Creates an IdentifierError object and computed the most similar
  /// identifier to actual.
  ///
  /// The method copies the contents of identifierType and actual into
  /// identifierType and actualName. It the calculates the identifier most
  /// likely  to be the intended one among the ones in declaredIdentifiers
  /// and the built-in ones, storing it into predictedName.
  IdentifierError(const IdentifierType &identifierType, llvm::StringRef actual,
                  const std::set<std::string> &declaredIdentifiers);

  /// @brief Returns the type of identifier that caused the error.
  IdentifierType getIdentifierType() const;

  /// @brief Returns the name of identifier that caused the error.
  std::string getActual() const;

  /// @brief Returns the name of identifier predicted to be the one the user
  /// intended to write.
  std::string getPredicted() const;

private:
  // Type of identifier that generated the error.
  const IdentifierType identifierType;
  // Name of the identifier that generated the error.
  const std::string actualName;
  // Name of the identifier predicted to be the one the user intended.
  std::string predictedName;

  // Threshold per unit length below which the semantic distance step is
  // skipped. The threshold is computed as this value multipled by the
  // number of characters in actualName.
  constexpr static unsigned int threshold = 1;

  // Set of built-in functions in the language.
  static const std::set<std::string> builtInFunctions;

  // Set of built-in types in the language.
  static const std::set<std::string> builtInTypes;

  /// @brief Computes the identifier with the lowest syntactic distance to
  /// actual among the given ones. Returns its name and distance.
  ///
  /// The method computes the edit distances between actual and all elements of
  /// possibleIdentifiers. It returns a pair, with the distance of the closest
  /// identifier as the first element and said identifier as the second element
  /// @ref marco::codegen::lowering::EditDistance.
  std::pair<unsigned int, std::string> computeLowestEditDistance(
      llvm::StringRef actual,
      const std::set<std::string> &possibleIdentifiers) const;

  /// @brief Computes the identifier with the highest semantic similarity to
  /// actual among the given ones. Returns its name and similarity.
  ///
  /// The method computes the semantic similarity between actual and all
  /// elements of possibleIdentifiers. It returns a pair, with the similarity of
  /// the closest identifier as the first element and said identifier as the
  /// second element
  /// @ref marco::codegen::lowering::SentenceDistanceCalculator.
  std::pair<float, std::string> computeHighestSemanticSimilarity(
      llvm::StringRef actual,
      const std::set<std::string> &possibleIdentifiers) const;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_IDENTIFIERERROR_H
