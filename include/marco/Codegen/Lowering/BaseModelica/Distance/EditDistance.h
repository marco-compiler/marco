#ifndef MARCO_CODEGEN_LOWERING_DISTANCE_EDITDISTANCE_H
#define MARCO_CODEGEN_LOWERING_DISTANCE_EDITDISTANCE_H

#include "llvm/ADT/StringRef.h"

/// @file EditDistance.h
/// @brief Header file for the editDistance function.
///
/// The file contains the declaration of the editDistance function, which
/// computes the optimal string alignment distance between two words.

namespace marco::codegen::lowering::bmodelica {
/// @brief Computes the edit distance between actual and expected.
///
/// This function computes and returns the optimal string alignment distance
/// of the two words. Note that the distance is not symmetric. The distance
/// equates to the unlikelihood that a word wrongly written as actual was
/// intended to be the word expected, with a higher distance meaning a lower
/// likelihood.
/// The distance is based on the Damerau–Levenshtein distance, which is
/// similar to the well-known Levenshtein distance, but has a special case
/// for transpositions.
/// Furthermore, the optimal string alignment distance differs from the
/// Damerau–Levenshtein distance in the fact that it does not allow for a
/// substring to be edited more than once, resulting in a higher computational
/// efficiency.
///
/// The distance implements four different edit operations, which refer to
/// possible errors in typing expected:
/// - insertion: an additional character is typed
/// - deletion: a character is removed
/// - substitution: a character is typed instead of another one
/// - transposition: two consecutive characters swap places
/// Each operation has a cost defined in
/// @ref marco::codegen::lowering::EditDistanceParameters.
///
/// The words can only contain digits, letters and '_', which are the only
/// characters allowed for Modelica identifiers.
unsigned int editDistance(llvm::StringRef actual, llvm::StringRef expected);
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_DISTANCE_EDITDISTANCE_H