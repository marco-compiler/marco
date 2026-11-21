#ifndef MARCO_DIALECT_MODELICA_TRANSFORMS_CONNECTIONEXPANSION_ACCESSFUNCTIONUTILS_H
#define MARCO_DIALECT_MODELICA_TRANSFORMS_CONNECTIONEXPANSION_ACCESSFUNCTIONUTILS_H

#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/IndexSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>

namespace mlir::modelica::connection {
/// Result of an affine map of the form 'd + offset', or 'offset' if the
/// dimension is missing.
struct SimpleResult {
  std::optional<unsigned> dimension;
  int64_t offset;
};

using SimpleResults = llvm::SmallVector<SimpleResult, 3>;

/// Get the results of an access function if they are all of the form
/// 'd + offset' or 'offset'.
std::optional<SimpleResults>
getSimpleResults(const marco::modeling::AccessFunction &accessFunction);

/// Apply an affine access function to a point.
marco::modeling::Point
evaluate(const marco::modeling::AccessFunction &accessFunction,
         const marco::modeling::Point &point);

/// Check whether the range has a dimension spanning multiple indices that is
/// used by more than one result.
bool hasRepeatedDimensions(const SimpleResults &results,
                           const marco::modeling::MultidimensionalRange &range);

/// Get the indices accessed by an affine access function when applied to the
/// given ones.
marco::modeling::IndexSet
getImage(const marco::modeling::AccessFunction &accessFunction,
         const marco::modeling::IndexSet &indices);

/// Get the smallest range containing the indices accessed by an affine access
/// function when applied to the given ones.
marco::modeling::MultidimensionalRange
getImageBounds(const marco::modeling::AccessFunction &accessFunction,
               const marco::modeling::MultidimensionalRange &range);

/// Check whether the indices accessed by an affine access function, when
/// applied to the given ones, are among the allowed ones.
bool isAccessWithin(const marco::modeling::AccessFunction &accessFunction,
                    const marco::modeling::IndexSet &indices,
                    const marco::modeling::IndexSet &allowedIndices);

/// Get the indices, among the given ones, that an affine access function maps
/// to the accessed ones.
marco::modeling::IndexSet
getPreimage(const marco::modeling::AccessFunction &accessFunction,
            const marco::modeling::IndexSet &accessedIndices,
            const marco::modeling::IndexSet &indices);

/// Check whether two affine access functions give the same indices when
/// applied to the given ones.
bool agreeOn(const marco::modeling::AccessFunction &lhs,
             const marco::modeling::AccessFunction &rhs,
             const marco::modeling::IndexSet &indices);

/// Check whether an affine access function can be inverted on the given
/// indices by range-wise inverses (see 'getInverseOn'). Any function can be
/// inverted on a single point.
bool isInvertibleOn(const marco::modeling::AccessFunction &accessFunction,
                    const marco::modeling::IndexSet &indices);

/// Get the inverse of an access function on a range, on which it is
/// injective.
std::unique_ptr<marco::modeling::AccessFunction>
getInverseOn(const marco::modeling::AccessFunction &accessFunction,
             const marco::modeling::MultidimensionalRange &range);

/// Get the identity access function of the given rank.
std::unique_ptr<marco::modeling::AccessFunction>
getIdentity(mlir::MLIRContext *context, size_t rank);
} // namespace mlir::modelica::connection

#endif // MARCO_DIALECT_MODELICA_TRANSFORMS_CONNECTIONEXPANSION_ACCESSFUNCTIONUTILS_H
