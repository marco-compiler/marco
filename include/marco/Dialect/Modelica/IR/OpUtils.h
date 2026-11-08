#ifndef MARCO_DIALECT_MODELICA_IR_OPUTILS_H
#define MARCO_DIALECT_MODELICA_IR_OPUTILS_H

#include "marco/Dialect/Modelica/IR/Ops.h"
#include "mlir/IR/OpImplementation.h"

// Utilities shared by the implementations of the operations.

namespace mlir::modelica::detail {
/// Get the values yielded by a region terminated by a 'modelica.yield', and
/// check that their number is within the given bounds. An error mentioning
/// the given description of the region is emitted on failure.
mlir::FailureOr<mlir::ValueRange> getYieldedValues(mlir::Operation *op,
                                                   mlir::Region &region,
                                                   llvm::StringRef regionName,
                                                   size_t minValues,
                                                   size_t maxValues);

/// Verify that a region yields a single Boolean scalar. If 'allowVector' is
/// true, a vector of Boolean values is also accepted.
mlir::LogicalResult verifyCondition(mlir::Operation *op, mlir::Region &region,
                                    bool allowVector = false);

/// Verify that a value is a component reference, emitting an error
/// mentioning the given description of the value otherwise.
mlir::LogicalResult verifyComponentReference(mlir::Operation *op,
                                             mlir::Value value,
                                             llvm::StringRef description,
                                             bool allowDerivative = false);

/// Print the attributes of an operation, including the inherent ones stored
/// as properties, except for the elided ones.
void printAttributes(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                     llvm::ArrayRef<llvm::StringRef> elidedAttrs,
                     bool withKeyword = true);

/// Parse an optional region, keeping it empty if not present.
mlir::ParseResult parseOptionalRegion(mlir::OpAsmParser &parser,
                                      mlir::Region &region);

/// Print a region, if not empty, preceded by the given keyword.
void printOptionalRegion(mlir::OpAsmPrinter &printer, mlir::Region &region,
                         llvm::StringRef keyword = {});

/// Verify the content of a modification region, which can contain at most
/// one value.
mlir::LogicalResult verifyModificationRegion(mlir::Operation *op,
                                             mlir::Region &region);

/// Parse a region, adding an empty block to it if the region is empty.
mlir::ParseResult parseRegionWithBlock(mlir::OpAsmParser &parser,
                                       mlir::Region &region);
} // namespace mlir::modelica::detail

#endif // MARCO_DIALECT_MODELICA_IR_OPUTILS_H
