#ifndef MARCO_DIALECT_MODELICA_IR_CONSTANTFOLDING_H
#define MARCO_DIALECT_MODELICA_IR_CONSTANTFOLDING_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace mlir::modelica {
/// Get the value of a Boolean literal.
std::optional<bool> getBooleanValue(mlir::Attribute attr);

/// Get the value of an Integer literal.
std::optional<int64_t> getIntegerValue(mlir::Attribute attr);

/// Get the value of a numeric literal, converting Integer values to Real.
std::optional<double> getRealValue(mlir::Attribute attr);

/// Get whether the attribute is a unitful literal.
bool hasUnit(mlir::Attribute attr);

/// Get a numeric literal of the given type. The value is truncated in case
/// of Integer type. Returns a null attribute if the value can't be
/// represented.
mlir::Attribute getNumericAttr(mlir::Type type, double value);

/// Convert a scalar literal to the given scalar type, applying the implicit
/// conversion of Integer values to Real ones. The unit of the literal, if
/// any, is preserved. Returns a null attribute if the conversion is not
/// possible.
mlir::Attribute convertScalarAttr(mlir::Attribute attr, mlir::Type type);

/// Compare two scalar literals of simple types. Returns a
/// negative value, zero, or a positive value if the first literal is
/// respectively less than, equal to, or greater than the second one. Returns
/// std::nullopt if the literals can't be compared or have units.
std::optional<int> compareScalars(mlir::Attribute lhs, mlir::Attribute rhs);

/// Get the number of elements of the range 'start:step:stop'. The step can be
/// a null attribute, meaning a step of 1.
std::optional<int64_t> getRangeSize(mlir::Attribute start, mlir::Attribute stop,
                                    mlir::Attribute step);

/// Get the elements of a constant, in row-major order. Scalars are seen as
/// arrays with a single element.
llvm::SmallVector<mlir::Attribute> getConstantElements(mlir::Attribute attr);

/// Callback computing the value of a single element.
using ScalarFoldFn = llvm::function_ref<mlir::Attribute(
    mlir::Type elementType, llvm::ArrayRef<mlir::Attribute> operands)>;

/// Fold an element-wise operation. Scalar operands are broadcast to the
/// shape of the array operands, if any. The result type must be either a
/// scalar type or a statically shaped tensor. Returns a null attribute if
/// any operand is not a constant or if the callback fails for any element.
mlir::Attribute foldElementwise(mlir::Type resultType,
                                llvm::ArrayRef<mlir::Attribute> operands,
                                ScalarFoldFn fn);
} // namespace mlir::modelica

#endif // MARCO_DIALECT_MODELICA_IR_CONSTANTFOLDING_H
