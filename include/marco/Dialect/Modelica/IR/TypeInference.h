#ifndef MARCO_DIALECT_MODELICA_IR_TYPEINFERENCE_H
#define MARCO_DIALECT_MODELICA_IR_TYPEINFERENCE_H

#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::modelica {
/// Get whether the type is one of the scalar types of the dialect.
bool isScalarType(mlir::Type type);

/// Get whether the type is a scalar Integer or Real type.
bool isNumericType(mlir::Type type);

/// Get whether the type is a scalar simple type, that is Boolean, Integer,
/// Real, String or enumeration.
bool isSimpleType(mlir::Type type);

/// Get the shape of a value type. Scalars have an empty shape.
llvm::ArrayRef<int64_t> getShape(mlir::Type type);

/// Get the type of a value with the given shape and element type. An empty
/// shape leads to a scalar type.
mlir::Type getValueType(llvm::ArrayRef<int64_t> shape, mlir::Type elementType);

/// Get the type resulting from the implicit conversion of Integer values to
/// Real ones: Real if any of the two numeric types is
/// Real, Integer otherwise.
mlir::Type getPromotedNumericType(mlir::Type lhs, mlir::Type rhs);

/// Get whether two types are equal, or are both tensors having the same
/// element type and the same rank, and differing only for dimensions that
/// are dynamic in one of the two.
bool areCompatibleTypes(mlir::Type lhs, mlir::Type rhs);

/// Range version of 'areCompatibleTypes'.
bool areCompatibleTypes(mlir::TypeRange lhs, mlir::TypeRange rhs);

/// Get whether a value of type 'source' can be used where a value of type
/// 'destination' is expected, possibly through the implicit conversion of
/// Integer values to Real ones. Objects are considered
/// compatible regardless of their classes, as the verification of the
/// subtyping relation requires the lookup of the classes.
bool isCoercibleTo(mlir::Type source, mlir::Type destination);

/// Get the value of an Integer constant SSA value, if any.
std::optional<int64_t> getConstantIntegerValue(mlir::Value value);

namespace detail {
/// Signature of the type inference functions used by the operations.
#define MODELICA_INFER_FN(name)                                                \
  mlir::LogicalResult name(mlir::MLIRContext *context,                         \
                           std::optional<mlir::Location> location,             \
                           mlir::ValueRange operands,                          \
                           llvm::SmallVectorImpl<mlir::Type> &results)

MODELICA_INFER_FN(inferAdd);
MODELICA_INFER_FN(inferAddEW);
MODELICA_INFER_FN(inferArray);
MODELICA_INFER_FN(inferArrayReduce);
MODELICA_INFER_FN(inferBooleanBinary);
MODELICA_INFER_FN(inferBooleanNullary);
MODELICA_INFER_FN(inferBooleanUnary);
MODELICA_INFER_FN(inferCardinality);
MODELICA_INFER_FN(inferCat);
MODELICA_INFER_FN(inferChange);
MODELICA_INFER_FN(inferConnectionsPredicate);
MODELICA_INFER_FN(inferConnectionsUniqueRootIndices);
MODELICA_INFER_FN(inferCross);
MODELICA_INFER_FN(inferDelay);
MODELICA_INFER_FN(inferDer);
MODELICA_INFER_FN(inferDiagonal);
MODELICA_INFER_FN(inferDiv);
MODELICA_INFER_FN(inferEnumToInteger);
MODELICA_INFER_FN(inferFill);
MODELICA_INFER_FN(inferGetInstanceName);
MODELICA_INFER_FN(inferHomotopy);
MODELICA_INFER_FN(inferIdentity);
MODELICA_INFER_FN(inferIntegerFill);
MODELICA_INFER_FN(inferLinspace);
MODELICA_INFER_FN(inferMatrix);
MODELICA_INFER_FN(inferMinMax);
MODELICA_INFER_FN(inferMul);
MODELICA_INFER_FN(inferNDims);
MODELICA_INFER_FN(inferNthRoot);
MODELICA_INFER_FN(inferNumericBroadcast);
MODELICA_INFER_FN(inferNumericToInteger);
MODELICA_INFER_FN(inferNumericToReal);
MODELICA_INFER_FN(inferNumericUnary);
MODELICA_INFER_FN(inferOuterProduct);
MODELICA_INFER_FN(inferPow);
MODELICA_INFER_FN(inferRange);
MODELICA_INFER_FN(inferRealBroadcast);
MODELICA_INFER_FN(inferRelational);
MODELICA_INFER_FN(inferSameAsOperand);
MODELICA_INFER_FN(inferSample);
MODELICA_INFER_FN(inferScalar);
MODELICA_INFER_FN(inferSemiLinear);
MODELICA_INFER_FN(inferSize);
MODELICA_INFER_FN(inferSkew);
MODELICA_INFER_FN(inferSmooth);
MODELICA_INFER_FN(inferSub);
MODELICA_INFER_FN(inferSymmetric);
MODELICA_INFER_FN(inferTime);
MODELICA_INFER_FN(inferTranspose);
MODELICA_INFER_FN(inferVector);

#undef MODELICA_INFER_FN
} // namespace detail
} // namespace mlir::modelica

#endif // MARCO_DIALECT_MODELICA_IR_TYPEINFERENCE_H
