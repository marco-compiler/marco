#ifndef MARCO_DIALECT_MODELICA_TRANSFORMS_EVALUATION_H
#define MARCO_DIALECT_MODELICA_TRANSFORMS_EVALUATION_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include <functional>
#include <optional>

namespace mlir::modelica {
/// Evaluator of the expressions of the dialect, computing their values as
/// literals. The values of the operations are obtained by means of their
/// folders, while the values of the leaves, such as component references, can
/// be provided by a hook.
class Evaluator {
public:
  /// Hook computing the value of an SSA value. It returns std::nullopt if the
  /// value is to be computed by the evaluator, and a null attribute if the
  /// value can't be computed.
  using Hook = std::function<std::optional<mlir::Attribute>(mlir::Value)>;

  explicit Evaluator(Hook hook = nullptr);

  /// Set the value of an SSA value, such as the value of an iterator.
  void setValue(mlir::Value value, mlir::Attribute attr);

  /// Compute the value of an SSA value. Returns a null attribute if the
  /// value can't be computed.
  mlir::Attribute evaluate(mlir::Value value);

private:
  mlir::Attribute evaluateResult(mlir::OpResult result);

  /// Evaluate the operations building arrays out of other arrays, whose
  /// result shape may be known only once the operands are known.
  mlir::Attribute evaluateArrayConstruction(mlir::Operation *op);

  /// Evaluate the value yielded by a region, given the values of its
  /// arguments.
  mlir::Attribute evaluateRegion(mlir::Region &region,
                                 llvm::ArrayRef<mlir::Attribute> arguments);

private:
  Hook hook;
  llvm::DenseMap<mlir::Value, mlir::Attribute> values;
  llvm::DenseMap<mlir::Value, mlir::Attribute> cache;
};

/// Get the zero-based position selected by a scalar index of type Integer,
/// Boolean or enumeration.
std::optional<int64_t> getIndexValue(mlir::Attribute index);

/// Apply subscripts to a constant array. Null subscripts
/// represent colons, while vector subscripts select multiple elements.
/// Returns a null attribute if the subscripts are invalid.
mlir::Attribute subscriptConstant(mlir::Attribute array,
                                  llvm::ArrayRef<mlir::Attribute> subscripts);

/// Convert a literal to the given type, applying the implicit conversion of
/// Integer values to Real ones to all its elements. Returns a null attribute
/// if the conversion is not possible.
mlir::Attribute convertConstant(mlir::Attribute attr, mlir::Type type);
} // namespace mlir::modelica

#endif // MARCO_DIALECT_MODELICA_TRANSFORMS_EVALUATION_H
