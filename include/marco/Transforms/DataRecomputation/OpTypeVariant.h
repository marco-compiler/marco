#ifndef MARCO_TRANSFORMS_DATARECOMPUTATION_OPTYPEVARIANT_H
#define MARCO_TRANSFORMS_DATARECOMPUTATION_OPTYPEVARIANT_H

#include "mlir/IR/Operation.h"
#include <variant>

namespace mlir::detail {
struct DRInvalidMarker {};

template <class... OpTys>
bool isAnyOp(mlir::Operation *op) {
  return (mlir::isa<OpTys>(op) || ...);
}

template <class... OpTys>
struct OpTypeList {
  static bool isAnyOp(mlir::Operation *op) {
    return (op != nullptr) && ::mlir::detail::isAnyOp<OpTys...>(op);
  }

  static bool isAnyOp(DRInvalidMarker) { return false; }

  using VariantT = std::variant<OpTys...>;
};

/// Specialization with an "empty" flag value in the beginning
template <class... OpTys>
struct OpTypeList<DRInvalidMarker, OpTys...> {
  static bool isAnyOp(mlir::Operation *op) {
    return (op != nullptr) && ::mlir::detail::isAnyOp<OpTys...>(op);
  }

  static bool isAnyOp(DRInvalidMarker) { return false; }

  using VariantT = std::variant<DRInvalidMarker, OpTys...>;
};

template <class OpTypeListInstance>
struct VariantResolver {

  using VariantT = typename OpTypeListInstance::VariantT;

  template <class... OpTys>
  static VariantT resolveImpl(mlir::Operation *op, OpTypeList<OpTys...>) {
    VariantT result;

    bool success = ([&result](mlir::Operation *op) -> bool {
      if (OpTys resolvedOp = mlir::dyn_cast<OpTys>(op)) {
        result = resolvedOp;
        return true;
      }
      return false;
    }(op) || ...);

    if (!success) {
      llvm_unreachable("No support for op type");
    }

    return result;
  }

  static VariantT resolve(mlir::Operation *op) {
    return resolveImpl(op, OpTypeListInstance{});
  }
};

/// Specialized Variant resolver with an invalid marker
template <class... OpTys>
struct VariantResolver<OpTypeList<DRInvalidMarker, OpTys...>> {
  using VariantT = typename OpTypeList<DRInvalidMarker, OpTys...>::VariantT;

  static VariantT resolve(mlir::Operation *op) {

    VariantT result;

    bool success = ([&result](mlir::Operation *op) -> bool {
      if (OpTys resolvedOp = mlir::dyn_cast<OpTys>(op)) {
        result = resolvedOp;
        return true;
      }
      return false;
    }(op) || ...);

    if (!success) {
      return {DRInvalidMarker{}};
    }
    return result;
  }
};

} // namespace mlir::detail

#endif /* end of include guard:                                                \
          MARCO_TRANSFORMS_DATARECOMPUTATION_OPTYPEVARIANT_H */
