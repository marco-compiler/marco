#include "marco/Dialect/Modelica/IR/ConstantFolding.h"
#include "marco/Dialect/Modelica/IR/Attributes.h"
#include "marco/Dialect/Modelica/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cmath>
#include <limits>

using namespace ::mlir::modelica;

namespace mlir::modelica {
std::optional<bool> getBooleanValue(mlir::Attribute attr) {
  if (auto booleanAttr = mlir::dyn_cast_or_null<BooleanAttr>(attr)) {
    return booleanAttr.getValue();
  }

  return std::nullopt;
}

std::optional<int64_t> getIntegerValue(mlir::Attribute attr) {
  if (auto integerAttr = mlir::dyn_cast_or_null<IntegerAttr>(attr)) {
    return integerAttr.getValue();
  }

  return std::nullopt;
}

std::optional<double> getRealValue(mlir::Attribute attr) {
  if (auto integerAttr = mlir::dyn_cast_or_null<IntegerAttr>(attr)) {
    return static_cast<double>(integerAttr.getValue());
  }

  if (auto realAttr = mlir::dyn_cast_or_null<RealAttr>(attr)) {
    return realAttr.getValue().convertToDouble();
  }

  return std::nullopt;
}

bool hasUnit(mlir::Attribute attr) {
  if (auto integerAttr = mlir::dyn_cast_or_null<IntegerAttr>(attr)) {
    return static_cast<bool>(integerAttr.getUnit());
  }

  if (auto realAttr = mlir::dyn_cast_or_null<RealAttr>(attr)) {
    return static_cast<bool>(realAttr.getUnit());
  }

  return false;
}

mlir::Attribute getNumericAttr(mlir::Type type, double value) {
  if (mlir::isa<IntegerType>(type)) {
    double truncated = std::trunc(value);

    if (!std::isfinite(truncated) ||
        truncated <
            static_cast<double>(std::numeric_limits<int64_t>::min()) ||
        truncated >=
            static_cast<double>(std::numeric_limits<int64_t>::max())) {
      return {};
    }

    return IntegerAttr::get(type.getContext(), type,
                            static_cast<int64_t>(truncated),
                            mlir::StringAttr());
  }

  if (mlir::isa<RealType>(type)) {
    if (!std::isfinite(value)) {
      return {};
    }

    return RealAttr::get(type.getContext(), type, llvm::APFloat(value),
                         mlir::StringAttr());
  }

  return {};
}

mlir::Attribute convertScalarAttr(mlir::Attribute attr, mlir::Type type) {
  if (!attr) {
    return {};
  }

  if (mlir::cast<mlir::TypedAttr>(attr).getType() == type) {
    return attr;
  }

  if (auto integerAttr = mlir::dyn_cast<IntegerAttr>(attr);
      integerAttr && mlir::isa<RealType>(type)) {
    return RealAttr::get(
        type.getContext(), type,
        llvm::APFloat(static_cast<double>(integerAttr.getValue())),
        integerAttr.getUnit());
  }

  return {};
}

std::optional<int> compareScalars(mlir::Attribute lhs, mlir::Attribute rhs) {
  if (!lhs || !rhs || hasUnit(lhs) || hasUnit(rhs)) {
    return std::nullopt;
  }

  auto compare = [](auto x, auto y) -> int {
    return x < y ? -1 : (y < x ? 1 : 0);
  };

  if (auto x = getBooleanValue(lhs)) {
    if (auto y = getBooleanValue(rhs)) {
      return compare(*x, *y);
    }

    return std::nullopt;
  }

  if (auto x = mlir::dyn_cast<StringValueAttr>(lhs)) {
    if (auto y = mlir::dyn_cast<StringValueAttr>(rhs)) {
      return compare(x.getValue().compare(y.getValue()), 0);
    }

    return std::nullopt;
  }

  if (auto x = mlir::dyn_cast<EnumLiteralAttr>(lhs)) {
    if (auto y = mlir::dyn_cast<EnumLiteralAttr>(rhs);
        y && x.getType() == y.getType()) {
      return compare(x.getOrdinal(), y.getOrdinal());
    }

    return std::nullopt;
  }

  if (auto x = getIntegerValue(lhs)) {
    if (auto y = getIntegerValue(rhs)) {
      return compare(*x, *y);
    }
  }

  auto x = getRealValue(lhs);
  auto y = getRealValue(rhs);

  if (!x || !y || std::isnan(*x) || std::isnan(*y)) {
    return std::nullopt;
  }

  return compare(*x, *y);
}

std::optional<int64_t> getRangeSize(mlir::Attribute start,
                                    mlir::Attribute stop,
                                    mlir::Attribute step) {
  if (!start || !stop || hasUnit(start) || hasUnit(stop) || hasUnit(step)) {
    return std::nullopt;
  }

  // Boolean and enumeration ranges.
  if (!step && !getRealValue(start)) {
    auto comparison = compareScalars(start, stop);

    if (!comparison) {
      return std::nullopt;
    }

    if (*comparison > 0) {
      return 0;
    }

    if (auto x = mlir::dyn_cast<EnumLiteralAttr>(start)) {
      auto y = mlir::cast<EnumLiteralAttr>(stop);
      return y.getOrdinal() - x.getOrdinal() + 1;
    }

    return *getBooleanValue(stop) - *getBooleanValue(start) + 1;
  }

  // Integer ranges.
  if (auto j = getIntegerValue(start)) {
    auto k = getIntegerValue(stop);
    auto d = step ? getIntegerValue(step) : std::optional<int64_t>(1);

    if (k && d) {
      if (*d == 0) {
        return std::nullopt;
      }

      if ((*d > 0 && *j > *k) || (*d < 0 && *j < *k)) {
        return 0;
      }

      // n = div(k - j, d), with the size being n + 1.
      return (*k - *j) / *d + 1;
    }
  }

  // Real ranges.
  auto j = getRealValue(start);
  auto k = getRealValue(stop);
  auto d = step ? getRealValue(step) : std::optional<double>(1);

  if (!j || !k || !d || *d == 0) {
    return std::nullopt;
  }

  if ((*d > 0 && *j > *k) || (*d < 0 && *j < *k)) {
    return 0;
  }

  // n = floor((k - j) / d), with the size being n + 1.
  double n = std::floor((*k - *j) / *d);

  if (!std::isfinite(n) ||
      n >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }

  return static_cast<int64_t>(n) + 1;
}

llvm::SmallVector<mlir::Attribute> getConstantElements(mlir::Attribute attr) {
  if (auto arrayAttr = mlir::dyn_cast_or_null<ArrayValueAttr>(attr)) {
    return llvm::SmallVector<mlir::Attribute>(arrayAttr.getElements());
  }

  if (attr) {
    return {attr};
  }

  return {};
}

mlir::Attribute foldElementwise(mlir::Type resultType,
                                llvm::ArrayRef<mlir::Attribute> operands,
                                ScalarFoldFn fn) {
  if (llvm::any_of(operands, [](mlir::Attribute attr) { return !attr; })) {
    return {};
  }

  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);

  if (!tensorType) {
    if (llvm::any_of(operands, [](mlir::Attribute attr) {
          return mlir::isa<ArrayValueAttr>(attr);
        })) {
      return {};
    }

    return fn(resultType, operands);
  }

  if (!tensorType.hasStaticShape()) {
    return {};
  }

  int64_t numElements = tensorType.getNumElements();
  llvm::SmallVector<mlir::Attribute> elements;
  llvm::SmallVector<mlir::Attribute> scalars(operands.size());

  for (mlir::Attribute operand : operands) {
    if (auto arrayAttr = mlir::dyn_cast<ArrayValueAttr>(operand);
        arrayAttr &&
        static_cast<int64_t>(arrayAttr.getElements().size()) != numElements) {
      return {};
    }
  }

  for (int64_t i = 0; i < numElements; ++i) {
    for (auto [scalar, operand] : llvm::zip(scalars, operands)) {
      if (auto arrayAttr = mlir::dyn_cast<ArrayValueAttr>(operand)) {
        scalar = arrayAttr.getElements()[i];
      } else {
        scalar = operand;
      }
    }

    mlir::Attribute element = fn(tensorType.getElementType(), scalars);

    if (!element) {
      return {};
    }

    elements.push_back(element);
  }

  return ArrayValueAttr::get(resultType.getContext(), resultType, elements);
}
} // namespace mlir::modelica
