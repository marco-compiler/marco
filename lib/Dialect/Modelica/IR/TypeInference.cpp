#include "marco/Dialect/Modelica/IR/TypeInference.h"
#include "marco/Dialect/Modelica/IR/Attributes.h"
#include "marco/Dialect/Modelica/IR/ConstantFolding.h"
#include "marco/Dialect/Modelica/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

using namespace ::mlir::modelica;

static constexpr int64_t kDynamic = mlir::ShapedType::kDynamic;

//===---------------------------------------------------------------------===//
// Type utilities
//===---------------------------------------------------------------------===//

namespace mlir::modelica {
bool isScalarType(mlir::Type type) {
  return mlir::isa<BooleanType, IntegerType, RealType, StringType,
                   EnumerationType, ObjectType>(type);
}

bool isNumericType(mlir::Type type) {
  return mlir::isa<IntegerType, RealType>(type);
}

bool isSimpleType(mlir::Type type) {
  return mlir::isa<BooleanType, IntegerType, RealType, StringType,
                   EnumerationType>(type);
}

llvm::ArrayRef<int64_t> getShape(mlir::Type type) {
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    return tensorType.getShape();
  }

  return {};
}

mlir::Type getValueType(llvm::ArrayRef<int64_t> shape,
                        mlir::Type elementType) {
  if (shape.empty()) {
    return elementType;
  }

  return mlir::RankedTensorType::get(shape, elementType);
}

mlir::Type getPromotedNumericType(mlir::Type lhs, mlir::Type rhs) {
  if (mlir::isa<RealType>(lhs)) {
    return lhs;
  }

  if (mlir::isa<RealType>(rhs)) {
    return rhs;
  }

  return lhs;
}

/// Merge two shapes having the same rank, keeping the static dimensions.
static std::optional<llvm::SmallVector<int64_t>>
mergeShapes(llvm::ArrayRef<int64_t> lhs, llvm::ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return std::nullopt;
  }

  llvm::SmallVector<int64_t> result;

  for (auto [lhsDim, rhsDim] : llvm::zip(lhs, rhs)) {
    if (lhsDim == kDynamic) {
      result.push_back(rhsDim);
    } else if (rhsDim == kDynamic || lhsDim == rhsDim) {
      result.push_back(lhsDim);
    } else {
      return std::nullopt;
    }
  }

  return result;
}

bool areCompatibleTypes(mlir::Type lhs, mlir::Type rhs) {
  if (lhs == rhs) {
    return true;
  }

  auto lhsTensor = mlir::dyn_cast<mlir::RankedTensorType>(lhs);
  auto rhsTensor = mlir::dyn_cast<mlir::RankedTensorType>(rhs);

  if (!lhsTensor || !rhsTensor) {
    return false;
  }

  return lhsTensor.getElementType() == rhsTensor.getElementType() &&
         mergeShapes(lhsTensor.getShape(), rhsTensor.getShape());
}

bool areCompatibleTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areCompatibleTypes(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

bool isCoercibleTo(mlir::Type source, mlir::Type destination) {
  if (areCompatibleTypes(source, destination)) {
    return true;
  }

  mlir::Type sourceElementType = mlir::getElementTypeOrSelf(source);
  mlir::Type destinationElementType = mlir::getElementTypeOrSelf(destination);

  // Objects of different classes may be compatible according to the
  // subtyping rules, whose verification requires the lookup
  // of the classes and is thus deferred to instantiation.
  bool areObjects = mlir::isa<ObjectType>(sourceElementType) &&
                    mlir::isa<ObjectType>(destinationElementType);

  if (!areObjects && (!mlir::isa<IntegerType>(sourceElementType) ||
                      !mlir::isa<RealType>(destinationElementType))) {
    return false;
  }

  if (mlir::isa<mlir::RankedTensorType>(source) !=
      mlir::isa<mlir::RankedTensorType>(destination)) {
    return false;
  }

  return mergeShapes(getShape(source), getShape(destination)).has_value();
}

std::optional<int64_t> getConstantIntegerValue(mlir::Value value) {
  mlir::Attribute attr;

  if (!mlir::matchPattern(value, mlir::m_Constant(&attr))) {
    return std::nullopt;
  }

  return getIntegerValue(attr);
}
} // namespace mlir::modelica

//===---------------------------------------------------------------------===//
// Inference helpers
//===---------------------------------------------------------------------===//

namespace {
/// Helper to emit an error and return failure.
template <typename... Args>
mlir::LogicalResult error(std::optional<mlir::Location> location,
                          Args &&...args) {
  return mlir::emitOptionalError(location, std::forward<Args>(args)...);
}

mlir::LogicalResult checkNumOperands(std::optional<mlir::Location> location,
                                     mlir::ValueRange operands,
                                     size_t expected) {
  if (operands.size() != expected) {
    return error(location, "expected ", expected, " operands, but got ",
                 operands.size());
  }

  return mlir::success();
}

bool isScalar(mlir::Type type) { return isScalarType(type); }

bool isTensor(mlir::Type type) {
  return mlir::isa<mlir::RankedTensorType>(type);
}

int64_t getRank(mlir::Type type) {
  return static_cast<int64_t>(getShape(type).size());
}

bool hasNumericElements(mlir::Type type) {
  return isNumericType(mlir::getElementTypeOrSelf(type));
}

bool hasBooleanElements(mlir::Type type) {
  return mlir::isa<BooleanType>(mlir::getElementTypeOrSelf(type));
}

/// Get the size of a dimension as determined by an Integer SSA value, if
/// the value is a constant.
int64_t getDimFromValue(mlir::Value value) {
  if (auto constant = getConstantIntegerValue(value);
      constant && *constant >= 0) {
    return *constant;
  }

  return kDynamic;
}

/// Get whether two dimensions may be equal.
bool areCompatibleDims(int64_t lhs, int64_t rhs) {
  return lhs == kDynamic || rhs == kDynamic || lhs == rhs;
}

/// Get the static dimension among two compatible ones, if any.
int64_t mergeDims(int64_t lhs, int64_t rhs) {
  return lhs == kDynamic ? rhs : lhs;
}

/// Compute the shape resulting from an element-wise operation. If
/// 'allowScalar' is true, scalar operands are broadcast to the shape of the
/// other operand.
mlir::FailureOr<llvm::SmallVector<int64_t>>
getElementwiseShape(std::optional<mlir::Location> location, mlir::Type lhs,
                    mlir::Type rhs, bool allowScalar) {
  if (allowScalar) {
    if (isScalar(lhs)) {
      return llvm::SmallVector<int64_t>(getShape(rhs));
    }

    if (isScalar(rhs)) {
      return llvm::SmallVector<int64_t>(getShape(lhs));
    }
  }

  if (isScalar(lhs) != isScalar(rhs)) {
    return error(location, "operands must be both scalars or both arrays, but "
                           "got ",
                 lhs, " and ", rhs);
  }

  auto shape = mergeShapes(getShape(lhs), getShape(rhs));

  if (!shape) {
    return error(location, "operands have incompatible shapes: ", lhs,
                 " and ", rhs);
  }

  return *shape;
}

/// Compute the shape resulting from an element-wise operation applied to
/// multiple operands, according to the vectorization rules.
mlir::FailureOr<llvm::SmallVector<int64_t>>
getVectorizedShape(std::optional<mlir::Location> location,
                   mlir::TypeRange types) {
  std::optional<llvm::SmallVector<int64_t>> shape;

  for (mlir::Type type : types) {
    if (isScalar(type)) {
      continue;
    }

    if (!shape) {
      shape = llvm::SmallVector<int64_t>(getShape(type));
      continue;
    }

    auto merged = mergeShapes(*shape, getShape(type));

    if (!merged) {
      return error(location, "array operands have incompatible shapes");
    }

    shape = *merged;
  }

  if (!shape) {
    return llvm::SmallVector<int64_t>();
  }

  return *shape;
}

/// Infer the type of a binary element-wise operation between numeric
/// values, with an optional fixed result element type.
mlir::LogicalResult
inferNumericElementwise(std::optional<mlir::Location> location,
                        mlir::ValueRange operands, bool allowScalar,
                        mlir::Type resultElementType,
                        llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  mlir::Type lhs = operands[0].getType();
  mlir::Type rhs = operands[1].getType();

  if (!hasNumericElements(lhs) || !hasNumericElements(rhs)) {
    return error(location, "operands must be numeric, but got ", lhs, " and ",
                 rhs);
  }

  auto shape = getElementwiseShape(location, lhs, rhs, allowScalar);

  if (mlir::failed(shape)) {
    return mlir::failure();
  }

  if (!resultElementType) {
    resultElementType = getPromotedNumericType(mlir::getElementTypeOrSelf(lhs),
                                               mlir::getElementTypeOrSelf(rhs));
  }

  results.push_back(getValueType(*shape, resultElementType));
  return mlir::success();
}

/// Infer the type of a String concatenation.
mlir::LogicalResult
inferStringConcatenation(std::optional<mlir::Location> location,
                         mlir::Type lhs, mlir::Type rhs, bool allowScalar,
                         llvm::SmallVectorImpl<mlir::Type> &results) {
  auto shape = getElementwiseShape(location, lhs, rhs, allowScalar);

  if (mlir::failed(shape)) {
    return mlir::failure();
  }

  results.push_back(getValueType(*shape, StringType::get(lhs.getContext())));
  return mlir::success();
}

bool hasStringElements(mlir::Type type) {
  return mlir::isa<StringType>(mlir::getElementTypeOrSelf(type));
}

/// Infer the type of an operation transforming each numeric element of its
/// only operand into an element of the given type. A null type means that
/// the element type is preserved.
mlir::LogicalResult
inferNumericMap(std::optional<mlir::Location> location,
                mlir::ValueRange operands, mlir::Type resultElementType,
                llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();

  if (!hasNumericElements(type)) {
    return error(location, "operand must be numeric, but got ", type);
  }

  if (!resultElementType) {
    resultElementType = mlir::getElementTypeOrSelf(type);
  }

  results.push_back(getValueType(getShape(type), resultElementType));
  return mlir::success();
}

/// Get the element type resulting from the combination of two types, that is
/// the promoted numeric type if both are numeric, or the type itself if they
/// are equal. Returns a null type if the two types can't be combined.
mlir::Type combineElementTypes(mlir::Type lhs, mlir::Type rhs) {
  if (isNumericType(lhs) && isNumericType(rhs)) {
    return getPromotedNumericType(lhs, rhs);
  }

  if (lhs == rhs) {
    return lhs;
  }

  return {};
}
} // namespace

//===---------------------------------------------------------------------===//
// Inference functions
//===---------------------------------------------------------------------===//

namespace mlir::modelica::detail {
mlir::LogicalResult inferNumericUnary(mlir::MLIRContext *context,
                                      std::optional<mlir::Location> location,
                                      mlir::ValueRange operands,
                                      llvm::SmallVectorImpl<mlir::Type> &results) {
  return inferNumericMap(location, operands, nullptr, results);
}

mlir::LogicalResult
inferNumericToInteger(mlir::MLIRContext *context,
                      std::optional<mlir::Location> location,
                      mlir::ValueRange operands,
                      llvm::SmallVectorImpl<mlir::Type> &results) {
  return inferNumericMap(location, operands, IntegerType::get(context),
                         results);
}

mlir::LogicalResult
inferNumericToReal(mlir::MLIRContext *context,
                   std::optional<mlir::Location> location,
                   mlir::ValueRange operands,
                   llvm::SmallVectorImpl<mlir::Type> &results) {
  return inferNumericMap(location, operands, RealType::get(context), results);
}

mlir::LogicalResult inferAdd(mlir::MLIRContext *context,
                             std::optional<mlir::Location> location,
                             mlir::ValueRange operands,
                             llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  mlir::Type lhs = operands[0].getType();
  mlir::Type rhs = operands[1].getType();

  if (hasStringElements(lhs) && hasStringElements(rhs)) {
    return inferStringConcatenation(location, lhs, rhs, false, results);
  }

  return inferNumericElementwise(location, operands, false, nullptr, results);
}

mlir::LogicalResult inferAddEW(mlir::MLIRContext *context,
                               std::optional<mlir::Location> location,
                               mlir::ValueRange operands,
                               llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  mlir::Type lhs = operands[0].getType();
  mlir::Type rhs = operands[1].getType();

  if (hasStringElements(lhs) && hasStringElements(rhs)) {
    return inferStringConcatenation(location, lhs, rhs, true, results);
  }

  return inferNumericElementwise(location, operands, true, nullptr, results);
}

mlir::LogicalResult inferSub(mlir::MLIRContext *context,
                             std::optional<mlir::Location> location,
                             mlir::ValueRange operands,
                             llvm::SmallVectorImpl<mlir::Type> &results) {
  return inferNumericElementwise(location, operands, false, nullptr, results);
}

mlir::LogicalResult
inferNumericBroadcast(mlir::MLIRContext *context,
                      std::optional<mlir::Location> location,
                      mlir::ValueRange operands,
                      llvm::SmallVectorImpl<mlir::Type> &results) {
  return inferNumericElementwise(location, operands, true, nullptr, results);
}

mlir::LogicalResult
inferRealBroadcast(mlir::MLIRContext *context,
                   std::optional<mlir::Location> location,
                   mlir::ValueRange operands,
                   llvm::SmallVectorImpl<mlir::Type> &results) {
  return inferNumericElementwise(location, operands, true,
                                 RealType::get(context), results);
}

mlir::LogicalResult inferMul(mlir::MLIRContext *context,
                             std::optional<mlir::Location> location,
                             mlir::ValueRange operands,
                             llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  mlir::Type lhs = operands[0].getType();
  mlir::Type rhs = operands[1].getType();

  if (!hasNumericElements(lhs) || !hasNumericElements(rhs)) {
    return error(location, "operands must be numeric, but got ", lhs, " and ",
                 rhs);
  }

  mlir::Type elementType = getPromotedNumericType(
      mlir::getElementTypeOrSelf(lhs), mlir::getElementTypeOrSelf(rhs));

  // Scalar multiplication.
  if (isScalar(lhs) || isScalar(rhs)) {
    auto shape = getElementwiseShape(location, lhs, rhs, true);

    if (mlir::failed(shape)) {
      return mlir::failure();
    }

    results.push_back(getValueType(*shape, elementType));
    return mlir::success();
  }

  // Multiplication of matrices and vectors.
  llvm::ArrayRef<int64_t> lhsShape = getShape(lhs);
  llvm::ArrayRef<int64_t> rhsShape = getShape(rhs);

  if (lhsShape.size() > 2 || rhsShape.size() > 2) {
    return error(location, "multiplication is defined only between scalars, "
                           "vectors and matrices, but got ",
                 lhs, " and ", rhs);
  }

  if (!areCompatibleDims(lhsShape.back(), rhsShape.front())) {
    return error(location, "incompatible dimensions for multiplication: ", lhs,
                 " and ", rhs);
  }

  llvm::SmallVector<int64_t> shape;

  if (lhsShape.size() == 2) {
    shape.push_back(lhsShape[0]);
  }

  if (rhsShape.size() == 2) {
    shape.push_back(rhsShape[1]);
  }

  results.push_back(getValueType(shape, elementType));
  return mlir::success();
}

mlir::LogicalResult inferDiv(mlir::MLIRContext *context,
                             std::optional<mlir::Location> location,
                             mlir::ValueRange operands,
                             llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  mlir::Type lhs = operands[0].getType();
  mlir::Type rhs = operands[1].getType();

  if (!hasNumericElements(lhs) || !isNumericType(rhs)) {
    return error(location, "expected a numeric dividend and a numeric scalar "
                           "divisor, but got ",
                 lhs, " and ", rhs);
  }

  results.push_back(getValueType(getShape(lhs), RealType::get(context)));
  return mlir::success();
}

mlir::LogicalResult inferPow(mlir::MLIRContext *context,
                             std::optional<mlir::Location> location,
                             mlir::ValueRange operands,
                             llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  mlir::Type base = operands[0].getType();
  mlir::Type exponent = operands[1].getType();

  // Scalar exponentiation.
  if (isNumericType(base) && isNumericType(exponent)) {
    results.push_back(RealType::get(context));
    return mlir::success();
  }

  // Exponentiation of square matrices.
  llvm::ArrayRef<int64_t> shape = getShape(base);

  if (hasNumericElements(base) && shape.size() == 2 &&
      areCompatibleDims(shape[0], shape[1]) &&
      mlir::isa<IntegerType>(exponent)) {
    int64_t size = mergeDims(shape[0], shape[1]);

    results.push_back(
        getValueType({size, size}, mlir::getElementTypeOrSelf(base)));

    return mlir::success();
  }

  return error(location, "exponentiation requires numeric scalars, or a "
                         "square matrix and an Integer exponent, but got ",
               base, " and ", exponent);
}

mlir::LogicalResult inferRelational(mlir::MLIRContext *context,
                                    std::optional<mlir::Location> location,
                                    mlir::ValueRange operands,
                                    llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  mlir::Type lhs = operands[0].getType();
  mlir::Type rhs = operands[1].getType();

  if (!isSimpleType(lhs) || !isSimpleType(rhs) ||
      !combineElementTypes(lhs, rhs)) {
    return error(location, "relational operators require scalar operands of "
                           "comparable simple types, but got ",
                 lhs, " and ", rhs);
  }

  results.push_back(BooleanType::get(context));
  return mlir::success();
}

mlir::LogicalResult inferBooleanUnary(mlir::MLIRContext *context,
                                      std::optional<mlir::Location> location,
                                      mlir::ValueRange operands,
                                      llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();

  if (!hasBooleanElements(type)) {
    return error(location, "operand must be Boolean, but got ", type);
  }

  results.push_back(type);
  return mlir::success();
}

mlir::LogicalResult
inferBooleanBinary(mlir::MLIRContext *context,
                   std::optional<mlir::Location> location,
                   mlir::ValueRange operands,
                   llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  mlir::Type lhs = operands[0].getType();
  mlir::Type rhs = operands[1].getType();

  if (!hasBooleanElements(lhs) || !hasBooleanElements(rhs)) {
    return error(location, "operands must be Boolean, but got ", lhs, " and ",
                 rhs);
  }

  auto shape = getElementwiseShape(location, lhs, rhs, false);

  if (mlir::failed(shape)) {
    return mlir::failure();
  }

  results.push_back(getValueType(*shape, BooleanType::get(context)));
  return mlir::success();
}

mlir::LogicalResult
inferBooleanNullary(mlir::MLIRContext *context,
                    std::optional<mlir::Location> location,
                    mlir::ValueRange operands,
                    llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 0))) {
    return mlir::failure();
  }

  results.push_back(BooleanType::get(context));
  return mlir::success();
}

mlir::LogicalResult inferRange(mlir::MLIRContext *context,
                               std::optional<mlir::Location> location,
                               mlir::ValueRange operands,
                               llvm::SmallVectorImpl<mlir::Type> &results) {
  if (operands.size() != 2 && operands.size() != 3) {
    return error(location, "expected 2 or 3 operands, but got ",
                 operands.size());
  }

  mlir::Type elementType;
  bool hasStep = operands.size() == 3;

  for (mlir::Value operand : operands) {
    mlir::Type type = operand.getType();

    if (!elementType) {
      elementType = type;
      continue;
    }

    elementType = combineElementTypes(elementType, type);

    if (!elementType) {
      return error(location, "range bounds and step have incompatible types");
    }
  }

  if (!isNumericType(elementType)) {
    if (!mlir::isa<BooleanType, EnumerationType>(elementType)) {
      return error(location, "range must be of Integer, Real, Boolean or "
                             "enumeration type, but got ",
                   elementType);
    }

    if (hasStep) {
      return error(location, "a step can be specified only for Integer and "
                             "Real ranges");
    }
  }

  // Compute the size of the range, if the operands are constant.
  int64_t size = kDynamic;
  llvm::SmallVector<mlir::Attribute, 3> constants;

  for (mlir::Value operand : operands) {
    mlir::Attribute attr;

    if (mlir::matchPattern(operand, mlir::m_Constant(&attr))) {
      constants.push_back(attr);
    }
  }

  if (constants.size() == operands.size()) {
    if (auto rangeSize = getRangeSize(constants[0], constants[1],
                                      hasStep ? constants[2] : nullptr)) {
      size = *rangeSize;
    }
  }

  results.push_back(mlir::RankedTensorType::get({size}, elementType));
  return mlir::success();
}

mlir::LogicalResult inferArray(mlir::MLIRContext *context,
                               std::optional<mlir::Location> location,
                               mlir::ValueRange operands,
                               llvm::SmallVectorImpl<mlir::Type> &results) {
  if (operands.empty()) {
    return error(location, "array constructor requires at least one element");
  }

  llvm::SmallVector<int64_t> shape(getShape(operands[0].getType()));
  mlir::Type elementType = mlir::getElementTypeOrSelf(operands[0].getType());

  for (mlir::Value operand : operands.drop_front()) {
    mlir::Type type = operand.getType();
    auto mergedShape = mergeShapes(shape, getShape(type));

    if (!mergedShape || isScalar(type) != shape.empty()) {
      return error(location, "array elements must have the same shape");
    }

    shape = *mergedShape;
    elementType =
        combineElementTypes(elementType, mlir::getElementTypeOrSelf(type));

    if (!elementType) {
      return error(location, "array elements have incompatible types");
    }
  }

  shape.insert(shape.begin(), static_cast<int64_t>(operands.size()));
  results.push_back(mlir::RankedTensorType::get(shape, elementType));
  return mlir::success();
}

mlir::LogicalResult inferCat(mlir::MLIRContext *context,
                             std::optional<mlir::Location> location,
                             mlir::ValueRange operands,
                             llvm::SmallVectorImpl<mlir::Type> &results) {
  if (operands.size() < 2) {
    return error(location, "concatenation requires at least one array");
  }

  mlir::ValueRange arrays = operands.drop_front();
  int64_t rank = getRank(arrays[0].getType());

  if (rank == 0) {
    return error(location, "concatenation requires array operands");
  }

  mlir::Type elementType = mlir::getElementTypeOrSelf(arrays[0].getType());

  for (mlir::Value array : arrays.drop_front()) {
    if (getRank(array.getType()) != rank) {
      return error(location, "concatenated arrays must have the same number "
                             "of dimensions");
    }

    elementType = combineElementTypes(
        elementType, mlir::getElementTypeOrSelf(array.getType()));

    if (!elementType) {
      return error(location, "concatenated arrays have incompatible types");
    }
  }

  std::optional<int64_t> dim = getConstantIntegerValue(operands[0]);

  if (!dim) {
    // The concatenated dimension is not known.
    results.push_back(mlir::RankedTensorType::get(
        llvm::SmallVector<int64_t>(rank, kDynamic), elementType));

    return mlir::success();
  }

  if (*dim < 0 || *dim >= rank) {
    return error(location, "concatenation dimension ", *dim,
                 " is out of bounds for arrays of rank ", rank);
  }

  llvm::SmallVector<int64_t> shape(getShape(arrays[0].getType()));

  for (mlir::Value array : arrays.drop_front()) {
    llvm::ArrayRef<int64_t> arrayShape = getShape(array.getType());

    for (int64_t i = 0; i < rank; ++i) {
      if (i == *dim) {
        if (shape[i] == kDynamic || arrayShape[i] == kDynamic) {
          shape[i] = kDynamic;
        } else {
          shape[i] += arrayShape[i];
        }
      } else {
        if (!areCompatibleDims(shape[i], arrayShape[i])) {
          return error(location, "concatenated arrays must have the same "
                                 "sizes along non-concatenated dimensions");
        }

        shape[i] = mergeDims(shape[i], arrayShape[i]);
      }
    }
  }

  results.push_back(mlir::RankedTensorType::get(shape, elementType));
  return mlir::success();
}

mlir::LogicalResult inferIdentity(mlir::MLIRContext *context,
                                  std::optional<mlir::Location> location,
                                  mlir::ValueRange operands,
                                  llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  if (!mlir::isa<IntegerType>(operands[0].getType())) {
    return error(location, "the size must be an Integer scalar");
  }

  int64_t size = getDimFromValue(operands[0]);

  results.push_back(
      mlir::RankedTensorType::get({size, size}, IntegerType::get(context)));

  return mlir::success();
}

mlir::LogicalResult inferDiagonal(mlir::MLIRContext *context,
                                  std::optional<mlir::Location> location,
                                  mlir::ValueRange operands,
                                  llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();

  if (getRank(type) != 1 || !hasNumericElements(type)) {
    return error(location, "operand must be a numeric vector, but got ", type);
  }

  int64_t size = getShape(type)[0];

  results.push_back(mlir::RankedTensorType::get(
      {size, size}, mlir::getElementTypeOrSelf(type)));

  return mlir::success();
}

mlir::LogicalResult
inferIntegerFill(mlir::MLIRContext *context,
                 std::optional<mlir::Location> location,
                 mlir::ValueRange operands,
                 llvm::SmallVectorImpl<mlir::Type> &results) {
  if (operands.empty()) {
    return error(location, "at least one dimension must be specified");
  }

  llvm::SmallVector<int64_t> shape;

  for (mlir::Value operand : operands) {
    if (!mlir::isa<IntegerType>(operand.getType())) {
      return error(location, "dimensions must be Integer scalars");
    }

    shape.push_back(getDimFromValue(operand));
  }

  results.push_back(
      mlir::RankedTensorType::get(shape, IntegerType::get(context)));

  return mlir::success();
}

mlir::LogicalResult inferFill(mlir::MLIRContext *context,
                              std::optional<mlir::Location> location,
                              mlir::ValueRange operands,
                              llvm::SmallVectorImpl<mlir::Type> &results) {
  if (operands.size() < 2) {
    return error(location, "at least one dimension must be specified");
  }

  llvm::SmallVector<int64_t> shape;

  for (mlir::Value operand : operands.drop_front()) {
    if (!mlir::isa<IntegerType>(operand.getType())) {
      return error(location, "dimensions must be Integer scalars");
    }

    shape.push_back(getDimFromValue(operand));
  }

  mlir::Type valueType = operands[0].getType();
  llvm::append_range(shape, getShape(valueType));

  results.push_back(
      mlir::RankedTensorType::get(shape, mlir::getElementTypeOrSelf(valueType)));

  return mlir::success();
}

mlir::LogicalResult inferLinspace(mlir::MLIRContext *context,
                                  std::optional<mlir::Location> location,
                                  mlir::ValueRange operands,
                                  llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 3))) {
    return mlir::failure();
  }

  if (!isNumericType(operands[0].getType()) ||
      !isNumericType(operands[1].getType())) {
    return error(location, "the bounds must be numeric scalars");
  }

  results.push_back(mlir::RankedTensorType::get({getDimFromValue(operands[2])},
                                                RealType::get(context)));

  return mlir::success();
}

mlir::LogicalResult inferNDims(mlir::MLIRContext *context,
                               std::optional<mlir::Location> location,
                               mlir::ValueRange operands,
                               llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  results.push_back(IntegerType::get(context));
  return mlir::success();
}

mlir::LogicalResult inferSize(mlir::MLIRContext *context,
                              std::optional<mlir::Location> location,
                              mlir::ValueRange operands,
                              llvm::SmallVectorImpl<mlir::Type> &results) {
  if (operands.size() != 1 && operands.size() != 2) {
    return error(location, "expected 1 or 2 operands, but got ",
                 operands.size());
  }

  int64_t rank = getRank(operands[0].getType());

  if (operands.size() == 2) {
    if (auto dim = getConstantIntegerValue(operands[1]);
        dim && (*dim < 0 || *dim >= rank)) {
      return error(location, "dimension ", *dim,
                   " is out of bounds for an array of rank ", rank);
    }

    results.push_back(IntegerType::get(context));
    return mlir::success();
  }

  results.push_back(
      mlir::RankedTensorType::get({rank}, IntegerType::get(context)));

  return mlir::success();
}

mlir::LogicalResult inferScalar(mlir::MLIRContext *context,
                                std::optional<mlir::Location> location,
                                mlir::ValueRange operands,
                                llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();

  if (!isTensor(type) || llvm::any_of(getShape(type), [](int64_t dim) {
        return !areCompatibleDims(dim, 1);
      })) {
    return error(location, "operand must be an array with all dimensions of "
                           "size 1, but got ",
                 type);
  }

  results.push_back(mlir::getElementTypeOrSelf(type));
  return mlir::success();
}

mlir::LogicalResult inferVector(mlir::MLIRContext *context,
                                std::optional<mlir::Location> location,
                                mlir::ValueRange operands,
                                llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();
  llvm::ArrayRef<int64_t> shape = getShape(type);

  if (llvm::count_if(shape, [](int64_t dim) { return dim > 1; }) > 1) {
    return error(location, "operand must have at most one dimension with "
                           "size greater than 1, but got ",
                 type);
  }

  int64_t size = 1;

  for (int64_t dim : shape) {
    if (dim == kDynamic) {
      size = kDynamic;
      break;
    }

    size *= dim;
  }

  results.push_back(
      mlir::RankedTensorType::get({size}, mlir::getElementTypeOrSelf(type)));

  return mlir::success();
}

mlir::LogicalResult inferMatrix(mlir::MLIRContext *context,
                                std::optional<mlir::Location> location,
                                mlir::ValueRange operands,
                                llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();
  llvm::SmallVector<int64_t> shape(getShape(type));

  for (size_t i = 2; i < shape.size(); ++i) {
    if (!areCompatibleDims(shape[i], 1)) {
      return error(location, "dimensions after the second one must have "
                             "size 1, but got ",
                   type);
    }
  }

  // Promote to two dimensions.
  shape.resize(std::max<size_t>(shape.size(), 2), 1);
  shape.resize(2);

  results.push_back(
      mlir::RankedTensorType::get(shape, mlir::getElementTypeOrSelf(type)));

  return mlir::success();
}

mlir::LogicalResult inferArrayReduce(mlir::MLIRContext *context,
                                     std::optional<mlir::Location> location,
                                     mlir::ValueRange operands,
                                     llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();

  if (!isTensor(type) || !isSimpleType(mlir::getElementTypeOrSelf(type))) {
    return error(location, "operand must be an array of simple type, but got ",
                 type);
  }

  results.push_back(mlir::getElementTypeOrSelf(type));
  return mlir::success();
}

mlir::LogicalResult inferTranspose(mlir::MLIRContext *context,
                                   std::optional<mlir::Location> location,
                                   mlir::ValueRange operands,
                                   llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();
  llvm::SmallVector<int64_t> shape(getShape(type));

  if (shape.size() < 2) {
    return error(location, "operand must have at least two dimensions, but "
                           "got ",
                 type);
  }

  std::swap(shape[0], shape[1]);

  results.push_back(
      mlir::RankedTensorType::get(shape, mlir::getElementTypeOrSelf(type)));

  return mlir::success();
}

mlir::LogicalResult inferOuterProduct(mlir::MLIRContext *context,
                                      std::optional<mlir::Location> location,
                                      mlir::ValueRange operands,
                                      llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  mlir::Type lhs = operands[0].getType();
  mlir::Type rhs = operands[1].getType();

  if (getRank(lhs) != 1 || getRank(rhs) != 1 || !hasNumericElements(lhs) ||
      !hasNumericElements(rhs)) {
    return error(location, "operands must be numeric vectors, but got ", lhs,
                 " and ", rhs);
  }

  results.push_back(mlir::RankedTensorType::get(
      {getShape(lhs)[0], getShape(rhs)[0]}, RealType::get(context)));

  return mlir::success();
}

mlir::LogicalResult inferSymmetric(mlir::MLIRContext *context,
                                   std::optional<mlir::Location> location,
                                   mlir::ValueRange operands,
                                   llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();
  llvm::ArrayRef<int64_t> shape = getShape(type);

  if (shape.size() != 2 || !areCompatibleDims(shape[0], shape[1]) ||
      !hasNumericElements(type)) {
    return error(location, "operand must be a square numeric matrix, but got ",
                 type);
  }

  int64_t size = mergeDims(shape[0], shape[1]);

  results.push_back(
      mlir::RankedTensorType::get({size, size}, RealType::get(context)));

  return mlir::success();
}

/// Check that the type is a numeric vector of size 3.
static bool isThreeVector(mlir::Type type) {
  return getRank(type) == 1 && areCompatibleDims(getShape(type)[0], 3) &&
         hasNumericElements(type);
}

mlir::LogicalResult inferCross(mlir::MLIRContext *context,
                               std::optional<mlir::Location> location,
                               mlir::ValueRange operands,
                               llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  if (!isThreeVector(operands[0].getType()) ||
      !isThreeVector(operands[1].getType())) {
    return error(location, "operands must be numeric vectors of size 3");
  }

  results.push_back(mlir::RankedTensorType::get({3}, RealType::get(context)));
  return mlir::success();
}

mlir::LogicalResult inferSkew(mlir::MLIRContext *context,
                              std::optional<mlir::Location> location,
                              mlir::ValueRange operands,
                              llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  if (!isThreeVector(operands[0].getType())) {
    return error(location, "operand must be a numeric vector of size 3");
  }

  results.push_back(
      mlir::RankedTensorType::get({3, 3}, RealType::get(context)));

  return mlir::success();
}

mlir::LogicalResult inferNthRoot(mlir::MLIRContext *context,
                                 std::optional<mlir::Location> location,
                                 mlir::ValueRange operands,
                                 llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  return inferNumericMap(location, operands.take_front(),
                         RealType::get(context), results);
}

mlir::LogicalResult
inferEnumToInteger(mlir::MLIRContext *context,
                   std::optional<mlir::Location> location,
                   mlir::ValueRange operands,
                   llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();

  if (!mlir::isa<EnumerationType>(mlir::getElementTypeOrSelf(type))) {
    return error(location, "operand must be an enumeration, but got ", type);
  }

  results.push_back(getValueType(getShape(type), IntegerType::get(context)));
  return mlir::success();
}

mlir::LogicalResult inferDelay(mlir::MLIRContext *context,
                               std::optional<mlir::Location> location,
                               mlir::ValueRange operands,
                               llvm::SmallVectorImpl<mlir::Type> &results) {
  if (operands.size() != 2 && operands.size() != 3) {
    return error(location, "expected 2 or 3 operands, but got ",
                 operands.size());
  }

  for (mlir::Value operand : operands.drop_front()) {
    if (!isNumericType(operand.getType())) {
      return error(location, "delay times must be numeric scalars");
    }
  }

  return inferNumericMap(location, operands.take_front(), nullptr, results);
}

mlir::LogicalResult inferDer(mlir::MLIRContext *context,
                             std::optional<mlir::Location> location,
                             mlir::ValueRange operands,
                             llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();

  if (!mlir::isa<RealType>(mlir::getElementTypeOrSelf(type))) {
    return error(location, "operand must be Real, but got ", type);
  }

  results.push_back(type);
  return mlir::success();
}

mlir::LogicalResult inferHomotopy(mlir::MLIRContext *context,
                                  std::optional<mlir::Location> location,
                                  mlir::ValueRange operands,
                                  llvm::SmallVectorImpl<mlir::Type> &results) {
  return inferNumericElementwise(location, operands, false,
                                 RealType::get(context), results);
}

mlir::LogicalResult inferSemiLinear(mlir::MLIRContext *context,
                                    std::optional<mlir::Location> location,
                                    mlir::ValueRange operands,
                                    llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 3))) {
    return mlir::failure();
  }

  for (mlir::Value operand : operands) {
    if (!hasNumericElements(operand.getType())) {
      return error(location, "operands must be numeric");
    }
  }

  auto shape = getVectorizedShape(location, operands.getTypes());

  if (mlir::failed(shape)) {
    return mlir::failure();
  }

  results.push_back(getValueType(*shape, RealType::get(context)));
  return mlir::success();
}

mlir::LogicalResult inferTime(mlir::MLIRContext *context,
                              std::optional<mlir::Location> location,
                              mlir::ValueRange operands,
                              llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 0))) {
    return mlir::failure();
  }

  results.push_back(RealType::get(context));
  return mlir::success();
}

mlir::LogicalResult
inferSameAsOperand(mlir::MLIRContext *context,
                   std::optional<mlir::Location> location,
                   mlir::ValueRange operands,
                   llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  results.push_back(operands[0].getType());
  return mlir::success();
}

mlir::LogicalResult inferSmooth(mlir::MLIRContext *context,
                                std::optional<mlir::Location> location,
                                mlir::ValueRange operands,
                                llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  results.push_back(operands[1].getType());
  return mlir::success();
}

mlir::LogicalResult inferSample(mlir::MLIRContext *context,
                                std::optional<mlir::Location> location,
                                mlir::ValueRange operands,
                                llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  if (!isNumericType(operands[0].getType()) ||
      !isNumericType(operands[1].getType())) {
    return error(location, "start and interval must be numeric scalars");
  }

  results.push_back(BooleanType::get(context));
  return mlir::success();
}

mlir::LogicalResult inferChange(mlir::MLIRContext *context,
                                std::optional<mlir::Location> location,
                                mlir::ValueRange operands,
                                llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  mlir::Type type = operands[0].getType();

  if (!isSimpleType(mlir::getElementTypeOrSelf(type))) {
    return error(location, "operand must be of simple type, but got ", type);
  }

  results.push_back(getValueType(getShape(type), BooleanType::get(context)));
  return mlir::success();
}

mlir::LogicalResult inferMinMax(mlir::MLIRContext *context,
                                std::optional<mlir::Location> location,
                                mlir::ValueRange operands,
                                llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 2))) {
    return mlir::failure();
  }

  mlir::Type lhs = operands[0].getType();
  mlir::Type rhs = operands[1].getType();

  mlir::Type elementType = combineElementTypes(
      mlir::getElementTypeOrSelf(lhs), mlir::getElementTypeOrSelf(rhs));

  // The restrictions of the reduction expressions apply also to the scalar
  // inputs.
  if (!elementType || !isSimpleType(elementType) ||
      mlir::isa<StringType>(elementType)) {
    return error(location, "operands must be of comparable enumeration, "
                           "Boolean, Integer or Real types, but got ",
                 lhs, " and ", rhs);
  }

  auto shape = getElementwiseShape(location, lhs, rhs, true);

  if (mlir::failed(shape)) {
    return mlir::failure();
  }

  results.push_back(getValueType(*shape, elementType));
  return mlir::success();
}

mlir::LogicalResult inferCardinality(mlir::MLIRContext *context,
                                     std::optional<mlir::Location> location,
                                     mlir::ValueRange operands,
                                     llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  results.push_back(IntegerType::get(context));
  return mlir::success();
}

mlir::LogicalResult
inferGetInstanceName(mlir::MLIRContext *context,
                     std::optional<mlir::Location> location,
                     mlir::ValueRange operands,
                     llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 0))) {
    return mlir::failure();
  }

  results.push_back(StringType::get(context));
  return mlir::success();
}

mlir::LogicalResult
inferConnectionsPredicate(mlir::MLIRContext *context,
                          std::optional<mlir::Location> location,
                          mlir::ValueRange operands,
                          llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 1))) {
    return mlir::failure();
  }

  results.push_back(BooleanType::get(context));
  return mlir::success();
}

mlir::LogicalResult
inferConnectionsUniqueRootIndices(mlir::MLIRContext *context,
                                  std::optional<mlir::Location> location,
                                  mlir::ValueRange operands,
                                  llvm::SmallVectorImpl<mlir::Type> &results) {
  if (mlir::failed(checkNumOperands(location, operands, 3))) {
    return mlir::failure();
  }

  mlir::Type roots = operands[0].getType();
  mlir::Type nodes = operands[1].getType();

  if (getRank(roots) != 1 || getRank(nodes) != 1) {
    return error(location, "roots and nodes must be vectors, but got ", roots,
                 " and ", nodes);
  }

  results.push_back(mlir::RankedTensorType::get({getShape(roots)[0]},
                                                IntegerType::get(context)));

  return mlir::success();
}
} // namespace mlir::modelica::detail
