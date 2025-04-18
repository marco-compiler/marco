#include "marco/Dialect/BaseModelica/Transforms/RuntimeVerifiableOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::ad::forward;

//===---------------------------------------------------------------------===//
// Helper functions
//===---------------------------------------------------------------------===//

static void verifyArgumentIsPositive(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value arg,
                                     bool strictComparison,
                                     const std::string &msg) {
  auto assertOp = builder.create<AssertOp>(loc, builder.getStringAttr(msg),
                                           builder.getI64IntegerAttr(2));

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&assertOp.getConditionRegion());

  mlir::Value zero;
  if (mlir::isa<RealType>(arg.getType()))
    zero = builder.create<ConstantOp>(
        loc, RealAttr::get(builder.getContext(), 0.0f));
  else if (mlir::isa<IntegerType>(arg.getType())) {
    zero = builder.create<ConstantOp>(
        loc, IntegerAttr::get(builder.getContext(), 0));
  } else {
    zero = builder.create<ConstantOp>(loc, builder.getIndexAttr(0));
  }

  mlir::Value condition;
  if (strictComparison)
    condition = builder.create<GtOp>(loc, arg, zero);
  else
    condition = builder.create<GteOp>(loc, arg, zero);

  builder.create<YieldOp>(assertOp.getLoc(), condition);
}

static void verifyArgumentIsNotZero(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::Value arg,
                                    const std::string &msg) {
  auto assertOp = builder.create<AssertOp>(loc, builder.getStringAttr(msg),
                                           builder.getI64IntegerAttr(2));

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&assertOp.getConditionRegion());

  mlir::Value zero;
  bool isIntegerArg = mlir::isa<IntegerType>(arg.getType());
  if (isIntegerArg) {
    zero = builder.create<ConstantOp>(
        loc, IntegerAttr::get(builder.getContext(), 0));
  }

  mlir::Value condition;
  if (isIntegerArg) {
    condition = builder.create<NotEqOp>(loc, arg, zero);
  } else {
    mlir::Value epsilon = builder.create<ConstantOp>(
        loc, RealAttr::get(builder.getContext(), 1E-4));
    mlir::Value rhsAbs =
        builder.create<AbsOp>(loc, RealType::get(builder.getContext()), arg);
    condition = builder.create<GteOp>(loc, rhsAbs, epsilon);
  }

  builder.create<YieldOp>(assertOp.getLoc(), condition);
}

static void verifyArgumentIsBetween(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::Value arg,
                                    double lower, bool isStrictLower,
                                    double upper, bool isStrictUpper,
                                    const std::string &msg) {
  auto assertOp = builder.create<AssertOp>(loc, builder.getStringAttr(msg),
                                           builder.getI64IntegerAttr(2));

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&assertOp.getConditionRegion());

  mlir::Value lowerBound;
  mlir::Value upperBound;
  if (mlir::isa<RealType>(arg.getType())) {
    lowerBound = builder.create<ConstantOp>(
        loc, RealAttr::get(builder.getContext(), lower));
    upperBound = builder.create<ConstantOp>(
        loc, RealAttr::get(builder.getContext(), upper));
  } else if (mlir::isa<IntegerType>(arg.getType())) {
    lowerBound = builder.create<ConstantOp>(
        loc, IntegerAttr::get(builder.getContext(), lower));
    upperBound = builder.create<ConstantOp>(
        loc, IntegerAttr::get(builder.getContext(), upper));
  } else {
    lowerBound =
        builder.create<ConstantOp>(loc, builder.getIndexAttr((int64_t)lower));
    upperBound =
        builder.create<ConstantOp>(loc, builder.getIndexAttr((int64_t)upper));
  }

  mlir::Value cond1;
  if (isStrictLower)
    cond1 = builder.create<GtOp>(loc, arg, lowerBound);
  else
    cond1 = builder.create<GteOp>(loc, arg, lowerBound);

  mlir::Value cond2;
  if (isStrictUpper)
    cond2 = builder.create<LtOp>(loc, arg, upperBound);
  else
    cond2 = builder.create<LteOp>(loc, arg, upperBound);

  mlir::Value condition = builder.create<AndOp>(loc, cond1, cond2);

  builder.create<YieldOp>(assertOp.getLoc(), condition);
}

static void verifyTensorIndexedAccess(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value tensor,
                                      mlir::ValueRange indices, int64_t rank,
                                      const std::string &msg) {

  for (int64_t i = 0; i < rank; i++) {
    auto index = *(indices.begin() + i);

    auto assertOp = builder.create<AssertOp>(loc, builder.getStringAttr(msg),
                                             builder.getI64IntegerAttr(2));

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&assertOp.getConditionRegion());

    mlir::Value zero = builder.create<ConstantOp>(
        loc, IntegerAttr::get(builder.getContext(), 0));

    mlir::Value dimIndex =
        builder.create<ConstantOp>(loc, builder.getIndexAttr(i));
    mlir::Value dim = builder.create<SizeOp>(
        loc, IntegerType::get(builder.getContext()), tensor, dimIndex);

    mlir::Value cond1 = builder.create<GteOp>(loc, index, zero);
    mlir::Value cond2 = builder.create<LtOp>(loc, index, dim);
    mlir::Value condition = builder.create<AndOp>(loc, cond1, cond2);

    builder.create<YieldOp>(assertOp.getLoc(), condition);
  }
}

namespace mlir::bmodelica {

//===---------------------------------------------------------------------===//
// Tensor operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// TensorExtractOp

struct TensorExtractOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<TensorExtractOpRuntimeVerifier, TensorExtractOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                              mlir::OpBuilder &builder,
                                              mlir::Location loc) const {
    auto castedOp = mlir::cast<TensorExtractOp>(op);                                                
    mlir::Value operand = castedOp.getTensor();
    mlir::ValueRange indices = castedOp.getIndices();
    int64_t rank = castedOp.getTensor().getType().getRank();

    verifyTensorIndexedAccess(
        builder, loc, operand, indices, rank,
        "Model error: TensorExtractOp out of bounds access");
  }
};

//===---------------------------------------------------------------------===//
// TensorInsertOp

struct TensorInsertOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<TensorInsertOpRuntimeVerifier, TensorInsertOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                              mlir::OpBuilder &builder,
                                              mlir::Location loc) const {
    auto castedOp = mlir::cast<TensorInsertOp>(op);                                                 
    mlir::Value operand = castedOp.getDestination();
    mlir::ValueRange indices = castedOp.getIndices();
    int64_t rank = castedOp.getDestination().getType().getRank();

    verifyTensorIndexedAccess(builder, loc, operand, indices, rank,
                              "Model error: TensorInsertOp out of bounds access");
  }
};

//===---------------------------------------------------------------------===//
// TensorInsertSliceOp

struct TensorInsertSliceOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<TensorInsertSliceOpRuntimeVerifier, TensorInsertSliceOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                              mlir::OpBuilder &builder,
                                              mlir::Location loc) const {
    auto castedOp = mlir::cast<TensorInsertSliceOp>(op);
    mlir::Value destination = castedOp.getDestination();
    mlir::Value source = castedOp.getValue();

    // verify if the source is a tensor
    if (auto tensorType = mlir::dyn_cast<TensorType>(castedOp.getValue().getType())) {
      int64_t firstOperand = 0;
      for (mlir::Value operand : castedOp.getSubscriptions()) {
        if (mlir::isa<RangeType>(operand.getType())) {
          break;
        }
        firstOperand++;
      }

      int64_t sourceRank = tensorType.getRank();

      // for each dimension of the destination, verify dimension of the source
      for (int64_t i = firstOperand; i < sourceRank + firstOperand; i++) {
        auto assertOp = builder.create<AssertOp>(
            loc,
            builder.getStringAttr("Model error: source array dimension greater "
                                  "than destination array dimension"),
            builder.getI64IntegerAttr(2));

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.createBlock(&assertOp.getConditionRegion());

        mlir::Value dim =
            builder.create<ConstantOp>(loc, builder.getIndexAttr(i));

        mlir::Value dimS = builder.create<ConstantOp>(
            loc, builder.getIndexAttr(i - firstOperand));

        mlir::Value sourceDim = builder.create<SizeOp>(
            loc, IntegerType::get(builder.getContext()), source, dimS);

        mlir::Value destDim = builder.create<SizeOp>(
            loc, IntegerType::get(builder.getContext()), destination, dim);

        mlir::Value cond = builder.create<LteOp>(loc, sourceDim, destDim);

        builder.create<YieldOp>(assertOp.getLoc(), cond);
      }
    }
  }
};

//===---------------------------------------------------------------------===//
// Array operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// DimOp

struct DimOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<DimOpRuntimeVerifier, DimOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                                      mlir::OpBuilder &builder,
                                                      mlir::Location loc) const {
    auto castedOp = mlir::cast<DimOp>(op);
    mlir::Value dim = castedOp.getDimension();
    auto arrayShapedType = mlir::cast<mlir::ShapedType>(castedOp.getArray().getType());
    int64_t rank = arrayShapedType.getRank();

    verifyArgumentIsBetween(builder, loc, dim, 0, false, rank, true,
                            "Model error: dimension index out of bounds");
  }
};

//===---------------------------------------------------------------------===//
// LoadOp

struct LoadOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<LoadOpRuntimeVerifier, LoadOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                              mlir::OpBuilder &builder,
                                              mlir::Location loc) const {
    auto castedOp = mlir::cast<LoadOp>(op);
    mlir::ValueRange indices = castedOp.getIndices();

    // This operation is also used to store scalar variables
    // so check how many indices we have.
    if (indices.size() > 0) {
      // Modelica arrays will eventually be converted to MLIR tensors
      // at some point down the pipeline, so convert everything
      // to tensor to avoid issues
      mlir::Value operand;
      auto operandShapedType = mlir::cast<mlir::ShapedType>(castedOp.getArrayType());
      auto operandTensorType = mlir::RankedTensorType::get(
          operandShapedType.getShape(), operandShapedType.getElementType());
      operand =
          builder.create<ArrayToTensorOp>(loc, operandTensorType, castedOp.getArray());

      auto tensorShapedType = mlir::cast<mlir::ShapedType>(operand.getType());
      int64_t rank = tensorShapedType.getRank();

      verifyTensorIndexedAccess(builder, loc, operand, indices, rank,
                                "Model error: LoadOp out of bounds access");
    }
  }
};

//===---------------------------------------------------------------------===//
// StoreOp

struct StoreOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<StoreOpRuntimeVerifier, StoreOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                              mlir::OpBuilder &builder,
                                              mlir::Location loc) const {
    auto castedOp = mlir::cast<StoreOp>(op);
    mlir::ValueRange indices = castedOp.getIndices();

    // This operation is also used to store scalar variables
    // so check how many indices we have.
    if (indices.size() > 0) {
      // Modelica arrays will eventually be converted to MLIR tensors
      // at some point down the pipeline, so convert everything
      // to tensor to avoid issues
      mlir::Value operand;
      auto operandShapedType = mlir::cast<mlir::ShapedType>(castedOp.getArrayType());
      auto operandTensorType = mlir::RankedTensorType::get(
          operandShapedType.getShape(), operandShapedType.getElementType());
      operand =
          builder.create<ArrayToTensorOp>(loc, operandTensorType, castedOp.getArray());

      auto tensorShapedType = mlir::cast<mlir::ShapedType>(operand.getType());
      int64_t rank = tensorShapedType.getRank();

      verifyTensorIndexedAccess(builder, loc, operand, indices, rank,
                                "Model error: StoreOp out of bounds access");
    }
  }
};

//===---------------------------------------------------------------------===//
// SubscriptionOp

struct SubscriptionOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<SubscriptionOpRuntimeVerifier, SubscriptionOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                              mlir::OpBuilder &builder,
                                              mlir::Location loc) const {
    auto castedOp = mlir::cast<SubscriptionOp>(op);

    // Modelica arrays will eventually be converted to MLIR tensors
    // at some point down the pipeline, so convert everything
    // to tensor to avoid issues
    auto operandShapedType = mlir::cast<mlir::ShapedType>(castedOp.getSourceArrayType());
    auto operandTensorType = mlir::RankedTensorType::get(
        operandShapedType.getShape(), operandShapedType.getElementType());
    auto tensorOperand =
        builder.create<ArrayToTensorOp>(loc, operandTensorType, castedOp.getSource());

    mlir::ValueRange indices = castedOp.getIndices();
    int64_t rank = castedOp.getSourceArrayType().getRank();

    verifyTensorIndexedAccess(builder, loc, tensorOperand, indices, rank,
                              "Model error: SubscriptionOp out of bounds access");
  }
};

//===---------------------------------------------------------------------===//
// Math operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// DivOp

struct DivOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<DivOpRuntimeVerifier, DivOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<DivOp>(op);
    mlir::Value rhs = castedOp.getRhs();
    verifyArgumentIsNotZero(builder, loc, rhs, "Model error: division by zero");
  }
};

//===---------------------------------------------------------------------===//
// DivEWOp

struct DivEWOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<DivEWOpRuntimeVerifier, DivEWOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<DivEWOp>(op);
    mlir::Value rhs = castedOp.getRhs();
    mlir::ShapedType rhsShapedType;
    llvm::SmallVector<mlir::Value> inductionVars;

    bool isRhsScalar = isScalar(rhs.getType());
    bool isIntegerArg;
    if (isRhsScalar) {
      isIntegerArg = mlir::isa<IntegerType>(rhs.getType());
    } else {
      rhsShapedType = mlir::dyn_cast<ShapedType>(rhs.getType());
      isIntegerArg = mlir::isa<IntegerType>(rhsShapedType.getElementType());
    }

    auto assertOp = builder.create<AssertOp>(
        loc, builder.getStringAttr("Model error: element-wise division by zero"),
        builder.getI64IntegerAttr(2));

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&assertOp.getConditionRegion());

    mlir::Value zero;
    if (isIntegerArg)
      zero = builder.create<ConstantOp>(
          loc, IntegerAttr::get(builder.getContext(), 0));
    else
      zero = builder.create<ConstantOp>(
          loc, RealAttr::get(builder.getContext(), 0.0f));

    if (!isRhsScalar) {
      mlir::Value zeroIdx =
          builder.create<ConstantOp>(loc, builder.getIndexAttr(0));
      mlir::Value oneIdx =
          builder.create<ConstantOp>(loc, builder.getIndexAttr(1));

      int64_t rank = rhsShapedType.getRank();
      llvm::SmallVector<mlir::Value, 10> dimSizes;
      for (int64_t i = 0; i < rank; i++) {
        mlir::Value dimIndex =
            builder.create<ConstantOp>(loc, builder.getIndexAttr(i));
        mlir::Value dim = builder.create<SizeOp>(
            loc, IndexType::get(builder.getContext()), rhs, dimIndex);

        dimSizes.emplace_back(dim);
      }

      for (int64_t i = 0; i < rank; i++) {
        auto forOp =
            builder.create<mlir::scf::ForOp>(loc, zeroIdx, dimSizes[i], oneIdx);
        inductionVars.emplace_back(forOp.getInductionVar());

        builder.setInsertionPointToStart(forOp.getBody());
      }
    }

    mlir::Value elemToCheck;
    if (isRhsScalar) {
      elemToCheck = rhs;
    } else {
      elemToCheck = builder.create<TensorExtractOp>(loc, rhs, inductionVars);
    }

    mlir::Value condition;
    if (isIntegerArg) {
      condition = builder.create<NotEqOp>(loc, elemToCheck, zero);
    } else {
      mlir::Value epsilon = builder.create<ConstantOp>(
          loc, RealAttr::get(builder.getContext(), 1E-4));
      mlir::Value elemAbs = builder.create<AbsOp>(
          loc, RealType::get(builder.getContext()), elemToCheck);
      condition = builder.create<GteOp>(loc, elemAbs, epsilon);
    }

    builder.create<YieldOp>(assertOp.getLoc(), condition);
  }
};

//===---------------------------------------------------------------------===//
// Built-in operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// AcosOp

struct AcosOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<AcosOpRuntimeVerifier, AcosOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<AcosOp>(op);
    mlir::Value operand = castedOp.getOperand();

    verifyArgumentIsBetween(builder, loc, operand, -1.0f, false, 1.0f, false,
                            "Model error: Argument of acos outside the domain. "
                            "It should be -1 <= arg <= 1");
  }
};

//===---------------------------------------------------------------------===//
// AsinOp

struct AsinOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<AsinOpRuntimeVerifier, AsinOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<AsinOp>(op);
    mlir::Value operand = castedOp.getOperand();

    verifyArgumentIsBetween(builder, loc, operand, -1.0f, false, 1.0f, false,
                            "Model error: Argument of asin outside the domain. "
                            "It should be -1 <= arg <= 1");
  }
};

//===---------------------------------------------------------------------===//
// DivTruncOp

struct DivTruncOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<DivTruncOpRuntimeVerifier, DivTruncOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<DivTruncOp>(op);
    mlir::Value rhs = castedOp.getY();
    verifyArgumentIsNotZero(builder, loc, rhs,
                            "Model error: integer division by zero");
  }
};

//===---------------------------------------------------------------------===//
// LogOp

struct LogOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<LogOpRuntimeVerifier, LogOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                              mlir::OpBuilder &builder,
                                              mlir::Location loc) const {
    auto castedOp = mlir::cast<LogOp>(op);                                            
    mlir::Value operand = castedOp.getOperand();

    verifyArgumentIsPositive(
        builder, loc, operand, true,
        "Model error: Argument of log outside the domain. It should be > 0");
  }
};

//===---------------------------------------------------------------------===//
// Log10Op

struct Log10OpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<Log10OpRuntimeVerifier, Log10Op> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<Log10Op>(op);
    mlir::Value operand = castedOp.getOperand();

    verifyArgumentIsPositive(
        builder, loc, operand, true,
        "Model error: Argument of log10 outside the domain. It should be > 0");
  }
};

//===---------------------------------------------------------------------===//
// ModOp

struct ModOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<ModOpRuntimeVerifier, ModOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<ModOp>(op);
    mlir::Value rhs = castedOp.getY();

    verifyArgumentIsNotZero(
        builder, loc, rhs,
        "Model error: taking the remainder of a division by 0");
  }
};

//===---------------------------------------------------------------------===//
// RemOp

struct RemOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<RemOpRuntimeVerifier, RemOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<RemOp>(op);
    mlir::Value rhs = castedOp.getY();

    verifyArgumentIsNotZero(
        builder, loc, rhs,
        "Model error: taking the remainder of a division by 0");
  }
};

//===---------------------------------------------------------------------===//
// SizeOp

struct SizeOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<SizeOpRuntimeVerifier, SizeOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<SizeOp>(op);
    if (castedOp.getNumOperands() == 2) {
      mlir::Value dim = castedOp.getDimension();
      auto arrayShapedType = mlir::cast<mlir::ShapedType>(castedOp.getArray().getType());
      int64_t rank = arrayShapedType.getRank();

      verifyArgumentIsBetween(builder, loc, dim, 0, false, rank, true,
                              "Model error: dimension index out of bounds");
    }
  }
};

//===---------------------------------------------------------------------===//
// SqrtOp

struct SqrtOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<SqrtOpRuntimeVerifier, SqrtOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<SqrtOp>(op);
    mlir::Value operand = castedOp.getOperand();
    verifyArgumentIsPositive(
        builder, loc, operand, false,
        "Model error: Argument of sqrt outside the domain. It should be >= 0");
  }
};

//===---------------------------------------------------------------------===//
// TanOp

struct TanOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<TanOpRuntimeVerifier, TanOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<TanOp>(op);
    mlir::Value operand = castedOp.getOperand();

    // Tangent of an integer will never have any problems
    if (mlir::isa<IntegerType>(operand.getType()))
      return;

    auto assertOp = builder.create<AssertOp>(
        loc,
        builder.getStringAttr("Model error: Argument of tan is invalid. It "
                              "should not be a multiple of pi/2"),
        builder.getI64IntegerAttr(2));

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&assertOp.getConditionRegion());

    mlir::Value operandAbs =
        builder.create<AbsOp>(loc, RealType::get(builder.getContext()), operand);

    mlir::Value pi = builder.create<ConstantOp>(
        loc, RealAttr::get(builder.getContext(), M_PI));

    mlir::Value piHalf = builder.create<ConstantOp>(
        loc, RealAttr::get(builder.getContext(), M_PI / 2));

    mlir::Value epsilon = builder.create<ConstantOp>(
        loc, RealAttr::get(builder.getContext(), 1E-4));

    // Multiples of pi are also multiples of pi/2
    // therefore a trivial check as (operand % pi/2)
    // would consider 2pi, 3pi, ... as illegal.
    // Therefore we need to consider illegal only values
    // multiples of pi/2 but NOT of pi.
    // For example:
    // 2pi is multiple of both pi and pi/2 ==> OK
    // 1.5pi is multiple of pi/2 but not of pi ==> illegal

    mlir::Value modPiHalf = builder.create<ModOp>(
        loc, RealType::get(builder.getContext()), operandAbs, piHalf);

    mlir::Value modPi = builder.create<ModOp>(
        loc, RealType::get(builder.getContext()), operandAbs, pi);

    // Remainder is not close to zero
    // (accounts for when the argument is approaching pi from
    // greater values)
    mlir::Value isMulPiHigher = builder.create<LteOp>(loc, modPi, epsilon);

    // Remainder is not close to pi
    // (accounts for when the argument is approaching pi from
    // lower values)
    mlir::Value diff = builder.create<SubOp>(loc, modPi, pi);
    mlir::Value diffAbs =
        builder.create<AbsOp>(loc, RealType::get(builder.getContext()), diff);
    mlir::Value isMulPiLower = builder.create<LteOp>(loc, diffAbs, epsilon);

    mlir::Value isMulPi = builder.create<OrOp>(loc, isMulPiLower, isMulPiHigher);

    // Same thing for pi/2

    mlir::Value isNotMulPiHalfHigher =
        builder.create<GteOp>(loc, modPiHalf, epsilon);

    diff = builder.create<SubOp>(loc, modPiHalf, piHalf);
    diffAbs =
        builder.create<AbsOp>(loc, RealType::get(builder.getContext()), diff);
    mlir::Value isNotMulPiHalfLower =
        builder.create<GteOp>(loc, diffAbs, epsilon);

    mlir::Value isNotMulPiHalf =
        builder.create<AndOp>(loc, isNotMulPiHalfLower, isNotMulPiHalfHigher);

    mlir::Value condition = builder.create<OrOp>(loc, isMulPi, isNotMulPiHalf);

    builder.create<YieldOp>(assertOp.getLoc(), condition);
  }
};
} // namespace

namespace mlir::bmodelica {
void registerRuntimeVerifiableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            BaseModelicaDialect *dialect) {
    AcosOp::attachInterface<::AcosOpRuntimeVerifier>(*context);
    AsinOp::attachInterface<::AsinOpRuntimeVerifier>(*context);
    DimOp::attachInterface<::DimOpRuntimeVerifier>(*context);
    DivEWOp::attachInterface<::DivEWOpRuntimeVerifier>(*context);
    DivOp::attachInterface<::DivOpRuntimeVerifier>(*context);
    DivTruncOp::attachInterface<::DivTruncOpRuntimeVerifier>(*context);
    Log10Op::attachInterface<::Log10OpRuntimeVerifier>(*context);
    LogOp::attachInterface<::LogOpRuntimeVerifier>(*context);
    LoadOp::attachInterface<::LoadOpRuntimeVerifier>(*context);
    ModOp::attachInterface<::ModOpRuntimeVerifier>(*context);
    RemOp::attachInterface<::RemOpRuntimeVerifier>(*context);
    SizeOp::attachInterface<::SizeOpRuntimeVerifier>(*context);
    SqrtOp::attachInterface<::SqrtOpRuntimeVerifier>(*context);
    StoreOp::attachInterface<::StoreOpRuntimeVerifier>(*context);
    SubscriptionOp::attachInterface<::SubscriptionOpRuntimeVerifier>(*context);
    TanOp::attachInterface<::TanOpRuntimeVerifier>(*context);
    TensorExtractOp::attachInterface<::TensorExtractOpRuntimeVerifier>(*context);
    TensorInsertOp::attachInterface<::TensorInsertOpRuntimeVerifier>(*context);
    TensorInsertSliceOp::attachInterface<::TensorInsertSliceOpRuntimeVerifier>(*context);
  });
}
} // namespace mlir::bmodelica
