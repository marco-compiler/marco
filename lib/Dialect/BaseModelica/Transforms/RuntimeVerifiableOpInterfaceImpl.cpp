#include "marco/Dialect/BaseModelica/Transforms/RuntimeVerifiableOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/RuntimeVerifiableOpInterface.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::ad::forward;

//===---------------------------------------------------------------------===//
// Helper functions
//===---------------------------------------------------------------------===//

namespace {
void verifyArgumentIsPositive(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value arg, bool strictComparison,
                              llvm::StringRef message) {
  auto assertOp = builder.create<AssertOp>(loc, builder.getStringAttr(message),
                                           AssertionLevel::Error);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&assertOp.getConditionRegion());

  mlir::Value zero =
      builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));

  mlir::Value condition;

  if (strictComparison) {
    condition = builder.create<GtOp>(loc, arg, zero);
  } else {
    condition = builder.create<GteOp>(loc, arg, zero);
  }

  builder.create<YieldOp>(assertOp.getLoc(), condition);
}

void verifyArgumentIsNotZero(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value arg, llvm::StringRef message) {
  auto assertOp = builder.create<AssertOp>(loc, builder.getStringAttr(message),
                                           AssertionLevel::Error);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&assertOp.getConditionRegion());

  mlir::Value zero =
      builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));

  mlir::Value condition = builder.create<NotEqOp>(loc, arg, zero);
  builder.create<YieldOp>(assertOp.getLoc(), condition);
}

void verifyArgumentIsBetween(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value arg, double lowerBound,
                             bool strictLowerBound, double upperBound,
                             bool strictUpperBound, llvm::StringRef message) {
  auto assertOp = builder.create<AssertOp>(loc, builder.getStringAttr(message),
                                           AssertionLevel::Error);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&assertOp.getConditionRegion());

  mlir::Value lowerBoundValue = builder.create<ConstantOp>(
      loc, RealAttr::get(builder.getContext(), lowerBound));

  mlir::Value lowerBoundCondition;

  if (strictLowerBound) {
    lowerBoundCondition = builder.create<GtOp>(loc, arg, lowerBoundValue);
  } else {
    lowerBoundCondition = builder.create<GteOp>(loc, arg, lowerBoundValue);
  }

  mlir::Value upperBoundValue = builder.create<ConstantOp>(
      loc, RealAttr::get(builder.getContext(), upperBound));

  mlir::Value upperBoundCondition;

  if (strictUpperBound) {
    upperBoundCondition = builder.create<LtOp>(loc, arg, upperBoundValue);
  } else {
    upperBoundCondition = builder.create<LteOp>(loc, arg, upperBoundValue);
  }

  mlir::Value condition =
      builder.create<AndOp>(loc, lowerBoundCondition, upperBoundCondition);

  builder.create<YieldOp>(assertOp.getLoc(), condition);
}

void verifyIndexedAccess(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value shapedValue, mlir::ValueRange indices,
                         llvm::StringRef message) {
  auto shapedType = mlir::cast<mlir::ShapedType>(shapedValue.getType());

  for (int64_t dim = 0, rank = shapedType.getRank(); dim < rank; dim++) {
    mlir::Value index = indices[dim];

    auto assertOp = builder.create<AssertOp>(
        loc, builder.getStringAttr(message), AssertionLevel::Error);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&assertOp.getConditionRegion());

    mlir::Value zero = builder.create<ConstantOp>(loc, builder.getIndexAttr(0));
    mlir::Value lowerBoundCondition = builder.create<GteOp>(loc, index, zero);

    mlir::Value upperBoundCondition;

    if (mlir::isa<mlir::TensorType>(shapedType)) {
      auto dimOp = builder.create<mlir::tensor::DimOp>(loc, shapedValue, dim);
      upperBoundCondition = builder.create<LtOp>(loc, index, dimOp);
    } else if (mlir::isa<ArrayType>(shapedType)) {
      mlir::Value dimIndex =
          builder.create<ConstantOp>(loc, builder.getIndexAttr(dim));

      auto dimOp = builder.create<DimOp>(loc, shapedValue, dimIndex);
      upperBoundCondition = builder.create<LtOp>(loc, index, dimOp);
    }

    mlir::Value condition = lowerBoundCondition;

    if (upperBoundCondition) {
      condition =
          builder.create<AndOp>(loc, lowerBoundCondition, upperBoundCondition);
    }

    builder.create<YieldOp>(assertOp.getLoc(), condition);
  }
}
} // namespace

namespace mlir::bmodelica {
//===---------------------------------------------------------------------===//
// Tensor operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// TensorExtractOp

struct TensorExtractOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<
          TensorExtractOpRuntimeVerifier, TensorExtractOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<TensorExtractOp>(op);

    verifyIndexedAccess(builder, loc, castedOp.getTensor(),
                        castedOp.getIndices(), "index out of bounds");
  }
};

//===---------------------------------------------------------------------===//
// TensorInsertOp

struct TensorInsertOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<
          TensorInsertOpRuntimeVerifier, TensorInsertOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<TensorInsertOp>(op);

    verifyIndexedAccess(builder, loc, castedOp.getDestination(),
                        castedOp.getIndices(), "index out of bounds");
  }
};

//===---------------------------------------------------------------------===//
// TensorInsertSliceOp

struct TensorInsertSliceOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<
          TensorInsertSliceOpRuntimeVerifier, TensorInsertSliceOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<TensorInsertSliceOp>(op);

    mlir::Value source = castedOp.getValue();
    mlir::Value destination = castedOp.getDestination();

    auto destinationTensorType =
        mlir::cast<mlir::TensorType>(destination.getType());

    int64_t destinationRank = destinationTensorType.getRank();
    int64_t destinationDimension = 0;

    auto sourceTensorType = mlir::cast<mlir::TensorType>(source.getType());
    int64_t sourceRank = sourceTensorType.getRank();
    int64_t sourceDimension = 0;

    for (mlir::Value subscript : castedOp.getSubscriptions()) {
      destinationDimension++;

      if (!mlir::isa<RangeType>(subscript.getType())) {
        continue;
      }

      if (subscript.getDefiningOp<UnboundedRangeOp>()) {
        continue;
      }

      auto assertOp = builder.create<AssertOp>(
          loc, builder.getStringAttr("incompatible shapes"),
          AssertionLevel::Error);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.createBlock(&assertOp.getConditionRegion());

      mlir::Value sourceDimSize =
          builder.create<mlir::tensor::DimOp>(loc, source, sourceDimension++);

      auto rangeSize = builder.create<RangeSizeOp>(loc, subscript);

      mlir::Value condition =
          builder.create<EqOp>(loc, sourceDimSize, rangeSize);

      builder.create<YieldOp>(assertOp.getLoc(), condition);
    }

    assert(destinationRank - destinationDimension ==
           sourceRank - sourceDimension);

    while (sourceDimension < sourceRank) {
      auto assertOp = builder.create<AssertOp>(
          loc, builder.getStringAttr("incompatible shapes"),
          AssertionLevel::Error);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.createBlock(&assertOp.getConditionRegion());

      mlir::Value sourceDimSize =
          builder.create<mlir::tensor::DimOp>(loc, source, sourceDimension);

      mlir::Value destinationDimSize = builder.create<mlir::tensor::DimOp>(
          loc, destination, destinationDimension);

      mlir::Value condition =
          builder.create<EqOp>(loc, sourceDimSize, destinationDimSize);

      builder.create<YieldOp>(assertOp.getLoc(), condition);

      ++sourceDimension;
      ++destinationDimension;
    }
  }
};

//===---------------------------------------------------------------------===//
// TensorViewOp

struct TensorViewOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<
          TensorViewOpRuntimeVerifier, TensorViewOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<TensorViewOp>(op);

    for (auto subscript : llvm::enumerate(castedOp.getSubscriptions())) {
      if (mlir::isa<RangeType>(subscript.value().getType())) {
        if (subscript.value().getDefiningOp<UnboundedRangeOp>()) {
          continue;
        }

        auto assertOp = builder.create<AssertOp>(
            loc, builder.getStringAttr("index out of bounds"),
            AssertionLevel::Error);

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.createBlock(&assertOp.getConditionRegion());

        mlir::Value rangeSize =
            builder.create<RangeSizeOp>(loc, subscript.value());

        auto dimOp = builder.create<mlir::tensor::DimOp>(
            loc, castedOp.getSource(), subscript.index());

        mlir::Value condition = builder.create<LteOp>(loc, rangeSize, dimOp);

        builder.create<YieldOp>(assertOp.getLoc(), condition);
      } else {
        auto assertOp = builder.create<AssertOp>(
            loc, builder.getStringAttr("index out of bounds"),
            AssertionLevel::Error);

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.createBlock(&assertOp.getConditionRegion());

        mlir::Value zero =
            builder.create<ConstantOp>(loc, builder.getIndexAttr(0));

        mlir::Value lowerBoundCondition =
            builder.create<GteOp>(loc, subscript.value(), zero);

        mlir::Value upperBoundCondition;

        auto dimOp = builder.create<mlir::tensor::DimOp>(
            loc, castedOp.getSource(), subscript.index());

        upperBoundCondition =
            builder.create<LtOp>(loc, subscript.value(), dimOp);

        mlir::Value condition = lowerBoundCondition;

        if (upperBoundCondition) {
          condition = builder.create<AndOp>(loc, lowerBoundCondition,
                                            upperBoundCondition);
        }

        builder.create<YieldOp>(assertOp.getLoc(), condition);
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
    : public RuntimeVerifiableOpInterface::ExternalModel<DimOpRuntimeVerifier,
                                                         DimOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<DimOp>(op);

    auto assertOp = builder.create<AssertOp>(
        loc,
        builder.getStringAttr(
            "the requested dimension is higher than the rank"),
        AssertionLevel::Error);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&assertOp.getConditionRegion());

    mlir::Value lowerBoundValue =
        builder.create<ConstantOp>(loc, builder.getIndexAttr(0));

    mlir::Value lowerBoundCondition =
        builder.create<GteOp>(loc, castedOp.getDimension(), lowerBoundValue);

    mlir::Value upperBoundValue = builder.create<ConstantOp>(
        loc, builder.getIndexAttr(castedOp.getArray().getType().getRank()));

    mlir::Value upperBoundCondition =
        builder.create<LtOp>(loc, castedOp.getDimension(), upperBoundValue);

    mlir::Value condition =
        builder.create<AndOp>(loc, lowerBoundCondition, upperBoundCondition);

    builder.create<YieldOp>(assertOp.getLoc(), condition);
  }
};

//===---------------------------------------------------------------------===//
// LoadOp

struct LoadOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<LoadOpRuntimeVerifier,
                                                         LoadOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<LoadOp>(op);

    verifyIndexedAccess(builder, loc, castedOp.getArray(),
                        castedOp.getIndices(), "index out of bounds");
  }
};

//===---------------------------------------------------------------------===//
// StoreOp

struct StoreOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<StoreOpRuntimeVerifier,
                                                         StoreOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<StoreOp>(op);

    verifyIndexedAccess(builder, loc, castedOp.getArray(),
                        castedOp.getIndices(), "index out of bounds");
  }
};

//===---------------------------------------------------------------------===//
// SubscriptionOp

struct SubscriptionOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<
          SubscriptionOpRuntimeVerifier, SubscriptionOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<SubscriptionOp>(op);

    verifyIndexedAccess(builder, loc, castedOp.getSource(),
                        castedOp.getIndices(), "index out of bounds");
  }
};

//===---------------------------------------------------------------------===//
// Math operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// DivOp

struct DivOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<DivOpRuntimeVerifier,
                                                         DivOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<DivOp>(op);
    verifyArgumentIsNotZero(builder, loc, castedOp.getRhs(),
                            "division by zero");
  }
};

//===---------------------------------------------------------------------===//
// DivEWOp

struct DivEWOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<DivEWOpRuntimeVerifier,
                                                         DivEWOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<DivEWOp>(op);
    mlir::Value rhs = castedOp.getRhs();

    if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(rhs.getType())) {
      int64_t rank = shapedType.getRank();

      mlir::Value zero =
          builder.create<ConstantOp>(loc, builder.getIndexAttr(0));
      mlir::Value one =
          builder.create<ConstantOp>(loc, builder.getIndexAttr(1));

      llvm::SmallVector<mlir::Value> lowerBounds(rank, zero);
      llvm::SmallVector<mlir::Value> upperBounds;
      llvm::SmallVector<mlir::Value> steps(rank, one);

      for (int64_t dim = 0; dim < rank; ++dim) {
        upperBounds.push_back(
            builder.create<mlir::tensor::DimOp>(loc, castedOp.getRhs(), dim));
      }

      mlir::scf::buildLoopNest(
          builder, loc, lowerBounds, upperBounds, steps,
          [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
              mlir::ValueRange indices) {
            mlir::Value element =
                nestedBuilder.create<TensorExtractOp>(nestedLoc, rhs, indices);

            verifyArgumentIsNotZero(nestedBuilder, nestedLoc, element,
                                    "division by zero");
          });
    } else {
      verifyArgumentIsNotZero(builder, loc, rhs, "division by zero");
    }
  }
};

//===---------------------------------------------------------------------===//
// Built-in operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// AcosOp

struct AcosOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<AcosOpRuntimeVerifier,
                                                         AcosOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<AcosOp>(op);
    mlir::Value operand = castedOp.getOperand();

    verifyArgumentIsBetween(builder, loc, operand, -1.0f, false, 1.0f, false,
                            "acos argument outside of [-1,1]");
  }
};

//===---------------------------------------------------------------------===//
// AsinOp

struct AsinOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<AsinOpRuntimeVerifier,
                                                         AsinOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<AsinOp>(op);
    mlir::Value operand = castedOp.getOperand();

    verifyArgumentIsBetween(builder, loc, operand, -1.0f, false, 1.0f, false,
                            "asin argument outside of [-1,1]");
  }
};

//===---------------------------------------------------------------------===//
// DivTruncOp

struct DivTruncOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<
          DivTruncOpRuntimeVerifier, DivTruncOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<DivTruncOp>(op);
    verifyArgumentIsNotZero(builder, loc, castedOp.getY(), "division by zero");
  }
};

//===---------------------------------------------------------------------===//
// LogOp

struct LogOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<LogOpRuntimeVerifier,
                                                         LogOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<LogOp>(op);
    mlir::Value operand = castedOp.getOperand();

    verifyArgumentIsPositive(builder, loc, operand, true,
                             "log argument outside of (0, +infinite)");
  }
};

//===---------------------------------------------------------------------===//
// Log10Op

struct Log10OpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<Log10OpRuntimeVerifier,
                                                         Log10Op> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<Log10Op>(op);
    mlir::Value operand = castedOp.getOperand();

    verifyArgumentIsPositive(builder, loc, operand, true,
                             "log_10 argument outside of (0, +infinite)");
  }
};

//===---------------------------------------------------------------------===//
// ModOp

struct ModOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<ModOpRuntimeVerifier,
                                                         ModOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<ModOp>(op);
    verifyArgumentIsNotZero(builder, loc, castedOp.getY(), "division by zero");
  }
};

//===---------------------------------------------------------------------===//
// RemOp

struct RemOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<RemOpRuntimeVerifier,
                                                         RemOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<RemOp>(op);
    verifyArgumentIsNotZero(builder, loc, castedOp.getY(), "division by zero");
  }
};

//===---------------------------------------------------------------------===//
// SizeOp

struct SizeOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<SizeOpRuntimeVerifier,
                                                         SizeOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<SizeOp>(op);

    if (castedOp.hasDimension()) {
      mlir::Value dimension = castedOp.getDimension();

      auto assertOp = builder.create<AssertOp>(
          loc, builder.getStringAttr("dimension index out of bounds"),
          AssertionLevel::Error);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.createBlock(&assertOp.getConditionRegion());

      mlir::Value lowerBoundValue =
          builder.create<ConstantOp>(loc, builder.getIndexAttr(0));

      mlir::Value lowerBoundCondition =
          builder.create<GteOp>(loc, dimension, lowerBoundValue);

      auto shapedType =
          mlir::cast<mlir::ShapedType>(castedOp.getArray().getType());

      mlir::Value upperBoundValue = builder.create<ConstantOp>(
          loc, builder.getIndexAttr(shapedType.getRank()));

      mlir::Value upperBoundCondition =
          builder.create<LtOp>(loc, dimension, upperBoundValue);

      mlir::Value condition =
          builder.create<AndOp>(loc, lowerBoundCondition, upperBoundCondition);

      builder.create<YieldOp>(assertOp.getLoc(), condition);
    }
  }
};

//===---------------------------------------------------------------------===//
// SqrtOp

struct SqrtOpRuntimeVerifier
    : public RuntimeVerifiableOpInterface::ExternalModel<SqrtOpRuntimeVerifier,
                                                         SqrtOp> {
  void generateRuntimeVerification(mlir::Operation *op,
                                   mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
    auto castedOp = mlir::cast<SqrtOp>(op);

    verifyArgumentIsPositive(builder, loc, castedOp.getOperand(), false,
                             "sqrt argument is less than zero");
  }
};
} // namespace mlir::bmodelica

namespace mlir::bmodelica {
void registerRuntimeVerifiableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            BaseModelicaDialect *dialect) {
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    context->getOrLoadDialect<mlir::scf::SCFDialect>();
    context->getOrLoadDialect<mlir::tensor::TensorDialect>();

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

    TensorExtractOp::attachInterface<::TensorExtractOpRuntimeVerifier>(
        *context);

    TensorInsertOp::attachInterface<::TensorInsertOpRuntimeVerifier>(*context);

    TensorInsertSliceOp::attachInterface<::TensorInsertSliceOpRuntimeVerifier>(
        *context);

    TensorViewOp::attachInterface<::TensorViewOpRuntimeVerifier>(*context);
  });
}
} // namespace mlir::bmodelica