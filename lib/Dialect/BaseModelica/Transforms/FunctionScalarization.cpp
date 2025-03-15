#include "marco/Dialect/BaseModelica/Transforms/FunctionScalarization.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_FUNCTIONSCALARIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

static unsigned int
getVectorizationRank(VectorizableOpInterface op,
                     mlir::SymbolTableCollection &symbolTableCollection) {
  // Only functions with exactly one result can be vectorized, as per
  // Modelica language specification.

  if (op->getNumResults() != 1) {
    return 0;
  }

  // List of the additional dimensions with respect to the expected scalar
  // arguments.
  llvm::SmallVector<int64_t, 3> dimensions;
  auto args = op.getArgs();

  if (args.empty()) {
    return 0;
  }

  for (const auto &arg : llvm::enumerate(args)) {
    mlir::Type argType = arg.value().getType();

    unsigned int argExpectedRank =
        op.getArgExpectedRank(arg.index(), symbolTableCollection);

    unsigned int argRank = 0;

    if (auto argShapedType = mlir::dyn_cast<mlir::ShapedType>(argType)) {
      argRank = argShapedType.getRank();
    }

    // Each argument must have a rank higher than the expected one
    // for the operation to be vectorized.

    if (argRank <= argExpectedRank) {
      return 0;
    }

    if (arg.index() == 0) {
      // If this is the first argument, then it will determine the
      // rank and dimensions of the result array, although the
      // dimensions can be also specialized by the other arguments
      // if initially unknown.

      for (unsigned int i = 0; i < argRank - argExpectedRank; ++i) {
        auto argShapedType = mlir::cast<mlir::ShapedType>(argType);
        auto dimension = argShapedType.getDimSize(arg.index());
        dimensions.push_back(dimension);
      }
    } else {
      // The rank difference must match with the one given by the first
      // argument, independently of the dimensions sizes.

      if (argRank != argExpectedRank + dimensions.size()) {
        return 0;
      }

      for (size_t i = 0; i < argRank - argExpectedRank; ++i) {
        auto argShapedType = mlir::cast<mlir::ShapedType>(argType);
        auto dimensionSize = argShapedType.getDimSize(i);

        if (dimensionSize == mlir::ShapedType::kDynamic) {
          // If the dimension is dynamic, then no further checks or
          // specializations are possible.
          continue;
        }

        // If the dimension determined by the first argument is fixed,
        // then also the dimension of the other arguments must match
        // (when that's fixed too).

        if (dimensions[i] != mlir::ShapedType::kDynamic &&
            dimensions[i] != dimensionSize) {
          return 0;
        }

        // If the dimension determined by the first argument is dynamic, then
        // set it to a required size.

        if (dimensions[i] == mlir::ShapedType::kDynamic) {
          dimensions[i] = dimensionSize;
        }
      }
    }
  }

  return dimensions.size();
}

/*
static mlir::LogicalResult scalarizeVectorizableOp(
    mlir::RewriterBase& rewriter,
    VectorizableOpInterface op,
    unsigned int vectorizationRank,
    bool assertions)
{
  mlir::Location loc = op.getLoc();
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  auto args = op.getArgs();

  // Allocate the result arrays. In theory, a vectorizable operation has one
  // and only one result as per the Modelica language specification.
  // However, in order to be resilient to specification changes, the
  // scalarization process has been implemented also to consider multiple
  // results. It must be noted that this extension has no performance
  // drawbacks, as it will just stop after processing the first and only
  // expected result.

  llvm::SmallVector<mlir::Value, 3> results;

  for (const auto& resultType : op->getResultTypes()) {
    assert(resultType.isa<ArrayType>());
    llvm::SmallVector<int64_t, 3> shape;
    llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

    for (const auto& dimension :
llvm::enumerate(resultType.cast<ArrayType>().getShape())) {
      shape.push_back(dimension.value());

      if (dimension.value() == -1) {
        // Get the actual size from the first operand. Others should have
        // the same size by construction.

        mlir::Value index = builder.create<mlir::arith::ConstantOp>(loc,
builder.getIndexAttr(dimension.index()));
        dynamicDimensions.push_back(builder.create<DimOp>(loc, args[0], index));
      }
    }

    auto arrayType = ArrayType::get(
        shape,
        resultType.cast<ArrayType>().getElementType());

    mlir::Value array =  builder.create<AllocOp>(loc, arrayType,
dynamicDimensions); results.push_back(array);
  }

  // If runtime assertions are enabled, check if the dimensions match

  if (assertions) {
    llvm::SmallVector<mlir::Value, 3> indexes;

    for (size_t i = 0; i < vectorizationRank; ++i) {
      builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(i));
    }

    for (mlir::Value index : indexes) {
      llvm::SmallVector<mlir::Value, 3> dimensions;

      for (mlir::Value arg : args) {
        dimensions.push_back(builder.create<DimOp>(loc, arg, index));
      }

      for (size_t i = 1; i < dimensions.size(); ++i) {
        mlir::Value condition = builder.create<mlir::arith::CmpIOp>(loc,
mlir::arith::CmpIPredicate::eq, dimensions[0], dimensions[i]);
        builder.create<mlir::cf::AssertOp>(loc, condition, "Incompatible
dimensions for vectorized function arguments");
      }
    }
  }

  mlir::Value zero = builder.create<mlir::arith::ConstantOp>(op->getLoc(),
builder.getIndexAttr(0)); mlir::Value one =
builder.create<mlir::arith::ConstantOp>(op->getLoc(), builder.getIndexAttr(1));
  llvm::SmallVector<mlir::Value, 3> indexes;

  for (unsigned int i = 0; i < vectorizationRank; ++i) {
    mlir::Value index = builder.create<ConstantOp>(op->getLoc(),
builder.getIndexAttr(i)); mlir::Value dimension =
builder.create<DimOp>(op->getLoc(), op.getArgs()[0], index);

    // The scf.parallel operation supports multiple indexes, but
    // unfortunately the OpenMP -> LLVM-IR doesn't do it yet. For now,
    // let's keep the parallelization only on one index, and especially
    // on the outer one so that we can take advantage of data locality.

    if (i == 0) {
      auto parallelOp = builder.create<mlir::scf::ParallelOp>(loc, zero,
dimension, one); indexes.push_back(parallelOp.getInductionVars()[0]);
      builder.setInsertionPointToStart(parallelOp.getBody());
    } else {
      auto forOp = builder.create<mlir::scf::ForOp>(loc, zero, dimension, one);
      indexes.push_back(forOp.getInductionVar());
      builder.setInsertionPointToStart(forOp.getBody());
    }
  }

  llvm::SmallVector<mlir::Value> scalarizedResults;

  if (mlir::failed(op.scalarize(builder, indexes, scalarizedResults))) {
    return mlir::failure();
  }

  // Copy the (not necessarily) scalar results into the result arrays
  for (auto result : llvm::enumerate(scalarizedResults)) {
    mlir::Value subscript = builder.create<SubscriptionOp>(loc,
results[result.index()], indexes); builder.create<AssignmentOp>(loc, subscript,
result.value());
  }

  // Replace the original operation with the newly allocated arrays
  op->replaceAllUsesWith(results);
  op->erase();

  return mlir::success();
}
*/

namespace {
class FunctionScalarizationPass
    : public impl::FunctionScalarizationPassBase<FunctionScalarizationPass> {
public:
  using FunctionScalarizationPassBase<
      FunctionScalarizationPass>::FunctionScalarizationPassBase;

  void runOnOperation() override;
};
} // namespace

namespace {
class ScalarizationPattern
    : public mlir::OpInterfaceRewritePattern<VectorizableOpInterface> {
public:
  ScalarizationPattern(mlir::MLIRContext *context,
                       mlir::SymbolTableCollection &symbolTableCollection)
      : mlir::OpInterfaceRewritePattern<VectorizableOpInterface>(context),
        symbolTableCollection(&symbolTableCollection) {}

  mlir::LogicalResult
  matchAndRewrite(VectorizableOpInterface op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    unsigned int vectorizationRank =
        getVectorizationRank(op, *symbolTableCollection);

    auto args = op.getArgs();

    // Allocate the result arrays. In theory, a vectorizable operation has
    // one and only one result as per the Modelica language specification.
    // However, in order to be resilient to specification changes, the
    // scalarization process has been implemented to also consider multiple
    // results.

    llvm::SmallVector<mlir::Value, 1> results;

    for (const auto &resultType : op->getResultTypes()) {
      assert(mlir::isa<mlir::TensorType>(resultType));
      auto resultTensorType = mlir::cast<mlir::TensorType>(resultType);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (int64_t dim = 0, rank = resultTensorType.getRank(); dim < rank;
           ++dim) {
        int64_t dimensionSize = resultTensorType.getDimSize(dim);

        if (dimensionSize == mlir::ShapedType::kDynamic) {
          // Get the actual size from the first operand. Others should have
          // the same size by construction.

          mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
              loc, rewriter.getIndexAttr(dim));

          dynamicDimensions.push_back(
              rewriter.create<mlir::tensor::DimOp>(loc, args[0], index));
        }
      }

      auto arrayType = ArrayType::get(resultTensorType.getShape(),
                                      resultTensorType.getElementType());

      mlir::Value array =
          rewriter.create<AllocOp>(loc, arrayType, dynamicDimensions);

      results.push_back(array);
    }

    // Compute the results.
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(0));

    llvm::SmallVector<mlir::Value, 3> indices;
    mlir::Operation *outerLoop = nullptr;

    for (unsigned int i = 0; i < vectorizationRank; ++i) {
      mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(i));

      mlir::Value dimension = rewriter.create<mlir::tensor::DimOp>(
          op.getLoc(), op.getArgs()[0], index);

      auto identityMap =
          mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

      auto forOp = rewriter.create<mlir::affine::AffineForOp>(
          loc, zero, identityMap, dimension, identityMap, 1);

      if (!outerLoop) {
        outerLoop = forOp;
      }

      indices.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    // Create the loop body.
    llvm::SmallVector<mlir::Type, 3> scalarResultTypes;

    for (mlir::Type type : op->getResultTypes()) {
      auto currentResultShapedType = mlir::cast<mlir::ShapedType>(type);

      auto slicedShapedType = currentResultShapedType.clone(
          currentResultShapedType.getShape().slice(indices.size()));

      mlir::Type newResultType = mlir::cast<mlir::Type>(slicedShapedType);

      if (slicedShapedType.getShape().empty()) {
        newResultType = slicedShapedType.getElementType();
      }

      scalarResultTypes.push_back(newResultType);
    }

    llvm::SmallVector<mlir::Value> scalarArgs;

    for (mlir::Value arg : args) {
      auto argTensorType = mlir::cast<mlir::TensorType>(arg.getType());

      if (argTensorType.getRank() == static_cast<int64_t>(indices.size())) {
        scalarArgs.push_back(
            rewriter.create<TensorExtractOp>(op.getLoc(), arg, indices));
      } else {
        scalarArgs.push_back(
            rewriter.create<TensorViewOp>(op.getLoc(), arg, indices));
      }
    }

    llvm::SmallVector<mlir::Value, 1> scalarResults;

    if (mlir::failed(op.scalarize(rewriter, scalarArgs, scalarResultTypes,
                                  scalarResults))) {
      return mlir::failure();
    }

    // Copy the (not necessarily) scalar results into the result arrays
    for (size_t i = 0, e = scalarResults.size(); i < e; ++i) {
      mlir::Value scalarResult = scalarResults[i];
      mlir::Value destination = results[i];

      if (auto scalarResultTensorType =
              mlir::dyn_cast<mlir::TensorType>(scalarResult.getType())) {
        destination =
            rewriter.create<SubscriptionOp>(op.getLoc(), destination, indices);

        auto sourceArrayType =
            ArrayType::get(scalarResultTensorType.getShape(),
                           scalarResultTensorType.getElementType());

        mlir::Value source = rewriter.create<TensorToArrayOp>(
            op.getLoc(), sourceArrayType, scalarResult);

        rewriter.create<ArrayCopyOp>(op.getLoc(), source, destination);
      } else {
        auto destinationArrayType =
            mlir::cast<ArrayType>(destination.getType());

        mlir::Value source = scalarResult;

        if (source.getType() != destinationArrayType.getElementType()) {
          source = rewriter.create<CastOp>(
              op.getLoc(), destinationArrayType.getElementType(), source);
        }

        rewriter.create<StoreOp>(op.getLoc(), source, destination, indices);
      }
    }

    // Replace the original operation.
    if (outerLoop) {
      rewriter.setInsertionPointAfter(outerLoop);
    }

    llvm::SmallVector<mlir::Value> tensorResults;

    for (mlir::Value result : results) {
      auto arrayType = mlir::cast<ArrayType>(result.getType());

      auto tensorType = mlir::RankedTensorType::get(arrayType.getShape(),
                                                    arrayType.getElementType());

      mlir::Value tensorResult =
          rewriter.create<ArrayToTensorOp>(op.getLoc(), tensorType, result);

      tensorResults.push_back(tensorResult);
    }

    rewriter.replaceOp(op, tensorResults);
    return mlir::success();
  }

private:
  mlir::SymbolTableCollection *symbolTableCollection;
};
} // namespace

void FunctionScalarizationPass::runOnOperation() {
  /*
  mlir::ModuleOp moduleOp = getOperation();
  mlir::IRRewriter rewriter(moduleOp);

  // List of the vectorized operations, with their respective rank.
  // The rank is stored for efficiency reasons.
  llvm::SmallVector<
      std::pair<VectorizableOpInterface, unsigned int>> vectorizedOps;

  // Reduce the cost of looking for symbols by using symbol tables.
  mlir::SymbolTableCollection symbolTableCollection;

  moduleOp.walk([&](VectorizableOpInterface op) {
    unsigned int rank = getVectorizationRank(op, symbolTableCollection);

    if (rank != 0) {
      vectorizedOps.emplace_back(op, rank);
    }
  });

  for (auto& op : vectorizedOps) {
    if (mlir::failed(scalarizeVectorizableOp(
            rewriter, op.first, op.second, assertions))) {
      return signalPassFailure();
    }
  }
   */

  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  target.markUnknownOpDynamicallyLegal([&](mlir::Operation *op) {
    auto vectorizableOp = mlir::dyn_cast<VectorizableOpInterface>(op);

    if (!vectorizableOp) {
      return true;
    }

    unsigned int rank =
        getVectorizationRank(vectorizableOp, symbolTableCollection);

    return rank == 0;
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<ScalarizationPattern>(&getContext(), symbolTableCollection);

  if (mlir::failed(mlir::applyPartialConversion(moduleOp, target,
                                                std::move(patterns)))) {
    return signalPassFailure();
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createFunctionScalarizationPass() {
  return std::make_unique<FunctionScalarizationPass>();
}

std::unique_ptr<mlir::Pass> createFunctionScalarizationPass(
    const FunctionScalarizationPassOptions &options) {
  return std::make_unique<FunctionScalarizationPass>(options);
}
} // namespace mlir::bmodelica
