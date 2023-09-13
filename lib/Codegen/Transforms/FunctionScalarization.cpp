#include "marco/Codegen/Transforms/FunctionScalarization.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Conversion/Passes.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_FUNCTIONSCALARIZATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

static unsigned int getVectorizationRank(
    VectorizableOpInterface op, mlir::SymbolTableCollection& symbolTable)
{
	// Only functions with exactly one result can be vectorized, as per
	// Modelica language specification.

	if (op->getNumResults() != 1) {
    return 0;
  }

	// List of the additional dimensions with respect to the expected scalar
	// arguments.

	llvm::SmallVector<long, 3> dimensions;

	auto args = op.getArgs();

	if (args.empty()) {
    return 0;
  }

	for (const auto& arg : llvm::enumerate(args)) {
		mlir::Type argType = arg.value().getType();

		unsigned int argExpectedRank = op.getArgExpectedRank(
        arg.index(), symbolTable);

		unsigned int argActualRank = argType.isa<ArrayType>() ?
																 argType.cast<ArrayType>().getRank() : 0;

		// Each argument must have a rank higher than the expected one
		// for the operation to be vectorized.

		if (argActualRank <= argExpectedRank) {
      return 0;
    }

		if (arg.index() == 0) {
			// If this is the first argument, then it will determine the
			// rank and dimensions of the result array, although the
			// dimensions can be also specialized by the other arguments
			// if initially unknown.

			for (size_t i = 0; i < argActualRank - argExpectedRank; ++i) {
				auto dimension = argType.cast<ArrayType>().getShape()[arg.index()];
				dimensions.push_back(dimension);
			}
		} else {
			// The rank difference must match with the one given by the first
			// argument, independently from the dimensions sizes.

			if (argActualRank != argExpectedRank + dimensions.size()) {
        return 0;
      }

			for (size_t i = 0; i < argActualRank - argExpectedRank; ++i) {
				auto dimension = argType.cast<ArrayType>().getShape()[i];

				// If the dimension is dynamic, then no further checks or
				// specializations are possible.

				if (dimension == -1) {
          continue;
        }

				// If the dimension determined by the first argument is fixed,
				// then also the dimension of the other arguments must match
				// (when that's fixed too).

				if (dimensions[i] != -1 && dimensions[i] != dimension) {
          return 0;
        }

				// If the dimension determined by the first argument is dynamic, then
				// set it to a required size.

        if (dimensions[i] == -1) {
          dimensions[i] = dimension;
        }
			}
		}
	}

	return dimensions.size();
}

static mlir::LogicalResult scalarizeVectorizableOp(
    mlir::OpBuilder& builder, VectorizableOpInterface op, unsigned int vectorizationRank, bool assertions)
{
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  mlir::ValueRange args = op.getArgs();

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

    for (const auto& dimension : llvm::enumerate(resultType.cast<ArrayType>().getShape())) {
      shape.push_back(dimension.value());

      if (dimension.value() == -1) {
        // Get the actual size from the first operand. Others should have
        // the same size by construction.

        mlir::Value index = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(dimension.index()));
        dynamicDimensions.push_back(builder.create<DimOp>(loc, args[0], index));
      }
    }

    auto arrayType = ArrayType::get(
        shape,
        resultType.cast<ArrayType>().getElementType());

    mlir::Value array =  builder.create<AllocOp>(loc, arrayType, dynamicDimensions);
    results.push_back(array);
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
        mlir::Value condition = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, dimensions[0], dimensions[i]);
        builder.create<mlir::cf::AssertOp>(loc, condition, "Incompatible dimensions for vectorized function arguments");
      }
    }
  }

  mlir::Value zero = builder.create<mlir::arith::ConstantOp>(op->getLoc(), builder.getIndexAttr(0));
  mlir::Value one = builder.create<mlir::arith::ConstantOp>(op->getLoc(), builder.getIndexAttr(1));
  llvm::SmallVector<mlir::Value, 3> indexes;

  for (unsigned int i = 0; i < vectorizationRank; ++i) {
    mlir::Value index = builder.create<ConstantOp>(op->getLoc(), builder.getIndexAttr(i));
    mlir::Value dimension = builder.create<DimOp>(op->getLoc(), op.getArgs()[0], index);

    // The scf.parallel operation supports multiple indexes, but
    // unfortunately the OpenMP -> LLVM-IR doesn't do it yet. For now,
    // let's keep the parallelization only on one index, and especially
    // on the outer one so that we can take advantage of data locality.

    if (i == 0) {
      auto parallelOp = builder.create<mlir::scf::ParallelOp>(loc, zero, dimension, one);
      indexes.push_back(parallelOp.getInductionVars()[0]);
      builder.setInsertionPointToStart(parallelOp.getBody());
    } else {
      auto forOp = builder.create<mlir::scf::ForOp>(loc, zero, dimension, one);
      indexes.push_back(forOp.getInductionVar());
      builder.setInsertionPointToStart(forOp.getBody());
    }
  }

  mlir::ValueRange scalarizedResults = op.scalarize(builder, indexes);

  // Copy the (not necessarily) scalar results into the result arrays
  for (auto result : llvm::enumerate(scalarizedResults)) {
    mlir::Value subscript = builder.create<SubscriptionOp>(loc, results[result.index()], indexes);
    builder.create<AssignmentOp>(loc, subscript, result.value());
  }

  // Replace the original operation with the newly allocated arrays
  op->replaceAllUsesWith(results);
  op->erase();

  return mlir::success();
}

namespace
{
  class FunctionScalarizationPass
      : public impl::FunctionScalarizationPassBase<FunctionScalarizationPass>
  {
    public:
      using FunctionScalarizationPassBase::FunctionScalarizationPassBase;

      void runOnOperation() override
      {
        auto module = getOperation();
        mlir::OpBuilder builder(module);

        // List of the vectorized operations, with their respective rank.
        // The rank is stored for efficiency reasons.
        llvm::SmallVector<
            std::pair<VectorizableOpInterface, unsigned int>, 3> vectorizedOps;

        // Create an instance of the symbol table in order to reduce the cost
        // of looking for symbols.
        mlir::SymbolTableCollection symbolTable;

        module->walk([&](VectorizableOpInterface op) {
          unsigned int rank = getVectorizationRank(op, symbolTable);

          if (rank != 0) {
            vectorizedOps.emplace_back(op, rank);
          }
        });

        for (auto& op : vectorizedOps) {
          if (mlir::failed(scalarizeVectorizableOp(
                  builder, op.first, op.second, assertions))) {
            return signalPassFailure();
          }
        }
      }
  };
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createFunctionScalarizationPass()
  {
    return std::make_unique<FunctionScalarizationPass>();
  }

  std::unique_ptr<mlir::Pass> createFunctionScalarizationPass(
      const FunctionScalarizationPassOptions& options)
  {
    return std::make_unique<FunctionScalarizationPass>(options);
  }
}
