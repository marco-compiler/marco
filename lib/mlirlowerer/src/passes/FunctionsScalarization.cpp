#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/FunctionsScalarization.h>

using namespace modelica::codegen;

static mlir::Value allocate(mlir::OpBuilder& builder, mlir::Location loc, PointerType pointerType, mlir::ValueRange dynamicDimensions = llvm::None)
{
	if (pointerType.getAllocationScope() == BufferAllocationScope::unknown)
		pointerType = pointerType.toMinAllowedAllocationScope();

	if (pointerType.getAllocationScope() == BufferAllocationScope::stack)
		return builder.create<AllocaOp>(loc, pointerType.getElementType(), pointerType.getShape(), dynamicDimensions);

	return builder.create<AllocOp>(loc, pointerType.getElementType(), pointerType.getShape(), dynamicDimensions, true);
}

class FunctionsScalarizationPass: public mlir::PassWrapper<FunctionsScalarizationPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
		registry.insert<mlir::StandardOpsDialect>();
		registry.insert<mlir::scf::SCFDialect>();
	}

	void runOnOperation() override
	{
		if (auto status = scalarizeBuiltInFunctions(); mlir::failed(status))
		{
			mlir::emitError(getOperation().getLoc(), "Error in scalarizing the vectorized built-in functions");
			return signalPassFailure();
		}

		if (auto status = scalarizeCalls(); mlir::failed(status))
		{
			mlir::emitError(getOperation().getLoc(), "Error in scalarizing the vector calls");
			return signalPassFailure();
		}
	}

	mlir::LogicalResult scalarizeBuiltInFunctions()
	{
		auto module = getOperation();
		mlir::OpBuilder builder(module);

		module->walk([&](VectorizableOpInterface op) {
			if (unsigned int vectorizationRank = op.vectorizationRank(); vectorizationRank != 0)
			{
				mlir::Location loc = op->getLoc();
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPoint(op);

				mlir::ValueRange args = op.getArgs();

				// Allocate the result arrays
				llvm::SmallVector<mlir::Value, 3> results;

				for (const auto& resultType : op->getResultTypes())
				{
					assert(resultType.isa<PointerType>());
					llvm::SmallVector<long, 3> shape;
					llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

					for (const auto& dimension : llvm::enumerate(resultType.cast<PointerType>().getShape()))
					{
						shape.push_back(dimension.value());

						if (dimension.value() == -1)
						{
							// Get the actual size from the first operand. Others should have
							// the same size by construction.

							mlir::Value index = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(dimension.index()));
							dynamicDimensions.push_back(builder.create<DimOp>(loc, args[0], index));
						}
					}

					auto arrayType = PointerType::get(
															 resultType.getContext(),
															 resultType.cast<PointerType>().getAllocationScope(),
															 resultType.cast<PointerType>().getElementType(),
															 shape);

					results.push_back(allocate(builder, loc, arrayType, dynamicDimensions));
				}

				mlir::Value zero = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getIndexAttr(0));
				mlir::Value one = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getIndexAttr(1));
				llvm::SmallVector<mlir::Value, 3> indexes;

				for (unsigned int i = 0; i < vectorizationRank; ++i)
				{
					// TODO: assert on dimensions
					mlir::Value index = builder.create<ConstantOp>(op->getLoc(), builder.getIndexAttr(i));
					mlir::Value dimension = builder.create<DimOp>(op->getLoc(), op.getArgs()[0], index);
					auto forOp = builder.create<mlir::scf::ForOp>(op->getLoc(), zero, dimension, one);
					indexes.push_back(forOp.getInductionVar());
					builder.setInsertionPointToStart(forOp.getBody());
				}

				mlir::ValueRange scalarizedResults = op.scalarize(builder, indexes);

				// Copy the (not necessarily) scalar results into the result arrays
				for (auto result : llvm::enumerate(scalarizedResults))
				{
					mlir::Value subscript = builder.create<SubscriptionOp>(loc, results[result.index()], indexes);
					builder.create<AssignmentOp>(loc, result.value(), subscript);
				}

				// Replace the original operation with the newly allocated arrays
				op->replaceAllUsesWith(results);
				op->erase();
			}
		});

		return mlir::success();
	}

	mlir::LogicalResult scalarizeCalls()
	{
		return mlir::success();
	}
};

std::unique_ptr<mlir::Pass> modelica::codegen::createFunctionsScalarizationPass()
{
	return std::make_unique<FunctionsScalarizationPass>();
}
