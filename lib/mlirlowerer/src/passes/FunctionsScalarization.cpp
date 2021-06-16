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

static void scalarize(mlir::OpBuilder& builder, VectorizableOpInterface op, FunctionsScalarizationOptions options)
{
	mlir::Location loc = op->getLoc();
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(op);

	unsigned int vectorizationRank = op.vectorizationRank();
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

	if (options.assertions)
	{
		llvm::SmallVector<mlir::Value, 3> indexes;

		for (size_t i = 0; i < vectorizationRank; ++i)
			builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(i));

		for (mlir::Value index : indexes)
		{
			llvm::SmallVector<mlir::Value, 3> dimensions;

			for (mlir::Value arg : args)
				dimensions.push_back(builder.create<DimOp>(loc, arg, index));

			for (size_t i = 1; i < dimensions.size(); ++i)
			{
				mlir::Value condition = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, dimensions[0], dimensions[i]);
				builder.create<mlir::AssertOp>(loc, condition, "Incompatible dimensions for vectorized function arguments");
			}
		}
	}

	mlir::Value zero = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getIndexAttr(0));
	mlir::Value one = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getIndexAttr(1));
	llvm::SmallVector<mlir::Value, 3> indexes;

	for (unsigned int i = 0; i < vectorizationRank; ++i)
	{
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

class FunctionsScalarizationPass: public mlir::PassWrapper<FunctionsScalarizationPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit FunctionsScalarizationPass(FunctionsScalarizationOptions options)
			: options(std::move(options))
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
		registry.insert<mlir::StandardOpsDialect>();
		registry.insert<mlir::scf::SCFDialect>();
	}

	void runOnOperation() override
	{
		auto module = getOperation();
		mlir::OpBuilder builder(module);

		module->walk([&](VectorizableOpInterface op) {
			if (op.isVectorized())
				scalarize(builder, op, options);
		});
	}

	private:
	FunctionsScalarizationOptions options;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createFunctionsScalarizationPass(FunctionsScalarizationOptions options)
{
	return std::make_unique<FunctionsScalarizationPass>(options);
}
