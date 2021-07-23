#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/FunctionsVectorization.h>

using namespace modelica::codegen;

static unsigned int getVectorizationRank(VectorizableOpInterface op)
{
	llvm::SmallVector<long, 2> expectedRanks;
	llvm::SmallVector<long, 3> dimensions;

	auto args = op.getArgs();

	if (args.empty())
		return 0;

	for (auto& arg : llvm::enumerate(args))
	{
		mlir::Type argType = arg.value().getType();
		unsigned int argExpectedRank = op.getArgExpectedRank(arg.index());

		unsigned int argActualRank = argType.isa<ArrayType>() ?
																 argType.cast<ArrayType>().getRank() : 0;

		// Each argument must have a rank higher than the expected one
		// for the operation to be vectorized.
		if (argActualRank <= argExpectedRank)
			return 0;

		if (arg.index() == 0)
		{
			// If this is the first argument, then it will determine the
			// rank and dimensions of the result array, although the
			// dimensions can be also specialized by the other arguments
			// if initially unknown.

			for (size_t i = 0; i < argActualRank - argExpectedRank; ++i)
			{
				auto& dimension = argType.cast<ArrayType>().getShape()[arg.index()];
				dimensions.push_back(dimension);
			}
		}
		else
		{
			// The rank difference must match with the one given by the first
			// argument, independently from the dimensions sizes.
			if (argActualRank != argExpectedRank + dimensions.size())
				return 0;

			for (size_t i = 0; i < argActualRank - argExpectedRank; ++i)
			{
				auto& dimension = argType.cast<ArrayType>().getShape()[i];

				// If the dimension is dynamic, then no further checks or
				// specializations are possible.
				if (dimension == -1)
					continue;

				// If the dimension determined by the first argument is fixed,
				// then also the dimension of the other arguments must match
				// (when that's fixed too).
				if (dimensions[i] != -1 && dimensions[i] != dimension)
					return 0;

				// If the dimension determined by the first argument is dynamic, then
				// set it to a required size.
				if (dimensions[i] == -1)
					dimensions[i] = dimension;
			}
		}
	}

	return dimensions.size();
}

static bool isVectorized(VectorizableOpInterface op)
{
	return getVectorizationRank(op) != 0;
}

static mlir::Value allocate(mlir::OpBuilder& builder, mlir::Location loc, ArrayType arrayType, mlir::ValueRange dynamicDimensions = llvm::None)
{
	if (arrayType.getAllocationScope() == BufferAllocationScope::unknown)
		arrayType = arrayType.toMinAllowedAllocationScope();

	if (arrayType.getAllocationScope() == BufferAllocationScope::stack)
		return builder.create<AllocaOp>(loc, arrayType.getElementType(), arrayType.getShape(), dynamicDimensions);

	return builder.create<AllocOp>(loc, arrayType.getElementType(), arrayType.getShape(), dynamicDimensions, true);
}

static void scalarize(mlir::OpBuilder& builder, VectorizableOpInterface op, FunctionsVectorizationOptions options)
{
	mlir::Location loc = op->getLoc();
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPoint(op);

	unsigned int vectorizationRank = getVectorizationRank(op);
	mlir::ValueRange args = op.getArgs();

	// Allocate the result arrays
	llvm::SmallVector<mlir::Value, 3> results;

	for (const auto& resultType : op->getResultTypes())
	{
		assert(resultType.isa<ArrayType>());
		llvm::SmallVector<long, 3> shape;
		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

		for (const auto& dimension : llvm::enumerate(resultType.cast<ArrayType>().getShape()))
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

		auto arrayType = ArrayType::get(
				resultType.getContext(),
				resultType.cast<ArrayType>().getAllocationScope(),
				resultType.cast<ArrayType>().getElementType(),
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

class FunctionsVectorizationPass: public mlir::PassWrapper<FunctionsVectorizationPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit FunctionsVectorizationPass(FunctionsVectorizationOptions options)
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
			if (isVectorized(op))
				scalarize(builder, op, options);
		});
	}

	private:
	FunctionsVectorizationOptions options;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createFunctionsVectorizationPass(FunctionsVectorizationOptions options)
{
	return std::make_unique<FunctionsVectorizationPass>(options);
}
