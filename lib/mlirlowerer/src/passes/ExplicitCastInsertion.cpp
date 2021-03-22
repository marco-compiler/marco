#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/ExplicitCastInsertion.h>

using namespace modelica;

static bool isNumeric(mlir::Type type)
{
	return type.isa<mlir::IndexType, BooleanType, IntegerType, RealType>();
}

static bool isNumeric(mlir::Value value)
{
	return isNumeric(value.getType());
}

struct CallOpScalarPattern : public mlir::OpRewritePattern<CallOp>
{
	using mlir::OpRewritePattern<CallOp>::OpRewritePattern;

	mlir::LogicalResult match(CallOp op) const override
	{
		auto module = op->getParentOfType<mlir::ModuleOp>();
		auto callee = module.lookupSymbol<mlir::FuncOp>(op.callee());
		assert(op.args().size() == callee.getArgumentTypes().size());
		auto pairs = llvm::zip(op.args(), callee.getArgumentTypes());

		for (auto [ arg, type ] : pairs)
		{
			mlir::Type actualType = arg.getType();

			if (!actualType.isa<PointerType>() && !type.isa<PointerType>())
				continue;

			if (!actualType.isa<PointerType>() && type.isa<PointerType>())
				return mlir::failure();

			if (actualType.isa<PointerType>() && !type.isa<PointerType>())
				return mlir::failure();

			if (actualType.cast<PointerType>().getRank() != type.cast<PointerType>().getRank())
				return mlir::failure();
		}

		return mlir::success();
	}

	void rewrite(CallOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();

		auto module = op->getParentOfType<mlir::ModuleOp>();
		auto callee = module.lookupSymbol<mlir::FuncOp>(op.callee());

		llvm::SmallVector<mlir::Value, 3> args;

		for (auto [ arg, type ] : llvm::zip(op.args(), callee.getArgumentTypes()))
		{
			if (arg.getType() != type)
			{
				if (arg.getType().isa<PointerType>())
					arg = rewriter.create<PtrCastOp>(location, arg, type);
				else
					arg = rewriter.create<CastOp>(location, arg, type);
			}

			args.push_back(arg);
		}

		rewriter.replaceOpWithNewOp<CallOp>(op, op.callee(), op.getResultTypes(), args, op.movedResults());
	}
};

struct CallOpElementWisePattern : public mlir::OpRewritePattern<CallOp>
{
	using mlir::OpRewritePattern<CallOp>::OpRewritePattern;

	mlir::LogicalResult match(CallOp op) const override
	{
		auto module = op->getParentOfType<mlir::ModuleOp>();
		auto callee = module.lookupSymbol<mlir::FuncOp>(op.callee());
		assert(op.args().size() == callee.getArgumentTypes().size());
		auto pairs = llvm::zip(op.args(), callee.getArgumentTypes());

		unsigned int rankDifference = 0;

		for (auto [ arg, type ] : pairs)
		{
			mlir::Type actualType = arg.getType();

			if (!actualType.isa<PointerType>())
				return mlir::failure();

			unsigned int typeRank = 0;

			if (type.isa<PointerType>())
				typeRank = type.cast<PointerType>().getRank();

			if (actualType.cast<PointerType>().getRank() == typeRank)
				return mlir::failure();

			unsigned int currentDifference = actualType.cast<PointerType>().getRank() - typeRank;

			if (rankDifference == 0)
				rankDifference = currentDifference;
			else if (currentDifference != rankDifference)
				return mlir::failure();
		}

		return mlir::success();
	}

	void rewrite(CallOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();

		// Get the calle
		auto module = op->getParentOfType<mlir::ModuleOp>();
		auto callee = module.lookupSymbol<mlir::FuncOp>(op.callee());

		// Ensure that the call has at least an argument. If not, we can't
		// determine the result arrays sizes.
		assert(op.args().size() - op.movedResults() > 0);

		// Allocate the result arrays
		llvm::SmallVector<mlir::Value, 3> results;

		for (mlir::Type resultType : op->getResultTypes())
		{
			assert(resultType.isa<PointerType>());
			llvm::SmallVector<long, 3> shape;
			llvm::SmallVector<mlir::Value, 3> dims;

			for (auto size : llvm::enumerate(resultType.cast<PointerType>().getShape()))
			{
				shape.push_back(size.value());

				if (size.value() == -1)
				{
					// Get the actual size from the first operand. Others should have
					// the same size by construction.

					mlir::Value index = rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(size.index()));
					dims.push_back(rewriter.create<DimOp>(location, op.args()[0], index));
				}
			}

			mlir::Value array = rewriter.create<AllocOp>(location, resultType.cast<PointerType>().getElementType(), shape, dims);
			results.push_back(array);
		}

		// Iterate on the indexes
		mlir::Value zero = rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(0));
		mlir::Value one = rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(1));

		assert(op.elementWiseRank() != 0);
		llvm::SmallVector<mlir::Value, 3> lowerBounds(op.elementWiseRank(), zero);
		llvm::SmallVector<mlir::Value, 3> upperBounds;
		llvm::SmallVector<mlir::Value, 3> steps(op.elementWiseRank(), one);

		for (unsigned int i = 0, e = op.elementWiseRank(); i < e; ++i)
		{
			mlir::Value dim = rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(i));
			upperBounds.push_back(rewriter.create<DimOp>(location, op.args()[0], dim));
		}

		mlir::scf::buildLoopNest(
				rewriter, location, lowerBounds, upperBounds, steps, llvm::None,
				[&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange position, mlir::ValueRange args) -> std::vector<mlir::Value> {
					llvm::SmallVector<mlir::Value, 3> scalarArgs;

					for (mlir::Value arg : op.args())
					{
						assert(arg.getType().isa<PointerType>());

						if (arg.getType().cast<PointerType>().getRank() == op.elementWiseRank())
							scalarArgs.push_back(rewriter.create<LoadOp>(location, arg, position));
						else
							scalarArgs.push_back(rewriter.create<SubscriptionOp>(location, arg, position));
					}

					llvm::SmallVector<mlir::Type, 3> scalarResultTypes;

					for (mlir::Type type : op.getResultTypes())
					{
						type = type.cast<PointerType>().slice(op.elementWiseRank());

						if (type.cast<PointerType>().getRank() == 0)
							type = type.cast<PointerType>().getElementType();

						scalarResultTypes.push_back(type);
					}

					auto scalarCall = rewriter.create<CallOp>(location, op.callee(), scalarResultTypes, scalarArgs);

					for (auto result : llvm::enumerate(scalarCall->getResults()))
					{
						mlir::Value subscript = rewriter.create<SubscriptionOp>(location, results[result.index()], position);
						rewriter.create<AssignmentOp>(location, result.value(), subscript);
					}

					return std::vector<mlir::Value>();
				});

		rewriter.replaceOp(op, results);
	}
};

struct SubscriptionOpPattern : public mlir::OpRewritePattern<SubscriptionOp>
{
	using mlir::OpRewritePattern<SubscriptionOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(SubscriptionOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		llvm::SmallVector<mlir::Value, 3> indexes;

		for (mlir::Value index : op.indexes())
		{
			if (!index.getType().isa<mlir::IndexType>())
				index = rewriter.create<CastOp>(location, index, rewriter.getIndexType());

			indexes.push_back(index);
		}

		rewriter.replaceOpWithNewOp<SubscriptionOp>(op, op.source(), indexes);
		return mlir::success();
	}
};

struct StoreOpPattern : public mlir::OpRewritePattern<StoreOp>
{
	using mlir::OpRewritePattern<StoreOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(StoreOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		mlir::Type elementType = op.memory().getType().cast<PointerType>().getElementType();
		mlir::Value value = rewriter.create<CastOp>(location, value, elementType);
		rewriter.replaceOpWithNewOp<StoreOp>(op, value, op.memory(), op.indexes());
		return mlir::success();
	}
};

struct ConditionOpPattern : public mlir::OpRewritePattern<ConditionOp>
{
	using mlir::OpRewritePattern<ConditionOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(ConditionOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		mlir::Value condition = rewriter.create<CastOp>(location, op.condition(), BooleanType::get(op.getContext()));
		rewriter.replaceOpWithNewOp<ConditionOp>(op, condition, op.args());
		return mlir::success();
	}
};

void populateExplicitCastInsertionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context)
{
	patterns.insert<
	    CallOpScalarPattern,
			CallOpElementWisePattern,
			SubscriptionOpPattern,
			StoreOpPattern,
			ConditionOpPattern>(context);
}

struct ExplicitCastInsertionTarget : public mlir::ConversionTarget
{
	ExplicitCastInsertionTarget(mlir::MLIRContext& context) : mlir::ConversionTarget(context)
	{
		addDynamicallyLegalOp<CallOp>(
				[](CallOp op)
				{
					auto module = op->getParentOfType<mlir::ModuleOp>();
					auto callee = module.lookupSymbol<mlir::FuncOp>(op.callee());
					assert(op.args().size() == callee.getArgumentTypes().size());
					auto pairs = llvm::zip(op.args(), callee.getArgumentTypes());

					return std::all_of(pairs.begin(), pairs.end(),
														 [&](const auto& pair)
														 {
															 return std::get<0>(pair).getType() == std::get<1>(pair);
														 });
				});

		addDynamicallyLegalOp<SubscriptionOp>(
				[](SubscriptionOp op)
				{
					auto indexes = op.indexes();
					return std::all_of(indexes.begin(), indexes.end(),
														 [](mlir::Value index)
														 { return index.getType().isa<mlir::IndexType>(); });
				});

		addDynamicallyLegalOp<StoreOp>(
				[](StoreOp op)
				{
					mlir::Type elementType = op.memory().getType().cast<PointerType>().getElementType();
					return op.value().getType() == elementType;
				});

		addDynamicallyLegalOp<ConditionOp>(
				[](ConditionOp op)
				{
					mlir::Type conditionType = op.condition().getType();
					return conditionType.isa<BooleanType>();
				});
	}

	bool isDynamicallyLegal(mlir::Operation* op) const override
	{
		return false;
	}
};

class ExplicitCastInsertionPass: public mlir::PassWrapper<ExplicitCastInsertionPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
		registry.insert<mlir::scf::SCFDialect>();
	}

	void runOnOperation() override
	{
		auto module = getOperation();

		ExplicitCastInsertionTarget target(getContext());
		target.addLegalDialect<ModelicaDialect>();
		target.addLegalDialect<mlir::scf::SCFDialect>();

		mlir::OwningRewritePatternList patterns;
		populateExplicitCastInsertionPatterns(patterns, &getContext());

		if (failed(applyPartialConversion(module, target, std::move(patterns))))
		{
			mlir::emitError(module.getLoc(), "Error in inserting the explicit casts\n");
			signalPassFailure();
		}
	}
};

std::unique_ptr<mlir::Pass> modelica::createExplicitCastInsertionPass()
{
	return std::make_unique<ExplicitCastInsertionPass>();
}
