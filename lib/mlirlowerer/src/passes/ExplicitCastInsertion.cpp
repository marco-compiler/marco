#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/ExplicitCastInsertion.h>

using namespace modelica;

struct CallOpPattern : public mlir::OpRewritePattern<CallOp>
{
	using mlir::OpRewritePattern<CallOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(CallOp op, mlir::PatternRewriter& rewriter) const override
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
		return mlir::success();
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
	    CallOpPattern,
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
	}

	void runOnOperation() override
	{
		auto module = getOperation();

		ExplicitCastInsertionTarget target(getContext());
		target.addLegalOp<CastOp, CastCommonOp, PtrCastOp>();

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
