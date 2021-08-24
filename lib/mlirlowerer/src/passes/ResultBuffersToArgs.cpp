#include <mlir/Conversion/Passes.h>
#include <marco/mlirlowerer/ModelicaDialect.h>
#include <marco/mlirlowerer/passes/ResultBuffersToArgs.h>

using namespace marco::codegen;

struct FunctionOpPattern : public mlir::OpRewritePattern<FunctionOp>
{
	using mlir::OpRewritePattern<FunctionOp>::OpRewritePattern;

	FunctionOpPattern(mlir::MLIRContext* context, std::function<bool(mlir::Type)> moveCondition)
			: mlir::OpRewritePattern<FunctionOp>(context),
				moveCondition(std::move(moveCondition))
	{
	}

	mlir::LogicalResult matchAndRewrite(FunctionOp op, mlir::PatternRewriter& rewriter) const override
	{
		auto functionType = op.getType();

		llvm::SmallVector<mlir::Type, 3> argsTypes;
		llvm::SmallVector<llvm::StringRef, 3> argsNames;

		llvm::SmallVector<mlir::Type, 3> resultsTypes;
		llvm::SmallVector<llvm::StringRef, 3> resultsNames;

		llvm::SmallVector<llvm::StringRef, 3> movedResults;

		// The moved arguments will be appended to the original arguments, so we
		// first need to store the original arguments name and type.

		argsTypes.append(functionType.getInputs().begin(), functionType.getInputs().end());

		for (const auto& name : op.argsNames())
			argsNames.push_back(name.cast<mlir::StringAttr>().getValue());

		// Then we move the movable results

		for (auto [name, type] : llvm::zip(op.resultsNames(), functionType.getResults()))
		{
			auto nameStr = name.cast<mlir::StringAttr>().getValue();

			if (moveCondition(type))
			{
				argsTypes.push_back(type.cast<ArrayType>().toUnknownAllocationScope());
				argsNames.push_back(nameStr);
				movedResults.push_back(nameStr);
			}
			else
			{
				resultsTypes.push_back(type);
				resultsNames.push_back(nameStr);
			}
		}

		// Create the function with the new signature and move the old body
		// into it. Then operate on this new function.

		assert(argsTypes.size() == argsNames.size());

		auto function = rewriter.replaceOpWithNewOp<FunctionOp>(
				op, op.getName(), rewriter.getFunctionType(argsTypes, resultsTypes), argsNames, resultsNames);

		if (op.isExternal())
			return mlir::success();

		mlir::BlockAndValueMapping mapping;

		// Clone the blocks structure of the original function
		for (auto& block : llvm::enumerate(op.getRegion().getBlocks()))
		{
			mlir::TypeRange types = block.index() == 0 ? mlir::TypeRange(argsTypes) : mlir::TypeRange(block.value().getArgumentTypes());

			mlir::Block* clonedBlock = rewriter.createBlock(
					&function.getRegion(), function.getRegion().end(), types);

			mapping.map(&block.value(), clonedBlock);
		}

		// Copy the old blocks content
		for (auto& sourceBlock : llvm::enumerate(op.getBody().getBlocks()))
		{
			auto& destinationBlock = *std::next(function.getBody().getBlocks().begin(), sourceBlock.index());

			for (const auto& arg : llvm::enumerate(sourceBlock.value().getArguments()))
				mapping.map(arg.value(), destinationBlock.getArgument(arg.index()));

			rewriter.setInsertionPointToStart(&destinationBlock);

			for (auto& sourceOp : sourceBlock.value().getOperations())
				rewriter.clone(sourceOp, mapping);
		}

		// Map the members for faster access
		llvm::StringMap<MemberCreateOp> members;

		function->walk([&members](MemberCreateOp op) {
			members[op.name()] = op;
		});

		for (auto movedResultName : llvm::enumerate(movedResults))
		{
			if (members.count(movedResultName.value()) == 0)
				return mlir::failure();

			auto memberCreateOp = members[movedResultName.value()];
			mlir::Value arg = function.getArgument(op.getNumArguments() + movedResultName.index());

			// Remove the operations operating on the member
			for (auto* memberUser : memberCreateOp->getUsers())
			{
				if (auto loadOp = mlir::dyn_cast<MemberLoadOp>(memberUser))
				{
					rewriter.replaceOp(loadOp, arg);

					// Fix the subscription operations, which still have the old
					// allocation scope in its result type.

					for (auto* loadUser : loadOp.getResult().getUsers())
					{
						if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(loadUser))
						{
							mlir::OpBuilder::InsertionGuard guard(rewriter);
							rewriter.setInsertionPoint(subscriptionOp);

							rewriter.replaceOpWithNewOp<SubscriptionOp>(
									subscriptionOp, arg, subscriptionOp.indexes());
						}
					}
				}
				else if (auto storeOp = mlir::dyn_cast<MemberStoreOp>(memberUser))
				{
					mlir::OpBuilder::InsertionGuard guard(rewriter);
					rewriter.setInsertionPoint(storeOp);
					rewriter.replaceOpWithNewOp<AssignmentOp>(storeOp, storeOp.value(), arg);
				}
			}

			rewriter.eraseOp(memberCreateOp);
		}

		return mlir::success();
	}

	private:
	std::function<bool(mlir::Type)> moveCondition;
};

struct CallOpPattern : public mlir::OpRewritePattern<CallOp>
{
	using mlir::OpRewritePattern<CallOp>::OpRewritePattern;

	CallOpPattern(mlir::MLIRContext* context, std::function<bool(mlir::Type)> moveCondition)
			: mlir::OpRewritePattern<CallOp>(context),
				moveCondition(std::move(moveCondition))
	{
	}

	mlir::LogicalResult matchAndRewrite(CallOp op, mlir::PatternRewriter& rewriter) const override
	{
		// Determine which results have become an argument and allocate their
		// buffers in the caller. Simultaneously, determine the new call results
		// types.

		llvm::SmallVector<unsigned int, 3> removedResultsIndexes;
		llvm::SmallVector<mlir::Value, 3> buffers;
		llvm::SmallVector<mlir::Type, 3> newResultsTypes;

		for (auto result : llvm::enumerate(op.getResults()))
		{
			if (moveCondition(result.value().getType()))
			{
				// The buffer can be allocated on the stack, because the result has
				// been moved precisely because it can be allocated there.

				auto arrayType = result.value().getType().cast<ArrayType>();

				mlir::Value buffer = rewriter.create<AllocaOp>(
						result.value().getLoc(), arrayType.getElementType(), arrayType.getShape());

				buffer = rewriter.create<ArrayCastOp>(
						buffer.getLoc(), buffer,
						buffer.getType().cast<ArrayType>().toUnknownAllocationScope());

				buffers.push_back(buffer);
				removedResultsIndexes.push_back(result.index());
			}
			else
			{
				newResultsTypes.push_back(result.value().getType());
			}
		}

		// The new call arguments are the old ones plus the new buffers
		llvm::SmallVector<mlir::Value, 3> args;
		args.append(op.args().begin(), op.args().end());
		args.append(buffers);

		auto newCall = rewriter.create<CallOp>(
				op->getLoc(), op.callee(), newResultsTypes, args, buffers.size());

		// Create a view over the old results that replace the moved results with
		// the new allocated buffers.

		llvm::SmallVector<mlir::Value, 3> resultsView;
		unsigned int buffersIndex = 0;
		unsigned int newResultsIndex = 0;

		for (const auto& result : op.getResults())
		{
			if (moveCondition(result.getType()))
				resultsView.push_back(buffers[buffersIndex++]);
			else
				resultsView.push_back(newCall.getResult(newResultsIndex++));
		}

		rewriter.replaceOp(op, resultsView);
		return mlir::success();
	}

	private:
	std::function<bool(mlir::Type)> moveCondition;
};

class ResultBuffersToArgsPass: public mlir::PassWrapper<ResultBuffersToArgsPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
	}

	void runOnOperation() override
	{
		llvm::SmallVector<llvm::StringRef, 3> modifiedFunctions;
		mlir::ConversionTarget target(getContext());

		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
			return true;
		});

		auto moveCondition = [](mlir::Type type) -> bool {
			if (auto arrayType = type.dyn_cast<ArrayType>())
				return arrayType.canBeOnStack();

			return false;
		};

		target.addDynamicallyLegalOp<FunctionOp>([&moveCondition](FunctionOp op) {
			return llvm::none_of(op.getType().getResults(), moveCondition);
		});

		target.addDynamicallyLegalOp<CallOp>([&moveCondition](CallOp op) {
			return llvm::none_of(op->getResults().getTypes(), moveCondition);
		});

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<FunctionOpPattern, CallOpPattern>(&getContext(), moveCondition);

		if (auto status = applyPartialConversion(getOperation(), target, std::move(patterns)); failed(status))
			return signalPassFailure();
	}
};

std::unique_ptr<mlir::Pass> marco::codegen::createResultBuffersToArgsPass()
{
	return std::make_unique<ResultBuffersToArgsPass>();
}
