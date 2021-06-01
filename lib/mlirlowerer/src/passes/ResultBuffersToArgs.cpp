#include <mlir/Conversion/Passes.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/ResultBuffersToArgs.h>

using namespace modelica::codegen;

struct FunctionOpPattern : public mlir::OpRewritePattern<FunctionOp>
{
	using mlir::OpRewritePattern<FunctionOp>::OpRewritePattern;

	FunctionOpPattern(mlir::MLIRContext* context, std::function<bool(mlir::Type)> moveCondition)
			: mlir::OpRewritePattern<FunctionOp>(context), moveCondition(std::move(moveCondition))
	{
	}

	mlir::LogicalResult matchAndRewrite(FunctionOp op, mlir::PatternRewriter& rewriter) const override
	{
		auto functionType = op.getType();

		llvm::SmallVector<mlir::Type, 3> argsTypes;
		llvm::SmallVector<llvm::StringRef, 3> argsNames;

		llvm::SmallVector<mlir::Type, 3> resultsTypes;
		llvm::SmallVector<llvm::StringRef, 3> resultsNames;

		llvm::SmallVector<unsigned int, 3> erasedResultIndexes;

		argsTypes.append(functionType.getInputs().begin(), functionType.getInputs().end());

		for (const auto& name : op.argsNames())
			argsNames.push_back(name.cast<mlir::StringAttr>().getValue());

		for (auto resultType : llvm::enumerate(functionType.getResults()))
		{
			auto name = op.resultsNames()[resultType.index()].cast<mlir::StringAttr>().getValue();

			if (auto pointerType = resultType.value().dyn_cast<PointerType>();
					pointerType && pointerType.canBeOnStack())
			{
				argsTypes.push_back(pointerType.toUnknownAllocationScope());
				argsNames.push_back(name);
				erasedResultIndexes.push_back(resultType.index());
			}
			else
			{
				resultsTypes.push_back(resultType.value());
				resultsNames.push_back(name);
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

		// Update the return operation
		auto returnOp = mlir::cast<ReturnOp>(function.getBody().back().getTerminator());
		llvm::SmallVector<mlir::Value, 3> returnValues;

		size_t entryArgCounter = op.getNumArguments();
		llvm::SmallVector<mlir::Value, 3> returnOperands;

		for (auto returnValue : llvm::enumerate(returnOp.values()))
		{
			if (moveCondition(returnValue.value().getType()))
			{
				mlir::Value arg = function.getArgument(entryArgCounter);

				// Get the member create operation
				auto* returnValueOp = returnValue.value().getDefiningOp();

				if (!mlir::isa<MemberLoadOp>(returnValueOp))
					return mlir::failure();

				auto memberCreateOp = mlir::cast<MemberLoadOp>(returnValueOp).member().getDefiningOp<MemberCreateOp>();

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
						rewriter.replaceOpWithNewOp<AssignmentOp>(storeOp, storeOp.value(), arg);
					}
				}

				rewriter.eraseOp(memberCreateOp);
				entryArgCounter++;
			}
			else
			{
				returnOperands.push_back(returnValue.value());
			}
		}

		rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, returnValues);
		return mlir::success();
	}

	private:
	std::function<bool(mlir::Type)> moveCondition;
};

struct CallOpPattern : public mlir::OpRewritePattern<CallOp>
{
	using mlir::OpRewritePattern<CallOp>::OpRewritePattern;

	CallOpPattern(mlir::MLIRContext* context, std::function<bool(mlir::Type)> moveCondition)
			: mlir::OpRewritePattern<CallOp>(context), moveCondition(std::move(moveCondition))
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

				auto pointerType = result.value().getType().cast<PointerType>();

				mlir::Value buffer = rewriter.create<AllocaOp>(
						result.value().getLoc(), pointerType.getElementType(), pointerType.getShape());

				buffer = rewriter.create<PtrCastOp>(
						buffer.getLoc(), buffer,
						buffer.getType().cast<PointerType>().toUnknownAllocationScope());

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
			if (auto pointerType = result.getType().dyn_cast<PointerType>();
					pointerType && pointerType.canBeOnStack())
			{
				resultsView.push_back(buffers[buffersIndex++]);
			}
			else
			{
				resultsView.push_back(newCall.getResult(newResultsIndex++));
			}
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

		target.addDynamicallyLegalOp<FunctionOp>([](FunctionOp op) {
			return llvm::none_of(op.getType().getResults(), [](mlir::Type type) {
				if (auto pointerType = type.dyn_cast<PointerType>())
					return pointerType.canBeOnStack();

				return false;
			});
		});

		target.addDynamicallyLegalOp<CallOp>([](CallOp op) {
			return llvm::none_of(op->getResults(), [](mlir::Value value) {
				if (auto pointerType = value.getType().dyn_cast<PointerType>())
					return pointerType.canBeOnStack();

				return false;
			});
		});

		mlir::OwningRewritePatternList patterns(&getContext());

		patterns.insert<FunctionOpPattern, CallOpPattern>(
				&getContext(), [](mlir::Type type) {
					if (auto pointerType = type.dyn_cast<PointerType>())
						return pointerType.canBeOnStack();

					return false;
				});

		if (auto status = applyPartialConversion(getOperation(), target, std::move(patterns)); failed(status))
			return signalPassFailure();
	}
};

std::unique_ptr<mlir::Pass> modelica::codegen::createResultBuffersToArgsPass()
{
	return std::make_unique<ResultBuffersToArgsPass>();
}
