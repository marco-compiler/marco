#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/AutomaticDifferentiation.h>

using namespace modelica::codegen;

static bool hasFloatBase(mlir::Type type) {
	if (type.isa<RealType>())
		return true;

	if (auto pointerType = type.dyn_cast<PointerType>();
			pointerType && pointerType.getElementType().isa<RealType>())
		return true;

	return false;
};

template <class T>
unsigned int numDigits(T number)
{
	unsigned int digits = 0;

	while (number != 0)
	{
		number /= 10;
		++digits;
	}

	return digits;
}

static std::string getPartialDerivativeName(llvm::StringRef functionName, llvm::StringRef arg)
{
	return "__der_" + functionName.str() + "_" + arg.str();
}

static std::string getDerVariableName(llvm::StringRef variableName, unsigned int order)
{
	// Compose the derivative member name according to the derivative order.
	// If the order is 1, then it is omitted.
	assert(order > 0);

	if (order == 1)
		return "der_" + variableName.str();

	return "der_" + std::to_string(order) + "_" + variableName.str();
}

static std::string getNextDerVariableName(llvm::StringRef currentName, unsigned int currentOrder)
{
	if (currentOrder == 1)
		return getDerVariableName(currentName, currentOrder);

	assert(currentName.rfind("der_") == 0);

	if (currentOrder == 2)
		return getDerVariableName(currentName.substr(5), currentOrder);

	return getDerVariableName(currentName.substr(5 + numDigits(currentOrder - 1)), currentOrder);
}

static FunctionOp createPartialDerivativeFunction(FunctionOp base, llvm::StringRef derivativeName, llvm::StringRef independentVar)
{
	mlir::OpBuilder builder(base);
	auto module = base->getParentOfType<mlir::ModuleOp>();

	llvm::SmallVector<llvm::StringRef, 3> argsNames;
	llvm::SmallVector<llvm::StringRef, 3> resultsNames;

	for (const auto& argName : base.argsNames())
		argsNames.push_back(argName.cast<mlir::StringAttr>().getValue());

	for (const auto& argName : base.resultsNames())
		resultsNames.push_back(argName.cast<mlir::StringAttr>().getValue());

	auto function = builder.create<FunctionOp>(base.getLoc(), derivativeName, base.getType(), argsNames, resultsNames);

	mlir::BlockAndValueMapping mapping;

	// Clone the blocks structure of the function to be derived. The
	// operations contained in the blocks are not copied.

	for (auto& block : base.getRegion().getBlocks())
	{
		mlir::Block* clonedBlock = builder.createBlock(
				&function.getRegion(), function.getRegion().end(), block.getArgumentTypes());

		mapping.map(&block, clonedBlock);
	}

	mlir::BlockAndValueMapping derivatives;
	builder.setInsertionPointToStart(&function.getRegion().front());

	// Create the arguments derivatives
	for (const auto& arg : function.getArguments())
	{
		mlir::Type type = arg.getType();

		auto memberType = type.isa<PointerType>() ?
											MemberType::get(type.cast<PointerType>()) :
											MemberType::get(builder.getContext(), MemberAllocationScope::stack, type);

		// TODO: handle dynamic dimensions
		mlir::Value der = builder.create<MemberCreateOp>(base.getLoc(), memberType, llvm::None);
		der = builder.create<MemberLoadOp>(base->getLoc(), type, der);
		derivatives.map(arg, der);
	}

	// Iterate over the original operations and create their derivatives
	// (if possible) inside the new function.

	for (auto& baseBlock : llvm::enumerate(base.getBody().getBlocks()))
	{
		auto& block = *std::next(function.getBlocks().begin(), baseBlock.index());
		builder.setInsertionPointToEnd(&block);

		// Map the original block arguments to the new block ones
		for (const auto& [original, mapped] : llvm::zip(baseBlock.value().getArguments(), block.getArguments()))
			mapping.map(original, mapped);

		for (auto& baseOp : baseBlock.value().getOperations())
		{
			if (auto returnOp = mlir::dyn_cast<ReturnOp>(baseOp))
			{
				llvm::SmallVector<mlir::Value, 3> derivedValues;

				for (mlir::Value value : returnOp.values())
					derivedValues.push_back(derivatives.lookup(mapping.lookup(value)));

				builder.create<ReturnOp>(returnOp.getLoc(), derivedValues);
				continue;
			}

			auto* cloned = builder.clone(baseOp, mapping);
			builder.setInsertionPoint(cloned);

			if (auto deriveInterface = mlir::dyn_cast<DerivativeInterface>(cloned))
			{
				deriveInterface.derive(
						builder, derivatives,
						[&function](mlir::OpBuilder& builder, std::function<mlir::ValueRange(mlir::OpBuilder&)> allocator) -> mlir::ValueRange {
							mlir::OpBuilder::InsertionGuard guard(builder);
							builder.setInsertionPointToStart(&function.getRegion().front());
							return allocator(builder);
						});
			}
			else
				return nullptr;
			// TODO: is there a better alternative to nullptr?

			builder.setInsertionPointAfter(cloned);
		}
	}

	return function;
}

static mlir::LogicalResult createFunctionDerivative(FunctionOp op)
{
	mlir::OpBuilder builder(op);
	auto module = op->getParentOfType<mlir::ModuleOp>();

	// Create the partial derivative functions
	for (const auto& [name, type] : llvm::zip(op.argsNames(), op.getType().getInputs()))
	{
		if (!hasFloatBase(type))
			continue;

		llvm::StringRef argName = name.cast<mlir::StringAttr>().getValue();
		std::string partialDerFunctionName = getPartialDerivativeName(op.getName(), argName);

		// Check if the derivative already exists
		if (module.lookupSymbol(partialDerFunctionName) != nullptr)
			continue;

		createPartialDerivativeFunction(op, partialDerFunctionName, name.cast<mlir::StringAttr>().getValue());
	}

	auto derivativeAttribute = op->getAttrOfType<DerivativeAttribute>("derivative");
	unsigned int order = derivativeAttribute.getOrder();

	// If the source already provides the derivative function, then stop
	if (module.lookupSymbol<FunctionOp>(derivativeAttribute.getName()))
		return mlir::success();

	// Create a map of the source arguments for a faster lookup
	llvm::StringMap<mlir::Value> sourceArgsMap;

	for (const auto& [name, value] : llvm::zip(op.argsNames(), op.getArguments()))
		sourceArgsMap[name.cast<mlir::StringAttr>().getValue()] = value;

	// The new arguments names are stored here because they will be appended
	// only when all the original arguments will have been processed. Note that
	// the name container stores std::strings instead of llvm::StringRefs, as the
	// latter would refer to a std::string allocated within the following loop
	// (and thus the references would then become invalid).
	llvm::SmallVector<std::string, 3> newArgsNamesBuffers;
	llvm::SmallVector<mlir::Type, 3> newArgsTypes;

	for (const auto& [name, type] : llvm::zip(op.argsNames(), op.getType().getInputs()))
	{
		llvm::StringRef argName = name.cast<mlir::StringAttr>().getValue();

		if (hasFloatBase(type))
		{
			// If the current argument name starts with der, we need to check if
			// the original function to be derived has a member whose derivative
			// may be the current one. If this is the case, then we don't need to
			// add the n-th derivative as it is already done when encountering that
			// member. If it is not, then it means the original function had a
			// "strange" member named "der_something" and the derivative function
			// will contain both "der_something" and "der_der_something"; that
			// original "der_something" could effectively be a derivative, but
			// this is an assumption we can't do.

			if (argName.rfind("der_") == 0)
			{
				auto isDerivative = [&](llvm::StringRef name) {
					for (const auto& arg : sourceArgsMap)
					{
						for (unsigned int i = 1; i < order; ++i)
							if (name == getDerVariableName(arg.first(), i))
								return true;
					}

					return false;
				};

				if (isDerivative(argName))
					continue;
			}

			newArgsNamesBuffers.push_back(getDerVariableName(argName, order));
			newArgsTypes.push_back(type);
		}
	}

	llvm::SmallVector<llvm::StringRef, 3> argsNames;
	llvm::SmallVector<mlir::Type, 3> argsTypes;

	for (const auto& [name, type] : llvm::zip(op.argsNames(), op.getType().getInputs()))
	{
		argsNames.push_back(name.cast<mlir::StringAttr>().getValue());
		argsTypes.push_back(type);
	}

	for (const auto& [name, type] : llvm::zip(newArgsNamesBuffers, newArgsTypes))
	{
		argsNames.push_back(name);
		argsTypes.push_back(type);
	}

	// Determine the result members
	llvm::SmallVector<std::string, 3> resultsNamesBuffers;
	llvm::SmallVector<llvm::StringRef, 3> resultsNames;
	llvm::SmallVector<mlir::Type, 3> resultsTypes;

	for (const auto& [name, type] : llvm::zip(op.resultsNames(), op.getType().getResults()))
	{
		if (!hasFloatBase(type))
			continue;

		resultsTypes.push_back(type);

		llvm::StringRef resultName = name.cast<mlir::StringAttr>().getValue();
		resultsNamesBuffers.push_back(getNextDerVariableName(resultName, order));
	}

	for (const auto& name : resultsNamesBuffers)
		resultsNames.push_back(name);

	// Create the derived function
	auto derivedFunction = builder.create<FunctionOp>(
			op->getLoc(),
			derivativeAttribute.getName(),
			builder.getFunctionType(argsTypes, resultsTypes),
			argsNames,
			resultsNames);

	builder.setInsertionPointToStart(derivedFunction.addEntryBlock());

	// Map the derived function arguments for a faster lookup
	llvm::StringMap<mlir::Value> argsMap;

	for (const auto& [name, value] : llvm::zip(
					 derivedFunction.argsNames(), derivedFunction.getArguments()))
		argsMap[name.cast<mlir::StringAttr>().getValue()] = value;

	llvm::SmallVector<mlir::Value, 3> partialDerivativeCallArgs;

	for (const auto& arg : op.argsNames())
		partialDerivativeCallArgs.push_back(argsMap[arg.cast<mlir::StringAttr>().getValue()]);

	llvm::SmallVector<mlir::Value, 3> derResults;

	for (const auto& [name, type] : llvm::zip(op.argsNames(), op.getType().getInputs()))
	{
		if (!hasFloatBase(type))
			continue;

		auto call = builder.create<CallOp>(
				op.getLoc(),
				getPartialDerivativeName(op.getName(), name.cast<mlir::StringAttr>().getValue()),
				op.getType().getResults(),
				partialDerivativeCallArgs);

		for (const auto& [resultName, callResult] : llvm::zip(op.resultsNames(), call.getResults()))
		{
			auto derName = getNextDerVariableName(name.cast<mlir::StringAttr>().getValue(), order);

			// TODO: change to element-wise multiplication
			mlir::Value mul = builder.create<MulOp>(
								 op.getLoc(), callResult.getType(), callResult,
								 argsMap[derName]);

			derResults.push_back(mul);
		}
	}

	llvm::SmallVector<mlir::Value, 3> results;

	for (size_t i = 0; i < derResults.size(); i += newArgsTypes.size())
	{
		mlir::Value sum = derResults[i];

		for (size_t j = 1; j < newArgsNamesBuffers.size(); ++j)
			sum = builder.create<AddOp>(op->getLoc(), sum.getType(), sum, derResults[i + j]);

		results.push_back(sum);
	}

	builder.create<ReturnOp>(op->getLoc(), results);
	return mlir::success();
}

/*
struct DerFunctionOpPattern : public mlir::OpRewritePattern<DerFunctionOp>
{
	using mlir::OpRewritePattern<DerFunctionOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(DerFunctionOp op, mlir::PatternRewriter& rewriter) const override
	{
		return mlir::success();
	}
};
 */

class AutomaticDifferentiationPass: public mlir::PassWrapper<AutomaticDifferentiationPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
	}

	void runOnOperation() override
	{
		auto module = getOperation();
		module.dump();

		module->walk([](FunctionOp op) {
			if (op.hasDerivative())
				createFunctionDerivative(op);
		});

		module->dump();

		/*
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		target.addDynamicallyLegalOp<DerFunctionOp>([](DerFunctionOp op) {
			// Mark the operation as illegal only if the function to be derived
			// is a standard one. This way, in a chain of partial derivatives one
			// derivation will take place only when all the previous one have
			// been computed.

			auto module = op->getParentOfType<mlir::ModuleOp>();
			auto* derivedFunction = module.lookupSymbol(op.derivedFunction());
			return !mlir::isa<FunctionOp>(derivedFunction);
		});

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<DerFunctionOpPattern>(&getContext());

		if (failed(applyPartialConversion(module, target, std::move(patterns))))
		{
			mlir::emitError(module.getLoc(), "Error during automatic differentiation\n");
			signalPassFailure();
		}
		 */
	}
};

std::unique_ptr<mlir::Pass> modelica::codegen::createAutomaticDifferentiationPass()
{
	return std::make_unique<AutomaticDifferentiationPass>();
}
