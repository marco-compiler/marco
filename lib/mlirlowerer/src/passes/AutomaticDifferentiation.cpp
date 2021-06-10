#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/AutomaticDifferentiation.h>
#include <stack>

using namespace modelica::codegen;

static bool hasFloatBase(mlir::Type type) {
	if (type.isa<RealType>())
		return true;

	if (auto pointerType = type.dyn_cast<PointerType>();
			pointerType && pointerType.getElementType().isa<RealType>())
		return true;

	return false;
}

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
		return getDerVariableName(currentName.substr(4), currentOrder);

	return getDerVariableName(currentName.substr(5 + numDigits(currentOrder - 1)), currentOrder);
}

static mlir::LogicalResult createPartialDerFunction(FunctionOp base, llvm::StringRef derivativeName, llvm::StringRef independentVar)
{
	mlir::OpBuilder builder(base);
	auto module = base->getParentOfType<mlir::ModuleOp>();

	llvm::SmallVector<llvm::StringRef, 3> argsNames;
	llvm::SmallVector<llvm::StringRef, 3> resultsNames;

	for (const auto& argName : base.argsNames())
		argsNames.push_back(argName.cast<mlir::StringAttr>().getValue());

	for (const auto& argName : base.resultsNames())
		resultsNames.push_back(argName.cast<mlir::StringAttr>().getValue());

	// Determine how many dimensions should be added to the results.
	// In fact, if the derivation is done with respect to an array argument,
	// then the result should become an array in which every index stores
	// the partial derivative with respect to that argument array index.

	llvm::SmallVector<long, 3> resultDimensions;

	{
		auto argIndex = [&](llvm::StringRef name) -> llvm::Optional<size_t> {
			for (const auto& argName : llvm::enumerate(base.argsNames()))
				if (argName.value().cast<mlir::StringAttr>().getValue() == name)
					return argName.index();

			return llvm::None;
		};

		auto argType = [&](llvm::StringRef name) -> llvm::Optional<mlir::Type> {
			return argIndex(name).map([&](size_t index) {
				return base.getType().getInput(index);
			});
		};

		auto independentArgType = argType(independentVar);

		if (!independentArgType.hasValue())
			return mlir::failure();

		if (auto pointerType = independentArgType->dyn_cast<PointerType>())
			for (const auto& dimension : pointerType.getShape())
				resultDimensions.push_back(dimension);
	}

	bool isVectorized = !resultDimensions.empty();

	// Determine the results types. If the derivation is done with respect to
	// a scalar variable, then the results types will be the same as the original
	// function.

	llvm::SmallVector<mlir::Type, 3> resultsTypes;

	for (const auto& type : base.getType().getResults())
	{
		if (isVectorized)
		{
			llvm::SmallVector<long, 3> dimensions(
					resultDimensions.begin(), resultDimensions.end());

			if (auto pointerType = type.dyn_cast<PointerType>())
			{
				for (auto dimension : pointerType.getShape())
					dimensions.push_back(dimension);

				resultsTypes.push_back(PointerType::get(
						type.getContext(),
						pointerType.getAllocationScope(),
						pointerType.getElementType(),
						dimensions));
			}
			else
			{
				resultsTypes.push_back(PointerType::get(
						type.getContext(), BufferAllocationScope::heap, type, dimensions));
			}
		}
		else
		{
			resultsTypes.push_back(type);
		}
	}

	auto function = builder.create<FunctionOp>(
			base.getLoc(), derivativeName,
			builder.getFunctionType(base.getType().getInputs(), resultsTypes),
			argsNames, resultsNames);

	mlir::BlockAndValueMapping mapping;

	// Clone the blocks structure of the function to be derived. The
	// operations contained in the blocks are not copied.

	for (auto& block : base.getRegion().getBlocks())
	{
		mlir::Block* clonedBlock = builder.createBlock(
				&function.getRegion(),
				function.getRegion().end(),
				block.getArgumentTypes());

		mapping.map(&block, clonedBlock);
	}

	mlir::BlockAndValueMapping derivatives;
	builder.setInsertionPointToStart(&function.getRegion().front());

	// Iterate over the original operations and create their derivatives
	// (if possible) inside the new function.

	llvm::SmallVector<DerivativeInterface> derivableOps;

	for (auto& sourceBlock : llvm::enumerate(base.getBody().getBlocks()))
	{
		auto& block = *std::next(function.getBlocks().begin(), sourceBlock.index());
		builder.setInsertionPointToStart(&block);

		// Map the original block arguments to the new block ones
		for (const auto& arg : llvm::enumerate(sourceBlock.value().getArguments()))
			mapping.map(arg.value(), block.getArgument(arg.index()));

		for (auto& baseOp : sourceBlock.value().getOperations())
			builder.clone(baseOp, mapping);
	}

	function.walk([&](mlir::Operation* op) {
		if (auto deriveInterface = mlir::dyn_cast<DerivativeInterface>(op))
			derivableOps.push_back(deriveInterface);
	});

	builder.setInsertionPointToStart(&function.getRegion().front());

	// Create the arguments derivatives
	for (const auto& [name, value, type] : llvm::zip(argsNames, function.getArguments(), function.getType().getInputs()))
	{
		auto memberType = type.isa<PointerType>() ?
											MemberType::get(type.cast<PointerType>()) :
											MemberType::get(builder.getContext(), MemberAllocationScope::stack, type);

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

		if (auto pointerType = type.dyn_cast<PointerType>())
			for (const auto& dimension : llvm::enumerate(pointerType.getShape()))
				if (dimension.value() == -1)
					dynamicDimensions.push_back(builder.create<DimOp>(
							value.getLoc(), value, builder.create<ConstantOp>(value.getLoc(), builder.getIndexAttr(dimension.index()))));

		mlir::Value der = builder.create<MemberCreateOp>(base.getLoc(), name, memberType, dynamicDimensions);
		der = builder.create<MemberLoadOp>(base->getLoc(), type, der);
		derivatives.map(value, der);

		//mlir::Value seed = [&independentVar, &builder](llvm::StringRef name) {

		//};
	}

	for (auto& op : derivableOps)
	{
		builder.setInsertionPoint(op);
		op.derive(builder, derivatives);
	}

	return mlir::success();
}

static void mapDerivatives(llvm::ArrayRef<llvm::StringRef> names,
													 mlir::ValueRange values,
													 mlir::BlockAndValueMapping& derivatives,
													 unsigned int maxOrder)
{
	// Map the values for a faster by-name lookup
	llvm::StringMap<mlir::Value> map;

	for (const auto& [name, value] : llvm::zip(names, values))
		map[name] = value;

	for (const auto& name : names)
	{
		// Given a variable "x", first search for "der_x". If it doesn't exist,
		// then also "der_2_x", "der_3_x", etc. will not exist and thus we can
		// say that "x" has no derivatives. If it exist, add the first order
		// derivative and then search for the other orders ones.

		auto candidateFirstOrderDer = getDerVariableName(name, 1);

		if (map.count(candidateFirstOrderDer) == 0)
			continue;

		mlir::Value der = map[candidateFirstOrderDer];
		derivatives.map(map[name], der);

		for (unsigned int i = 2; i <= maxOrder; ++i)
		{
			auto nextName = getDerVariableName(name, i);
			assert(map.count(nextName) != 0);
			mlir::Value nextDer = map[nextName];
			derivatives.map(der, nextDer);
			der = nextDer;
		}
	}
}

static mlir::LogicalResult createFullDerFunction(FunctionOp base)
{
	mlir::OpBuilder builder(base);
	auto module = base->getParentOfType<mlir::ModuleOp>();
	auto derivativeAttribute = base->getAttrOfType<DerivativeAttribute>("derivative");
	unsigned int order = derivativeAttribute.getOrder();

	// If the source already provides the derivative function, then stop
	if (module.lookupSymbol<FunctionOp>(derivativeAttribute.getName()))
		return mlir::success();

	// Create a map of the source arguments for a faster lookup
	llvm::StringMap<mlir::Value> sourceArgsMap;

	for (const auto& [name, value] : llvm::zip(base.argsNames(), base.getArguments()))
		sourceArgsMap[name.cast<mlir::StringAttr>().getValue()] = value;

	// The new arguments names are stored here because they will be appended
	// only when all the original arguments will have been processed. Note that
	// the name container stores std::strings instead of llvm::StringRefs, as the
	// latter would refer to a std::string allocated within the following loop
	// (and thus the references would then become invalid).
	llvm::SmallVector<std::string, 3> newArgsNamesBuffers;

	llvm::SmallVector<mlir::Type, 3> newArgsTypes;

	for (const auto& [name, type] : llvm::zip(base.argsNames(), base.getType().getInputs()))
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

	for (const auto& [name, type] : llvm::zip(base.argsNames(), base.getType().getInputs()))
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

	for (const auto& [name, type] : llvm::zip(base.resultsNames(), base.getType().getResults()))
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
			base->getLoc(),
			derivativeAttribute.getName(),
			builder.getFunctionType(argsTypes, resultsTypes),
			argsNames,
			resultsNames);

	mlir::Block* body = derivedFunction.addEntryBlock();
	builder.setInsertionPointToStart(body);

	mlir::BlockAndValueMapping mapping;

	// Clone the blocks structure of the function to be derived. The
	// operations contained in the blocks are not copied.

	for (auto& block : llvm::enumerate(base.getRegion().getBlocks()))
	{
		if (block.index() == 0)
		{
			// The first source block is ignored, as it has only a subset of the
			// derived function arguments.
			continue;
		}

		mlir::Block* clonedBlock = builder.createBlock(
				&derivedFunction.getRegion(),
				derivedFunction.getRegion().end(),
				block.value().getArgumentTypes());

		mapping.map(&block.value(), clonedBlock);
	}

	for (const auto& sourceArg : llvm::enumerate(base.getRegion().front().getArguments()))
		mapping.map(sourceArg.value(), body->getArgument(sourceArg.index()));

	mlir::BlockAndValueMapping derivatives;

	// Map the derivatives among the function arguments.
	mapDerivatives(argsNames, derivedFunction.getArguments(), derivatives, order);

	// Clone the original operations, which will be interleaved in the
	// resulting derivative function.
	for (auto& sourceBlock : llvm::enumerate(base.getBody().getBlocks()))
	{
		auto& block = *std::next(derivedFunction.getBlocks().begin(), sourceBlock.index());
		builder.setInsertionPointToStart(&block);

		// Map the original block arguments to the new block ones
		for (const auto& arg : llvm::enumerate(sourceBlock.value().getArguments()))
			mapping.map(arg.value(), block.getArgument(arg.index()));

		for (auto& baseOp : sourceBlock.value().getOperations())
			builder.clone(baseOp, mapping);
	}

	// Create the new members derivatives
	builder.setInsertionPointToStart(&derivedFunction.getRegion().front());

	llvm::SmallVector<llvm::StringRef, 3> allMembersNames;
	llvm::SmallVector<mlir::Value, 3> allMembersValues;

	derivedFunction.walk([&](MemberCreateOp op) {
		allMembersNames.push_back(op.name());
		allMembersValues.push_back(op.getResult());
	});

	llvm::SmallVector<mlir::Value, 3> toBeDerived;
	mapDerivatives(allMembersNames, allMembersValues, derivatives, order - 1);

	// Create the new members derivatives
	for (const auto& [name, value] : llvm::zip(allMembersNames, allMembersValues))
	{
		if (derivatives.contains(value))
		{
			derivatives.map(value, derivatives.lookup(value));
		}
		else
		{
			auto createOp = value.getDefiningOp<MemberCreateOp>();
			builder.setInsertionPointAfter(createOp);

			mlir::Value der = builder.create<MemberCreateOp>(
					value.getLoc(),
					getNextDerVariableName(name, order),
					value.getType(), createOp.dynamicDimensions());

			derivatives.map(value, der);
		}
	}

	// Determine the list of the derivable operations. We can't just derive as
	// we find them, as we would invalidate the operation walk's iterator.
	llvm::SmallVector<DerivativeInterface> derivableOps;

	derivedFunction.walk([&](mlir::Operation* op) {
		if (auto derivableOp = mlir::dyn_cast<DerivativeInterface>(op))
			derivableOps.push_back(derivableOp);
	});

	// Derive each derivable operation. The derivative is placed before the
	// old assignment, in order to avoid inconsistencies in case of
	// self-assignments (i.e. "y := y * 2" would invalidate the derivative
	// if placed before "y' := y' * 2").
	for (auto& op : derivableOps)
	{
		builder.setInsertionPoint(op);
		op.derive(builder, derivatives);
	}

	// Replace the old return operation with a new one returning the derivatives
	auto returnOp = mlir::cast<ReturnOp>(derivedFunction.getRegion().back().getTerminator());
	llvm::SmallVector<mlir::Value, 3> results;

	for (auto value : llvm::enumerate(returnOp.values()))
		if (hasFloatBase(base.getType().getResult(value.index())))
			results.push_back(derivatives.lookup(value.value()));

	builder.setInsertionPoint(returnOp);
	builder.create<ReturnOp>(returnOp.getLoc(), results);
	returnOp->erase();

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

		llvm::SmallVector<FunctionOp, 3> toBeDerived;

		module->walk([&](FunctionOp op) {
			if (op.hasDerivative())
				toBeDerived.push_back(op);
		});

		for (auto& function : toBeDerived)
		{
			if (mlir::failed(createFullDerFunction(function)))
			{
				module->dump();
				return signalPassFailure();
			}
		}

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
