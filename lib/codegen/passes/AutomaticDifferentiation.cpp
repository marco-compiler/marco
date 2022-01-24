#include "llvm/ADT/STLExtras.h"
#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/codegen/passes/AutomaticDifferentiation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include <queue>
#include <set>

using namespace marco::codegen;
using namespace modelica;

static bool hasFloatBase(mlir::Type type) {
	if (type.isa<RealType>())
		return true;

	if (auto arrayType = type.dyn_cast<ArrayType>();
			arrayType && arrayType.getElementType().isa<RealType>())
		return true;

	return false;
}

static void getDynamicDimensions(mlir::OpBuilder& builder,
																 mlir::Value value,
																 llvm::SmallVectorImpl<mlir::Value>& dynamicDimensions)
{
	if (auto arrayType = value.getType().dyn_cast<ArrayType>())
	{
		for (const auto& dimension : llvm::enumerate(arrayType.getShape()))
		{
			if (dimension.value() == -1)
			{
				mlir::Value index = builder.create<ConstantOp>(value.getLoc(), builder.getIndexAttr(dimension.index()));
				mlir::Value dim = builder.create<DimOp>(value.getLoc(), value, index);
				dynamicDimensions.push_back(dim);
			}
		}
	}
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

static mlir::Value createDerVariable(
		mlir::OpBuilder& builder,
		mlir::Value var,
		std::function<std::string()> derivativeName)
{
	if (auto memberCreateOp = var.getDefiningOp<MemberCreateOp>())
	{
		mlir::Value der = builder.create<MemberCreateOp>(
				var.getLoc(),
				derivativeName(),
				var.getType(),
				memberCreateOp.dynamicDimensions());

		return der;
	}

	mlir::Type type = var.getType();

	auto memberType = type.isa<ArrayType>() ?
										MemberType::get(type.cast<ArrayType>()) :
										MemberType::get(builder.getContext(), MemberAllocationScope::stack, type);

	llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
	getDynamicDimensions(builder, var, dynamicDimensions);

	mlir::Value der = builder.create<MemberCreateOp>(
			var.getLoc(), derivativeName(), memberType, dynamicDimensions);

	return der;
}

static std::string getPartialDerVariableName(llvm::StringRef currentName, llvm::StringRef independentVar)
{
	return "pder_" + independentVar.str() + "_" + currentName.str();
}

static mlir::LogicalResult createPartialDerFunction(mlir::OpBuilder& builder, DerFunctionOp derFunction)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPointAfter(derFunction);

	mlir::Location loc = derFunction->getLoc();
	auto module = derFunction->getParentOfType<mlir::ModuleOp>();

	llvm::SmallVector<llvm::StringRef, 3> independentVariables;

	for (const auto& independentVariable : derFunction.independentVariables())
		independentVariables.push_back(independentVariable.cast<mlir::StringAttr>().getValue());

	llvm::StringRef independentVar = independentVariables[0];

	llvm::SmallVector<mlir::Attribute, 3> argsNames;
	llvm::SmallVector<mlir::Attribute, 3> resultsNames;

	auto base = module.lookupSymbol<FunctionOp>(derFunction.derivedFunction());

	// The arguments remain the same, and so their names
	for (const auto& argName : base.argsNames())
		argsNames.push_back(argName);

	// The results are not the same anymore, as they are replaced by the
	// partial derivatives.

	for (const auto& argName : base.resultsNames())
	{
		std::string derivativeName = argName.cast<mlir::StringAttr>().getValue().str();

		for (const auto& independentVariable : independentVariables)
			derivativeName = getPartialDerVariableName(derivativeName, independentVariable);

		resultsNames.push_back(builder.getStringAttr(derivativeName));
	}

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

		if (auto arrayType = independentArgType->dyn_cast<ArrayType>())
			for (const auto& dimension : arrayType.getShape())
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

			if (auto arrayType = type.dyn_cast<ArrayType>())
			{
				for (auto dimension : arrayType.getShape())
					dimensions.push_back(dimension);

				resultsTypes.push_back(ArrayType::get(
						type.getContext(),
						arrayType.getAllocationScope(),
						arrayType.getElementType(),
						dimensions));
			}
			else
			{
				resultsTypes.push_back(ArrayType::get(
						type.getContext(), BufferAllocationScope::heap, type, dimensions));
			}
		}
		else
		{
			resultsTypes.push_back(type);
		}
	}

	auto derivedFunction = builder.create<FunctionOp>(
			loc, derFunction.name(),
			builder.getFunctionType(base.getType().getInputs(), resultsTypes),
			builder.getArrayAttr(argsNames),
			builder.getArrayAttr(resultsNames));

	mlir::BlockAndValueMapping mapping;

	// Clone the blocks structure of the function to be derived. The
	// operations contained in the blocks are not copied.

	for (auto& block : base.getRegion().getBlocks())
	{
		mlir::Block* clonedBlock = builder.createBlock(
				&derivedFunction.getBody(),
				derivedFunction.getBody().end(),
				block.getArgumentTypes());

		mapping.map(&block, clonedBlock);
	}

	builder.setInsertionPointToStart(&derivedFunction.getBody().front());

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

	// List of the operations to be derived
	std::set<mlir::Operation*> derivedOperations;
	std::queue<DerivativeInterface> derivableOps;
	std::set<mlir::Operation*> notToBeDerivedOps;

	// Create the members derivatives
	builder.setInsertionPointToStart(&derivedFunction.getBody().front());
	llvm::StringMap<mlir::BlockAndValueMapping> derivatives;

	// The members are mapped before creating the derivatives of the arguments,
	// as they would introduce new member operations that would be considered
	// twice.
	llvm::StringMap<mlir::Value> membersMap;

	derivedFunction.walk([&](MemberCreateOp op) {
		membersMap[op.name()] = op.getResult();
	});

	// Utility function to create the derivative of a member.
	// The callback function will receive the original variable, the new one
	// representing the derivative, and the independent variable name with
	// respect to which the derivative has been created.

	using varDerFnType = std::function<void(mlir::Value, mlir::Value, llvm::StringRef)>;

	auto createMemberDerivativeFn = [&](mlir::Value value,
																			llvm::StringRef name,
																			varDerFnType onDerivativeCreatedCallback = nullptr)
	{
		llvm::SmallVector<mlir::Value, 8> variables;
		llvm::SmallVector<std::string, 8> names;

		variables.push_back(value);
		names.push_back(name.str());

		// Each independent variable will lead to a new derivative for each
		// existing member up to that point.

		for (const auto& independentVariable : independentVariables)
		{
			llvm::SmallVector<mlir::Value, 3> newVariables;
			llvm::SmallVector<std::string, 3> newNames;

			assert(variables.size() == names.size());

			for (const auto& [var, name] : llvm::zip(variables, names))
			{
				// Create the new derivative variable
				auto derivativeName = getPartialDerVariableName(name, independentVariable);

				mlir::Value derVar = createDerVariable(builder, value, [&derivativeName]() {
					return derivativeName;
				});

				newVariables.push_back(derVar);
				newNames.push_back(derivativeName);

				// Create the seed
				builder.create<DerSeedOp>(loc, derVar, name == independentVariable ? 1 : 0);

				// Invoke the callback so that additional operations can be done on
				// the just created derivative.

				if (onDerivativeCreatedCallback != nullptr)
					onDerivativeCreatedCallback(var, derVar, independentVariable);
			}

			variables.append(newVariables);
			names.append(newNames);
		}
	};

	// Create the derivatives of the arguments of the function.
	for (const auto& [name, value, type] : llvm::zip(argsNames, derivedFunction.getArguments(), derivedFunction.getType().getInputs()))
	{
		createMemberDerivativeFn(
				value,
				name.cast<mlir::StringAttr>().getValue(),
				[&](mlir::Value var, mlir::Value derVar, llvm::StringRef independentVariable) {
					// Input arguments should not be mapped to the member itself, but rather
					// to the value (seed) they contain. This way, existing operations that
					// refer to the input arguments don't have to check whether to load
					// or not the value from the member. This is possible also because
					// input arguments get never written as per the Modelica standard.

					assert(derVar.getType().isa<MemberType>());
					mlir::Type type = derVar.getType().cast<MemberType>().unwrap();
					auto seed = builder.create<MemberLoadOp>(base->getLoc(), type, derVar);
					derivatives[independentVariable].map(var, seed.getResult());

					// The load operation is created just to provide access to the seed,
					// and thus should not be derived.
					notToBeDerivedOps.insert(seed.getOperation());
				});
	}

	// Create the derivatives of the other members.
	for (const auto& member : membersMap)
	{
		createMemberDerivativeFn(
				member.getValue(),
				member.getKey(),
				[&](mlir::Value var, mlir::Value derVar, llvm::StringRef independentVariable) {
					derivatives[independentVariable].map(var, derVar);
				});
	}

	// Utility function that determines whether an operation should be derived
	// with respect to a given independent variable. Multiple factors are taken
	// into account, such as if it has already been derived

	auto shouldOperationBeDerived = [&](mlir::Operation* op, llvm::StringRef independentVariable) {
		if (auto derivativeInterface = mlir::dyn_cast<DerivativeInterface>(op))
		{
			if (derivedOperations.count(derivativeInterface.getOperation()) != 0 ||
					notToBeDerivedOps.count(derivativeInterface.getOperation()) != 0)
				return false;

			// If an operation needs the derivative of a block argument, then
			// the argument is either an input to the function (which has
			// already been derived) or a loop argument. In this last case,
			// the argument is not derived and thus also the operation is not.

			llvm::SmallVector<mlir::Value, 3> operandsToBeDerived;
			derivativeInterface.getOperandsToBeDerived(operandsToBeDerived);

			for (const auto& arg : operandsToBeDerived)
				if (arg.isa<mlir::BlockArgument>() &&
						!derivatives[independentVariable].contains(arg))
					return false;

			return true;
		}

		return false;
	};

	for (const auto& independentVariable : independentVariables)
	{
		for (auto &region : derivedFunction->getRegions())
			for (auto& block : region)
				for (auto& nestedOp : llvm::make_early_inc_range(block))
					if (shouldOperationBeDerived(&nestedOp, independentVariable))
						derivableOps.push(&nestedOp);

		while (!derivableOps.empty())
		{
			auto& op = derivableOps.front();

			builder.setInsertionPoint(op);
			op.derive(builder, derivatives[independentVariable]);
			derivedOperations.insert(op.getOperation());

			llvm::SmallVector<mlir::Region*, 3> regions;
			op.getDerivableRegions(regions);

			for (auto& region : regions)
				for (auto nestedOp : region->getOps<DerivativeInterface>())
					if (shouldOperationBeDerived(nestedOp, independentVariable))
						derivableOps.push(nestedOp);

			derivableOps.pop();
		}
	}

	return mlir::success();
}

struct DerSeedOpPattern : public mlir::OpRewritePattern<DerSeedOp>
{
	using mlir::OpRewritePattern<DerSeedOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(DerSeedOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		auto memberType = op.member().getType().cast<MemberType>();
		auto arrayType = memberType.toArrayType();

		// TODO To be reconsidered when the derivation with respect to arrays will be supported

		mlir::Value seed = rewriter.create<ConstantOp>(loc, RealAttribute::get(op.getContext(), op.value()));

		if (arrayType.isScalar())
		{
			rewriter.create<MemberStoreOp>(loc, op.member(), seed);
		}
		else
		{
			auto memberCreateOp = op.member().getDefiningOp<MemberCreateOp>();
			auto buffer = rewriter.create<AllocaOp>(loc, arrayType.getElementType(), arrayType.getShape(), memberCreateOp.dynamicDimensions());
			rewriter.create<FillOp>(loc, seed, buffer);
			rewriter.create<MemberStoreOp>(loc, op.member(), buffer);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

/**
 * Compose the full derivative member name according to the derivative order.
 * If the order is 1, then it is omitted.
 *
 * @param variableName 	base variable name
 * @param order 				derivative order
 * @return derived variable name
 */
static std::string getFullDerVariableName(llvm::StringRef baseName, unsigned int order)
{
	assert(order > 0);

	if (order == 1)
		return "der_" + baseName.str();

	return "der_" + std::to_string(order) + "_" + baseName.str();
}

/**
 * Given a full derivative variable name of order n, compose the name of the
 * n + 1 variable order.
 *
 * @param currentName 	current variable name
 * @param currentOrder  current order
 * @return next order derived variable name
 */
static std::string getNextFullDerVariableName(llvm::StringRef currentName, unsigned int requestedOrder)
{
	if (requestedOrder == 1)
		return getFullDerVariableName(currentName, requestedOrder);

	assert(currentName.rfind("der_") == 0);

	if (requestedOrder == 2)
		return getFullDerVariableName(currentName.substr(4), requestedOrder);

	return getFullDerVariableName(currentName.substr(5 + numDigits(requestedOrder - 1)), requestedOrder);
}

static void mapFullDerivatives(llvm::ArrayRef<llvm::StringRef> names,
															 mlir::ValueRange values,
															 mlir::BlockAndValueMapping& derivatives)
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

		auto candidateFirstOrderDer = getFullDerVariableName(name, 1);

		if (map.count(candidateFirstOrderDer) == 0)
			continue;

		mlir::Value der = map[candidateFirstOrderDer];
		derivatives.map(map[name], der);

		unsigned int order = 2;
		bool found = true;

		while (found)
		{
			auto nextName = getFullDerVariableName(name, order);
			found = map.count(nextName) != 0;

			if (found)
			{
				mlir::Value nextDer = map[nextName];
				derivatives.map(der, nextDer);
				der = nextDer;
			}

			++order;
		}
	}
}

static mlir::LogicalResult createFullDerFunction(mlir::OpBuilder& builder, FunctionOp base)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPointAfter(base);

	auto derivativeAttribute = base->getAttrOfType<DerivativeAttribute>("derivative");
	unsigned int order = derivativeAttribute.getOrder();

	// If the source already provides the derivative function, then stop
	if (auto module = base->getParentOfType<mlir::ModuleOp>();
			module.lookupSymbol<FunctionOp>(derivativeAttribute.getName()))
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
							if (name == getFullDerVariableName(arg.first(), i))
								return true;
					}

					return false;
				};

				if (isDerivative(argName))
					continue;
			}

			newArgsNamesBuffers.push_back(getFullDerVariableName(argName, order));
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
		resultsNamesBuffers.push_back(getNextFullDerVariableName(resultName, order));
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

	// Map the derivatives among the function arguments
	mapFullDerivatives(argsNames, derivedFunction.getArguments(), derivatives);

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
	mapFullDerivatives(allMembersNames, allMembersValues, derivatives);

	// Create the new members derivatives
	for (const auto& [name, value] : llvm::zip(allMembersNames, allMembersValues))
	{
		if (derivatives.contains(value))
		{
			derivatives.map(value, derivatives.lookup(value));
		}
		else
		{
			builder.setInsertionPointAfterValue(value);

			mlir::Value der = createDerVariable(builder, value, [name = std::ref(name), &order]() {
				return getNextFullDerVariableName(name, order);
			});

			derivatives.map(value, der);
		}
	}

	// Determine the list of the derivable operations. We can't just derive as
	// we find them, as we would invalidate the operation walk's iterator.
	std::queue<DerivativeInterface> derivableOps;

	for (auto derivableOp : derivedFunction.getBody().getOps<DerivativeInterface>())
		derivableOps.push(derivableOp);

	// Derive each derivable operation
	while (!derivableOps.empty())
	{
		auto& op = derivableOps.front();

		op.derive(builder, derivatives);

		llvm::SmallVector<mlir::Region*, 3> regions;
		op.getDerivableRegions(regions);

		for (auto& region : regions)
			for (auto derivableOp : region->getOps<DerivativeInterface>())
				derivableOps.push(derivableOp);

		derivableOps.pop();
	}

	return mlir::success();
}

class AutomaticDifferentiationPass: public mlir::PassWrapper<AutomaticDifferentiationPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
	}

	void runOnOperation() override
	{
		if (mlir::failed(createFullDerFunctions()))
		{
			mlir::emitError(getOperation().getLoc(), "Error in creating the functions full derivatives");
			return signalPassFailure();
		}

		if (mlir::failed(createPartialDerFunctions()))
		{
			mlir::emitError(getOperation().getLoc(), "Error in creating the functions partial derivatives");
			return signalPassFailure();
		}

		if (mlir::failed(resolveTrivialDerCalls()))
		{
			mlir::emitError(getOperation().getLoc(), "Error in resolving the trivial derivative calls");
			return signalPassFailure();
		}
	}

	mlir::LogicalResult createFullDerFunctions()
	{
		auto module = getOperation();
		mlir::OpBuilder builder(module);

		llvm::SmallVector<FunctionOp, 3> toBeDerived;

		module->walk([&](FunctionOp op) {
			if (op.hasDerivative())
				toBeDerived.push_back(op);
		});

		// Sort the functions so that a function derivative is computed only
		// when the base function already has its body determined.

		llvm::sort(toBeDerived, [](FunctionOp first, FunctionOp second) {
			auto annotation = first->getAttrOfType<DerivativeAttribute>("derivative");
			return annotation.getName() == second.name();
		});

		for (auto& function : toBeDerived)
			if (auto status = createFullDerFunction(builder, function); mlir::failed(status))
				return status;

		return mlir::success();
	}

	mlir::LogicalResult createPartialDerFunctions()
	{
		auto module = getOperation();
		mlir::OpBuilder builder(module);

		llvm::SmallVector<DerFunctionOp, 3> toBeProcessed;

		// The conversion is done in an iterative way, because new derivative
		// functions may be created while converting the existing one (i.e. when
		// a function to be derived contains a call to an another function).

		auto findDerFunctions = [&]() -> bool {
			module->walk([&](DerFunctionOp op) {
				toBeProcessed.push_back(op);
			});

			return !toBeProcessed.empty();
		};

		while (findDerFunctions())
		{
			// Sort the functions so that a function derivative is computed only
			// when the base function already has its body determined.

			llvm::sort(toBeProcessed, [](DerFunctionOp first, DerFunctionOp second) {
				return first.name() == second.derivedFunction();
			});

			for (auto& function : toBeProcessed)
			{
				if (auto status = createPartialDerFunction(builder, function); mlir::failed(status))
					return status;

				function->erase();
			}

			toBeProcessed.clear();
		}

		// Convert the seed operations
		mlir::ConversionTarget target(getContext());
		target.addIllegalOp<DerSeedOp>();
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<DerSeedOpPattern>(&getContext());

		return applyFullConversion(module, target, std::move(patterns));
	}

	mlir::LogicalResult resolveTrivialDerCalls()
	{
		auto module = getOperation();
		mlir::OpBuilder builder(module);

		module.walk([&](DerOp op) {
			mlir::Value operand = op.operand();
			mlir::Operation* definingOp = operand.getDefiningOp();

			if (definingOp == nullptr)
				return;

			if (auto derivableOp = mlir::dyn_cast<DerivativeInterface>(definingOp))
			{
				auto classOp = op->getParentOfType<ClassInterface>();

				if (classOp == nullptr)
					return;

				llvm::SmallVector<mlir::Value, 3> members;
				llvm::SmallVector<llvm::StringRef, 3> names;
				classOp.getMembers(members, names);

				mlir::BlockAndValueMapping derivatives;
				mapFullDerivatives(names, members, derivatives);

				mlir::ValueRange ders = derivableOp.deriveTree(builder, derivatives);

				if (ders.size() != op->getNumResults())
					return;

				op->replaceAllUsesWith(ders);
				op.erase();
			}
		});

		return mlir::success();
	}
};

std::unique_ptr<mlir::Pass> marco::codegen::createAutomaticDifferentiationPass()
{
	return std::make_unique<AutomaticDifferentiationPass>();
}
