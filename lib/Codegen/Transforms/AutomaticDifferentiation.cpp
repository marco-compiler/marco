#include "marco/Codegen/Transforms/AutomaticDifferentiation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include <queue>
#include <set>

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  template<class T>
  unsigned int numDigits(T number)
  {
    unsigned int digits = 0;

    while (number != 0) {
      number /= 10;
      ++digits;
    }

    return digits;
  }
}

namespace marco::codegen
{
  std::string getFullDerVariableName(llvm::StringRef baseName, unsigned int order)
  {
    assert(order > 0);

    if (order == 1) {
      return "der_" + baseName.str();
    }

    return "der_" + std::to_string(order) + "_" + baseName.str();
  }

  std::string getNextFullDerVariableName(llvm::StringRef currentName, unsigned int requestedOrder)
  {
    if (requestedOrder == 1) {
      return getFullDerVariableName(currentName, requestedOrder);
    }

    assert(currentName.rfind("der_") == 0);

    if (requestedOrder == 2) {
      return getFullDerVariableName(currentName.substr(4), requestedOrder);
    }

    return getFullDerVariableName(currentName.substr(5 + numDigits(requestedOrder - 1)), requestedOrder);
  }
}

static std::string getPartialDerVariableName(llvm::StringRef currentName, llvm::StringRef independentVar)
{
  return "pder_" + independentVar.str() + "_" + currentName.str();
}

static std::string getPartialDerFunctionName(llvm::StringRef baseName)
{
  return "pder_" + baseName.str();
}

static std::string getPartialDerMemberName(llvm::StringRef baseName, unsigned int order)
{
  return "pder_" + std::to_string(order) + "_" + baseName.str();
}

static std::string getPartialDerTemplateFunctionName(llvm::StringRef baseName)
{
  return "pder_" + baseName.str();
}

static llvm::StringMap<mlir::Value> mapMembersByName(FunctionOp functionOp)
{
  llvm::StringMap<mlir::Value> result;

  functionOp->walk([&](MemberCreateOp op) {
    result[op.name()] = op.getResult();
  });

  return result;
}

static bool hasFloatBase(mlir::Type type) {
	if (type.isa<RealType>()) {
    return true;
  }

	if (auto arrayType = type.dyn_cast<ArrayType>(); arrayType && arrayType.getElementType().isa<RealType>()) {
    return true;
  }

	return false;
}

/// Map each partial derivative function to its base function
using PartialDerTemplateFunctionsMap = llvm::StringMap<FunctionOp>;

static std::pair<FunctionOp, unsigned int> getPartialDerBaseFunction(
    const PartialDerTemplateFunctionsMap& derFunctionsMap, FunctionOp functionOp)
{
  unsigned int order = 0;
  FunctionOp baseFunction = functionOp;

  while (derFunctionsMap.find(baseFunction.name()) != derFunctionsMap.end()) {
    ++order;
    baseFunction = derFunctionsMap.lookup(baseFunction.name());
  }

  return std::make_pair(baseFunction, order);
}

static mlir::LogicalResult createPartialDerFunction(mlir::OpBuilder& builder, DerFunctionOp derFunctionOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(derFunctionOp);

  auto loc = derFunctionOp.getLoc();
  auto module = derFunctionOp->getParentOfType<mlir::ModuleOp>();
  auto baseFunctionOp = module.lookupSymbol<FunctionOp>(derFunctionOp.derived_function());

  // Map the members for a faster lookup
  auto originalMembersMap = mapMembersByName(baseFunctionOp);

  // Determine the function signature
  llvm::SmallVector<mlir::Type, 6> argsTypes;
  llvm::SmallVector<mlir::Type, 6> resultsTypes;

  // The original arguments
  for (const auto& type : baseFunctionOp.getType().getInputs()) {
    argsTypes.push_back(type);
  }

  // The original results
  for (const auto& type : baseFunctionOp.getType().getResults()) {
    resultsTypes.push_back(type);
  }

  // Create the derived function
  auto derivedFunctionOp = builder.create<FunctionOp>(
      derFunctionOp.getLoc(),
      derFunctionOp.name(),
      builder.getFunctionType(argsTypes, resultsTypes));

  // Start the body of the function
  mlir::Block* entryBlock = builder.createBlock(&derivedFunctionOp.body());
  builder.setInsertionPointToStart(entryBlock);

  // Create the input members
  llvm::SmallVector<mlir::Value, 3> inputMembers;

  for (const auto& name : baseFunctionOp.inputMemberNames()) {
    auto originalMemberOp = originalMembersMap[name].getDefiningOp<MemberCreateOp>();

    auto memberOp = builder.create<MemberCreateOp>(
        originalMemberOp.getLoc(), name, originalMemberOp.getMemberType(), llvm::None);

    inputMembers.push_back(memberOp.getResult());
  }

  // Create the output members
  llvm::SmallVector<mlir::Value, 3> outputMembers;

  for (const auto& name : baseFunctionOp.outputMemberNames()) {
    auto originalMemberOp = originalMembersMap[name].getDefiningOp<MemberCreateOp>();

    auto memberOp = builder.create<MemberCreateOp>(
        originalMemberOp.getLoc(), name, originalMemberOp.getMemberType(), llvm::None);

    outputMembers.push_back(memberOp.getResult());
  }

  // Call the template function
  llvm::SmallVector<mlir::Value, 6> args;

  for (auto inputMember : inputMembers) {
    args.push_back(builder.create<MemberLoadOp>(loc, inputMember));
  }

  for (const auto& name : baseFunctionOp.inputMemberNames()) {
    auto originalMemberOp = originalMembersMap[name].getDefiningOp<MemberCreateOp>();
    assert(!originalMemberOp.getMemberType().unwrap().isa<ArrayType>());

    float seed = name == derFunctionOp.independent_vars()[0].cast<mlir::StringAttr>().getValue() ? 1 : 0;
    args.push_back(builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), seed)));
  }

  auto callOp = builder.create<CallOp>(
      loc,
      getPartialDerTemplateFunctionName(derFunctionOp.name()),
      resultsTypes, args);

  assert(callOp->getNumResults() == outputMembers.size());

  for (auto [outputMember, result] : llvm::zip(outputMembers, callOp->getResults())) {
    builder.create<MemberStoreOp>(loc, outputMember, result);
  }

  return mlir::success();
}

static mlir::LogicalResult createPartialDerFunctionOld(mlir::OpBuilder& builder, DerFunctionOp derFunction)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPointAfter(derFunction);

	auto loc = derFunction->getLoc();
	auto module = derFunction->getParentOfType<mlir::ModuleOp>();

	llvm::SmallVector<llvm::StringRef, 3> independentVariables;

	for (const auto& independentVariable : derFunction.independent_vars()) {
    independentVariables.push_back(independentVariable.cast<mlir::StringAttr>().getValue());
  }

	llvm::StringRef independentVar = independentVariables[0];

	llvm::SmallVector<mlir::Attribute, 3> argsNames;
	llvm::SmallVector<mlir::Attribute, 3> resultsNames;

	auto base = module.lookupSymbol<FunctionOp>(derFunction.derived_function());

  /*
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
						type.getContext(), ArrayAllocationScope::heap, type, dimensions));
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
	std::queue<DerivableOpInterface> derivableOps;
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
		if (auto derivativeInterface = mlir::dyn_cast<DerivableOpInterface>(op))
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
				for (auto nestedOp : region->getOps<DerivableOpInterface>())
					if (shouldOperationBeDerived(nestedOp, independentVariable))
						derivableOps.push(nestedOp);

			derivableOps.pop();
		}
	}
   */

	return mlir::success();
}

static void mapFullDerivatives(mlir::BlockAndValueMapping& mapping, llvm::ArrayRef<mlir::Value> members)
{
  llvm::StringMap<mlir::Value> membersByName;

  for (const auto& member : members) {
    auto memberOp = member.getDefiningOp<MemberCreateOp>();
    membersByName[memberOp.name()] = member;
  }

  for (const auto& member : members) {
    auto name = member.getDefiningOp<MemberCreateOp>().name();

    // Given a variable "x", first search for "der_x". If it doesn't exist,
    // then also "der_2_x", "der_3_x", etc. will not exist and thus we can
    // say that "x" has no derivatives. If it exists, add the first order
    // derivative and then search for the higher order ones.

    auto candidateFirstOrderDer = getFullDerVariableName(name, 1);
    auto derIt = membersByName.find(candidateFirstOrderDer);

    if (derIt == membersByName.end()) {
      continue;
    }

    mlir::Value der = derIt->second;
    mapping.map(member, der);

    unsigned int order = 2;
    bool found;

    do {
      auto nextName = getFullDerVariableName(name, order);
      auto nextDerIt = membersByName.find(nextName);
      found = nextDerIt != membersByName.end();

      if (found) {
        mlir::Value nextDer = nextDerIt->second;
        mapping.map(der, nextDer);
        der = nextDer;
      }

      ++order;
    } while (found);
  }
}

static bool isFullDerivative(llvm::StringRef name, FunctionOp originalFunction, unsigned int maxOrder) {
  if (maxOrder == 0) {
    return false;
  }

  // If the current argument name starts with der, we need to check if
  // the original function to be derived has a member whose derivative
  // may be the current one. If this is the case, then we don't need to
  // add the n-th derivative as it is already done when encountering that
  // member. If it is not, then it means the original function had a
  // "strange" member named "der_something" and the derivative function
  // will contain both "der_something" and "der_der_something"; the
  // original "der_something" could effectively be a derivative, but
  // this is an assumption we can't make.

  if (name.rfind("der_") == 0) {
    for (mlir::Value member : originalFunction.getMembers()) {
      auto originalMemberName = member.getDefiningOp<MemberCreateOp>().name();

      for (unsigned int i = 1; i <= maxOrder; ++i) {
        if (name == getFullDerVariableName(originalMemberName, i)) {
          return true;
        }
      }
    }
  }

  return false;
}

static mlir::ValueRange deriveTree(mlir::OpBuilder& builder, DerivableOpInterface op, mlir::BlockAndValueMapping& derivatives)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op.getOperation());

  llvm::SmallVector<mlir::Value, 3> toBeDerived;
  op.getOperandsToBeDerived(toBeDerived);

  for (mlir::Value operand : toBeDerived) {
    if (!derivatives.contains(operand)) {
      mlir::Operation* definingOp = operand.getDefiningOp();

      if (definingOp == nullptr) {
        return llvm::None;
      }

      if (!mlir::isa<DerivableOpInterface>(definingOp)) {
        return llvm::None;
      }
    }
  }

  for (mlir::Value operand : toBeDerived) {
    mlir::Operation* definingOp = operand.getDefiningOp();

    if (definingOp == nullptr) {
      continue;
    }

    if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(definingOp)) {
      auto derivedValues = deriveTree(builder, derivableOp, derivatives);

      if (derivedValues.size() != derivableOp->getNumResults()) {
        return llvm::None;
      }

      for (const auto& [base, derived] : llvm::zip(derivableOp->getResults(), derivedValues)) {
        derivatives.map(base, derived);
      }
    }
  }

  return op.derive(builder, derivatives);
}

namespace
{
  class ForwardAD
  {
    public:
      /// Check if an operation has already been already derived.
      bool isDerived(mlir::Operation* op) const;

      /// Set an operation as already derived.
      void setAsDerived(mlir::Operation* op);

      mlir::LogicalResult createFullDerFunction(
          mlir::OpBuilder& builder, FunctionOp functionOp);

      mlir::LogicalResult createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          PartialDerTemplateFunctionsMap& derFunctionsMap,
          DerFunctionOp derFunctionOp);

      FunctionOp createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          const PartialDerTemplateFunctionsMap& derFunctionsMap,
          mlir::Location loc,
          FunctionOp functionOp,
          llvm::StringRef derivedFunctionName);

      mlir::LogicalResult deriveFunctionBody(
          mlir::OpBuilder& builder,
          FunctionOp functionOp,
          mlir::BlockAndValueMapping& derivatives);

    private:
      // Keeps track of the operations that have already been derived
      std::set<mlir::Operation*> derivedOps;
  };
}

bool ForwardAD::isDerived(mlir::Operation* op) const
{
  return derivedOps.find(op) != derivedOps.end();
}

void ForwardAD::setAsDerived(mlir::Operation* op)
{
  derivedOps.insert(op);
}

mlir::LogicalResult ForwardAD::createFullDerFunction(
    mlir::OpBuilder& builder, FunctionOp functionOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  auto module = functionOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointAfter(functionOp);

  auto derivativeAttribute = functionOp->getAttrOfType<DerivativeAttr>("derivative");
  unsigned int order = derivativeAttribute.getOrder();

  if (auto derSymbol = module.lookupSymbol(derivativeAttribute.getName())) {
    // If the source already provides a symbol with the derived function name, then
    // check that it is a function. If it is, then it means the user already manually
    // provided the derivative.
    return mlir::LogicalResult::success(mlir::isa<FunctionOp>(derSymbol));
  }

  mlir::BlockAndValueMapping derivatives;
  mapFullDerivatives(derivatives, functionOp.getMembers());

  // Map the members for a faster lookup
  auto originalMembersMap = mapMembersByName(functionOp);

  // New members of the derived function
  llvm::SmallVector<std::string, 3> newInputMembersNames;
  llvm::SmallVector<MemberType, 3> newInputMembersTypes;

  llvm::SmallVector<std::string, 3> newOutputMembersNames;
  llvm::SmallVector<MemberType, 3> newOutputMembersTypes;

  llvm::SmallVector<std::string, 3> newProtectedMembersNames;
  llvm::SmallVector<MemberType, 3> newProtectedMembersTypes;

  llvm::StringMap<std::string> inverseDerivativesNamesMap;

  // Analyze the original input members
  for (const auto& name : functionOp.inputMemberNames()) {
    auto member = originalMembersMap[name].getDefiningOp<MemberCreateOp>();
    auto memberType = member.getMemberType();

    if (hasFloatBase(memberType.unwrap())) {
      if (isFullDerivative(name, functionOp, order - 1)) {
        continue;
      }

      auto derName = getFullDerVariableName(name, order);
      newInputMembersNames.push_back(derName);
      newInputMembersTypes.push_back(memberType);
      inverseDerivativesNamesMap[derName] = name;
    }
  }

  llvm::SmallVector<mlir::Type, 3> argsTypes;

  for (const auto& name : functionOp.inputMemberNames()) {
    argsTypes.push_back(originalMembersMap[name].getDefiningOp<MemberCreateOp>().getMemberType().unwrap());
  }

  for (const auto& type : newInputMembersTypes) {
    argsTypes.push_back(type.unwrap());
  }

  // Analyze the original output members
  llvm::SmallVector<std::string, 3> resultsNames;
  llvm::SmallVector<mlir::Type, 3> resultsTypes;

  for (const auto& name : functionOp.outputMemberNames()) {
    auto memberType = originalMembersMap[name].getDefiningOp<MemberCreateOp>().getMemberType();

    if (hasFloatBase(memberType.unwrap())) {
      auto derName = getNextFullDerVariableName(name, order);
      newOutputMembersNames.push_back(derName);
      newOutputMembersTypes.push_back(memberType);
      inverseDerivativesNamesMap[derName] = name;
    }
  }

  for (const auto& type : newOutputMembersTypes) {
    resultsTypes.push_back(type.unwrap());
  }

  // Analyze the original protected members
  for (const auto& name : functionOp.protectedMemberNames()) {
    auto originalMemberOp = originalMembersMap[name].getDefiningOp<MemberCreateOp>();

    if (derivatives.contains(originalMemberOp.getResult())) {
      // Avoid duplicates of original output members, which have become
      // protected members in the previous derivative functions.
      continue;
    }

    if (isFullDerivative(name, functionOp, order - 1)) {
      continue;
    }

    auto derName = getFullDerVariableName(name, order);
    newProtectedMembersNames.push_back(derName);

    auto memberType = originalMemberOp.getMemberType();
    newProtectedMembersTypes.push_back(memberType);
  }

  derivatives.clear();

  // Create the derived function
  auto derivedFunctionOp = builder.create<FunctionOp>(
      functionOp.getLoc(),
      derivativeAttribute.getName(),
      builder.getFunctionType(argsTypes, resultsTypes));

  // Start the body of the function
  mlir::Block* entryBlock = builder.createBlock(&derivedFunctionOp.body());
  builder.setInsertionPointToStart(entryBlock);

  // Clone the original operations, which will be interleaved in the
  // resulting derivative function.
  mlir::BlockAndValueMapping mapping;
  mlir::Operation* latestMemberCreateOp = nullptr;

  for (auto& baseOp : functionOp.bodyBlock()->getOperations()) {
    if (auto memberCreateOp = mlir::dyn_cast<MemberCreateOp>(baseOp)) {
      auto name = memberCreateOp.name();

      if (memberCreateOp.isInput()) {
        latestMemberCreateOp = builder.clone(baseOp, mapping);

      } else if (memberCreateOp.isOutput()) {
        // Convert the output members to protected members
        std::vector<mlir::Value> mappedDynamicDimensions;

        for (const auto& dynamicDimension : memberCreateOp.dynamicSizes()) {
          mappedDynamicDimensions.push_back(mapping.lookup(dynamicDimension));
        }

        auto mappedMemberType = memberCreateOp.getMemberType().withIOProperty(IOProperty::none);

        auto mappedMember = builder.create<MemberCreateOp>(
            memberCreateOp.getLoc(), name, mappedMemberType, mappedDynamicDimensions);

        mapping.map(memberCreateOp, mappedMember);
        latestMemberCreateOp = mappedMember.getOperation();
      } else {
        latestMemberCreateOp = builder.clone(baseOp, mapping);
      }
    } else {
      mlir::Operation* clonedOp = builder.clone(baseOp, mapping);

      if (isDerived(&baseOp)) {
        setAsDerived(clonedOp);
      }
    }
  }

  // Insert the new derivative members
  if (latestMemberCreateOp == nullptr) {
    builder.setInsertionPointToStart(derivedFunctionOp.bodyBlock());
  } else {
    builder.setInsertionPointAfter(latestMemberCreateOp);
  }

  auto createDerMemberFn = [&](llvm::ArrayRef<std::string> derNames, llvm::ArrayRef<MemberType> derTypes) {
    for (const auto& [name, type] : llvm::zip(derNames, derTypes)) {
      auto baseMemberName = inverseDerivativesNamesMap[name];
      auto baseMember = mapping.lookup(originalMembersMap[baseMemberName].getDefiningOp<MemberCreateOp>().getResult());

      auto derivedMember = builder.create<MemberCreateOp>(
          baseMember.getLoc(), name, type,
          baseMember.getDefiningOp<MemberCreateOp>().dynamicSizes());
    }
  };

  createDerMemberFn(newInputMembersNames, newInputMembersTypes);
  createDerMemberFn(newOutputMembersNames, newOutputMembersTypes);
  createDerMemberFn(newProtectedMembersNames, newProtectedMembersTypes);

  // Map the derivatives among members
  mapFullDerivatives(derivatives, derivedFunctionOp.getMembers());

  // Derive the operations
  return deriveFunctionBody(builder, derivedFunctionOp, derivatives);
}

mlir::LogicalResult ForwardAD::createPartialDerTemplateFunction(
    mlir::OpBuilder& builder, PartialDerTemplateFunctionsMap& derFunctionsMap, DerFunctionOp derFunctionOp)
{
  auto module = derFunctionOp->getParentOfType<mlir::ModuleOp>();
  FunctionOp baseFunctionOp = module.lookupSymbol<FunctionOp>(derFunctionOp.derived_function());
  std::string derivedFunctionName = getPartialDerFunctionName(derFunctionOp.name());

  for (const auto& independentVariable : derFunctionOp.independent_vars()) {
    auto derTemplate = createPartialDerTemplateFunction(builder, derFunctionsMap, derFunctionOp.getLoc(), baseFunctionOp, derivedFunctionName);

    if (derTemplate == nullptr) {
      return mlir::failure();
    }

    derFunctionsMap[derTemplate.name()] = baseFunctionOp;
    baseFunctionOp = derTemplate;
    derivedFunctionName = getPartialDerFunctionName(baseFunctionOp.name());
  }

  return mlir::success();

  /*
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(derFunctionOp);

  auto loc = derFunctionOp.getLoc();
  auto module = derFunctionOp->getParentOfType<mlir::ModuleOp>();
  auto baseFunctionOp = module.lookupSymbol<FunctionOp>(derFunctionOp.derived_function());

  // Map the members for a faster lookup
  auto originalMembersMap = mapMembersByName(baseFunctionOp);

  // New members of the derived function
  llvm::SmallVector<std::string, 3> newInputMembersNames;
  llvm::SmallVector<MemberType, 3> newInputMembersTypes;

  llvm::SmallVector<std::string, 3> newOutputMembersNames;
  llvm::SmallVector<MemberType, 3> newOutputMembersTypes;

  llvm::SmallVector<std::string, 3> newProtectedMembersNames;
  llvm::SmallVector<MemberType, 3> newProtectedMembersTypes;

  llvm::StringMap<std::string> inverseDerivativesNamesMap;

  // Analyze the original input members
  for (const auto& name : baseFunctionOp.inputMemberNames()) {
    auto member = originalMembersMap[name].getDefiningOp<MemberCreateOp>();
    auto memberType = member.getMemberType();
    auto derName = "pder_" + name.str();
    newInputMembersNames.push_back(derName);
    newInputMembersTypes.push_back(memberType.withType(RealType::get(builder.getContext())));
    inverseDerivativesNamesMap[derName] = name;
  }

  // Analyze the original output members
  for (const auto& name : baseFunctionOp.outputMemberNames()) {
    auto memberType = originalMembersMap[name].getDefiningOp<MemberCreateOp>().getMemberType();
    auto derName = "pder_" + name.str();
    newOutputMembersNames.push_back(derName);
    newOutputMembersTypes.push_back(memberType);
    inverseDerivativesNamesMap[derName] = name;
  }

  // Analyze the original protected members
  for (const auto& name : baseFunctionOp.protectedMemberNames()) {
    auto memberType = originalMembersMap[name].getDefiningOp<MemberCreateOp>().getMemberType();
    auto derName = "pder_" + name.str();
    newProtectedMembersNames.push_back(derName);
    newProtectedMembersTypes.push_back(memberType);
  }

  // Determine the function signature
  llvm::SmallVector<mlir::Type, 6> argsTypes;
  llvm::SmallVector<mlir::Type, 6> resultsTypes;

  // The original arguments
  for (const auto& type : baseFunctionOp.getType().getInputs()) {
    argsTypes.push_back(type);
  }

  // The seed values
  for (const auto& type : baseFunctionOp.getType().getInputs()) {
    // TODO may be array
    argsTypes.push_back(RealType::get(builder.getContext()));
  }

  // The original results
  for (const auto& type : baseFunctionOp.getType().getResults()) {
    resultsTypes.push_back(type);
  }

  // Create the derived function
  auto derivedFunctionOp = builder.create<FunctionOp>(
      derFunctionOp.getLoc(),
      getPartialDerTemplateFunctionName(derFunctionOp.name()),
      builder.getFunctionType(argsTypes, resultsTypes));

  // Start the body of the function
  mlir::Block* entryBlock = builder.createBlock(&derivedFunctionOp.body());
  builder.setInsertionPointToStart(entryBlock);

  // Clone the original operations, which will be interleaved in the
  // resulting derivative function.
  mlir::BlockAndValueMapping mapping;
  mlir::Operation* latestMemberCreateOp = nullptr;

  for (auto& baseOp : baseFunctionOp.bodyBlock()->getOperations()) {
    if (auto memberCreateOp = mlir::dyn_cast<MemberCreateOp>(baseOp)) {
      auto name = memberCreateOp.name();

      if (memberCreateOp.isInput()) {
        latestMemberCreateOp = builder.clone(baseOp, mapping);

      } else if (memberCreateOp.isOutput()) {
        // Convert the output members to protected members
        std::vector<mlir::Value> mappedDynamicDimensions;

        for (const auto& dynamicDimension : memberCreateOp.dynamicSizes()) {
          mappedDynamicDimensions.push_back(mapping.lookup(dynamicDimension));
        }

        auto mappedMemberType = memberCreateOp.getMemberType().withIOProperty(IOProperty::none);

        auto mappedMember = builder.create<MemberCreateOp>(
            memberCreateOp.getLoc(), name, mappedMemberType, mappedDynamicDimensions);

        mapping.map(memberCreateOp, mappedMember);
        latestMemberCreateOp = mappedMember.getOperation();
      } else {
        latestMemberCreateOp = builder.clone(baseOp, mapping);
      }
    } else {
      builder.clone(baseOp, mapping);
    }
  }

  mlir::BlockAndValueMapping derivatives;

  // Insert the new derivative members
  if (latestMemberCreateOp == nullptr) {
    builder.setInsertionPointToStart(derivedFunctionOp.bodyBlock());
  } else {
    builder.setInsertionPointAfter(latestMemberCreateOp);
  }

  auto createDerMemberFn = [&](llvm::ArrayRef<std::string> derNames, llvm::ArrayRef<MemberType> derTypes) {
    for (const auto& [name, type] : llvm::zip(derNames, derTypes)) {
      auto baseMemberName = inverseDerivativesNamesMap[name];
      auto baseMember = mapping.lookup(originalMembersMap[baseMemberName].getDefiningOp<MemberCreateOp>().getResult());

      auto derivedMember = builder.create<MemberCreateOp>(
          baseMember.getLoc(), name, type,
          baseMember.getDefiningOp<MemberCreateOp>().dynamicSizes());

      derivatives.map(baseMember, derivedMember.getResult());
    }
  };

  createDerMemberFn(newInputMembersNames, newInputMembersTypes);
  createDerMemberFn(newOutputMembersNames, newOutputMembersTypes);
  createDerMemberFn(newProtectedMembersNames, newProtectedMembersTypes);

  // Derive the operations
  return deriveFunctionBody(builder, derivedFunctionOp, derivatives);
   */
}

FunctionOp ForwardAD::createPartialDerTemplateFunction(
    mlir::OpBuilder& builder,
    const PartialDerTemplateFunctionsMap& derFunctionsMap,
    mlir::Location loc,
    FunctionOp functionOp,
    llvm::StringRef derivedFunctionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(functionOp);

  // Determine the characteristics of the base function
  FunctionOp baseFunctionOp;
  unsigned int currentOrder;

  std::tie(baseFunctionOp, currentOrder) = getPartialDerBaseFunction(derFunctionsMap, functionOp);
  unsigned int baseArgs = baseFunctionOp.getType().getNumInputs();

  mlir::BlockAndValueMapping derivatives;

  // Map the members for a faster lookup
  auto originalMembersMap = mapMembersByName(functionOp);

  // Determine the function signature
  llvm::SmallVector<mlir::Type, 6> argsTypes;
  llvm::SmallVector<mlir::Type, 6> resultsTypes;

  // The original arguments
  for (const auto& type : functionOp.getType().getInputs()) {
    argsTypes.push_back(type);
  }

  // The seed values
  for (const auto& type : functionOp.getType().getInputs()) {
    // TODO may be array
    argsTypes.push_back(RealType::get(builder.getContext()));
  }

  // The original results
  for (const auto& type : functionOp.getType().getResults()) {
    resultsTypes.push_back(type);
  }

  // Create the derived function
  auto derivedFunctionOp = builder.create<FunctionOp>(
      functionOp.getLoc(),
      derivedFunctionName,
      builder.getFunctionType(argsTypes, resultsTypes));

  // Start the body of the function
  mlir::Block* entryBlock = builder.createBlock(&derivedFunctionOp.body());
  builder.setInsertionPointToStart(entryBlock);

  // Clone the original operations, which will be interleaved in the
  // resulting derivative function.
  mlir::BlockAndValueMapping mapping;
  mlir::Operation* latestMemberCreateOp = nullptr;

  for (auto& baseOp : functionOp.bodyBlock()->getOperations()) {
    if (auto memberCreateOp = mlir::dyn_cast<MemberCreateOp>(baseOp)) {
      auto name = memberCreateOp.name();

      if (memberCreateOp.isInput()) {
        latestMemberCreateOp = builder.clone(baseOp, mapping);

      } else if (memberCreateOp.isOutput()) {
        // Convert the output members to protected members
        std::vector<mlir::Value> mappedDynamicDimensions;

        for (const auto& dynamicDimension : memberCreateOp.dynamicSizes()) {
          mappedDynamicDimensions.push_back(mapping.lookup(dynamicDimension));
        }

        auto mappedMemberType = memberCreateOp.getMemberType().withIOProperty(IOProperty::none);

        auto mappedMember = builder.create<MemberCreateOp>(
            memberCreateOp.getLoc(), name, mappedMemberType, mappedDynamicDimensions);

        mapping.map(memberCreateOp, mappedMember);
        latestMemberCreateOp = mappedMember.getOperation();
      } else {
        latestMemberCreateOp = builder.clone(baseOp, mapping);
      }
    } else {
      builder.clone(baseOp, mapping);
    }
  }

  // New members of the derived function
  llvm::SmallVector<std::string, 3> newInputMembersNames;
  llvm::SmallVector<MemberType, 3> newInputMembersTypes;

  llvm::SmallVector<std::string, 3> newOutputMembersNames;
  llvm::SmallVector<MemberType, 3> newOutputMembersTypes;

  llvm::SmallVector<std::string, 3> newProtectedMembersNames;
  llvm::SmallVector<MemberType, 3> newProtectedMembersTypes;

  llvm::StringMap<std::string> inverseDerivativesNamesMap;

  // Analyze the original input members
  for (const auto& name : functionOp.inputMemberNames()) {
    auto derName = getPartialDerMemberName(name, currentOrder + 1);
    auto type = originalMembersMap[name].getDefiningOp<MemberCreateOp>().getMemberType();

    newInputMembersNames.push_back(derName);
    newInputMembersTypes.push_back(type);

    inverseDerivativesNamesMap[derName] = name;
  }

  // Analyze the original output members
  for (const auto& name : functionOp.outputMemberNames()) {
    auto derName = getPartialDerMemberName(name, currentOrder + 1);
    auto type = originalMembersMap[name].getDefiningOp<MemberCreateOp>().getMemberType();

    newInputMembersNames.push_back(derName);
    newInputMembersTypes.push_back(type);

    inverseDerivativesNamesMap[derName] = name;
  }

  // Analyze the original protected members
  for (const auto& name : functionOp.protectedMemberNames()) {
    auto derName = getPartialDerMemberName(name, currentOrder + 1);
    auto type = originalMembersMap[name].getDefiningOp<MemberCreateOp>().getMemberType();

    newInputMembersNames.push_back(derName);
    newInputMembersTypes.push_back(type);

    inverseDerivativesNamesMap[derName] = name;
  }

  // Insert the new derivative members
  if (latestMemberCreateOp == nullptr) {
    builder.setInsertionPointToStart(derivedFunctionOp.bodyBlock());
  } else {
    builder.setInsertionPointAfter(latestMemberCreateOp);
  }

  auto createDerMemberFn = [&](llvm::ArrayRef<std::string> derNames, llvm::ArrayRef<MemberType> derTypes) {
    for (const auto& [name, type] : llvm::zip(derNames, derTypes)) {
      auto baseMemberName = inverseDerivativesNamesMap[name];
      auto baseMember = mapping.lookup(originalMembersMap[baseMemberName].getDefiningOp<MemberCreateOp>().getResult());

      auto derivedMember = builder.create<MemberCreateOp>(
          baseMember.getLoc(), name, type,
          baseMember.getDefiningOp<MemberCreateOp>().dynamicSizes());

      derivatives.map(baseMember, derivedMember.getResult());
    }
  };

  createDerMemberFn(newInputMembersNames, newInputMembersTypes);
  createDerMemberFn(newOutputMembersNames, newOutputMembersTypes);
  createDerMemberFn(newProtectedMembersNames, newProtectedMembersTypes);

  // Derive the operations
  if (auto res = deriveFunctionBody(builder, derivedFunctionOp, derivatives); mlir::failed(res)) {
    return nullptr;
  }

  return derivedFunctionOp;
}

mlir::LogicalResult ForwardAD::deriveFunctionBody(
    mlir::OpBuilder& builder,
    FunctionOp functionOp,
    mlir::BlockAndValueMapping& derivatives)
{
  // Determine the list of the derivable operations. We can't just derive as
  // we find them, as we would invalidate the operation walk's iterator.
  std::queue<DerivableOpInterface> derivableOps;

  for (auto derivableOp : functionOp.body().getOps<DerivableOpInterface>()) {
    derivableOps.push(derivableOp);
  }

  while (!derivableOps.empty()) {
    auto& op = derivableOps.front();

    builder.setInsertionPointAfter(op);

    if (!isDerived(op.getOperation())) {
      mlir::ValueRange derivedValues = op.derive(builder, derivatives);
      assert(op->getNumResults() == derivedValues.size());
      setAsDerived(op.getOperation());

      if (!derivedValues.empty()) {
        for (const auto& [base, derived] : llvm::zip(op->getResults(), derivedValues)) {
          derivatives.map(base, derived);
        }
      }
    }

    llvm::SmallVector<mlir::Region*, 3> regions;
    op.getDerivableRegions(regions);

    for (auto& region : regions) {
      for (auto derivableOp : region->getOps<DerivableOpInterface>()) {
        derivableOps.push(derivableOp);
      }
    }

    derivableOps.pop();
  }

  return mlir::success();
}

namespace
{
  class AutomaticDifferentiationPass: public mlir::PassWrapper<AutomaticDifferentiationPass, mlir::OperationPass<mlir::ModuleOp>>
  {
    public:
    void getDependentDialects(mlir::DialectRegistry& registry) const override
    {
      registry.insert<ModelicaDialect>();
    }

    void runOnOperation() override
    {
      if (mlir::failed(createFullDerFunctions())) {
        mlir::emitError(getOperation().getLoc(), "Error in creating the functions full derivatives");
        return signalPassFailure();
      }

      if (mlir::failed(createPartialDerFunctions())) {
        mlir::emitError(getOperation().getLoc(), "Error in creating the functions partial derivatives");
        return signalPassFailure();
      }

      if (mlir::failed(resolveTrivialDerCalls())) {
        mlir::emitError(getOperation().getLoc(), "Error in resolving the trivial derivative calls");
        return signalPassFailure();
      }
    }

    /*
    mlir::LogicalResult addPartialDerFunctions()
    {
      // If using the SUNDIALS IDA library as a solver, we also need the partial
      // function derivatives of all call operations in order to compute the
      // symbolic jacobian.

      // TODO: Fix partial derivatives of arrays and matrixes.
      mlir::ModuleOp module = getOperation();
      mlir::OpBuilder builder(module);
      mlir::OpBuilder::InsertionGuard guard(builder);

      llvm::SmallVector<FunctionOp, 3> funcToBeDerived;
      llvm::SmallVector<DerFunctionOp, 3> derFuncToBeDerived;

      module->walk([&](FunctionOp op) {
        if (op.getNumArguments() == 1 && op.getNumResults() == 1)
          funcToBeDerived.push_back(op);
      });

      module->walk([&](DerFunctionOp op) {
        if (op.independentVariables().size() == 1)
          derFuncToBeDerived.push_back(op);
      });

      // Add the partial derivative of all FunctionOp
      for (FunctionOp& function : funcToBeDerived)
      {
        std::string pderName = getPartialDerFunctionName(function.name());

        if (module.lookupSymbol<FunctionOp>(pderName) == nullptr &&
            module.lookupSymbol<DerFunctionOp>(pderName) == nullptr)
        {
          builder.setInsertionPointAfter(function);
          mlir::Attribute independentVariable = function.argsNames()[0];
          builder.create<DerFunctionOp>(function.getLoc(), pderName, function.getName(), independentVariable);
        }
      }

      // Add the partial derivative of all DerFunctionOp
      for (DerFunctionOp& op : derFuncToBeDerived)
      {
        std::string pderName = getPartialDerFunctionName(op.name());

        if (module.lookupSymbol<FunctionOp>(pderName) == nullptr &&
            module.lookupSymbol<DerFunctionOp>(pderName) == nullptr)
        {
          builder.setInsertionPointAfter(op);
          builder.create<DerFunctionOp>(op.getLoc(), pderName, op.getName(), op.independentVariables());
        }
      }

      return mlir::success();
    }
     */

    mlir::LogicalResult createFullDerFunctions()
    {
      auto module = getOperation();
      mlir::OpBuilder builder(module);

      llvm::SmallVector<FunctionOp, 3> toBeDerived;

      module->walk([&](FunctionOp op) {
        if (op->hasAttrOfType<DerivativeAttr>("derivative")) {
          toBeDerived.push_back(op);
        }
      });

      // Sort the functions so that a function derivative is computed only
      // when the base function already has its body determined.

      llvm::sort(toBeDerived, [](FunctionOp first, FunctionOp second) {
        auto annotation = first->getAttrOfType<DerivativeAttr>("derivative");
        return annotation.getName() == second.name();
      });

      ForwardAD forwardAD;

      for (auto& function : toBeDerived) {
        if (auto res = forwardAD.createFullDerFunction(builder, function); mlir::failed(res)) {
          return res;
        }
      }

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

      ForwardAD forwardAD;
      PartialDerTemplateFunctionsMap derFunctionsMap;

      while (findDerFunctions()) {
        // Sort the functions so that a function derivative is computed only
        // when the base function already has its body determined.

        llvm::sort(toBeProcessed, [](DerFunctionOp first, DerFunctionOp second) {
          return first.name() == second.derived_function();
        });

        for (auto& function : toBeProcessed) {
          if (auto res = forwardAD.createPartialDerTemplateFunction(builder, derFunctionsMap, function); mlir::failed(res)) {
            return res;
          }

          /*
          if (auto res = createPartialDerFunction(builder, function); mlir::failed(res)) {
            return res;
          }
           */

          function->erase();
        }

        toBeProcessed.clear();
      }

      return mlir::success();
    }

    mlir::LogicalResult resolveTrivialDerCalls()
    {
      auto module = getOperation();
      mlir::OpBuilder builder(module);

      module.walk([&](DerOp op) {
        mlir::Value operand = op.operand();
        mlir::Operation* definingOp = operand.getDefiningOp();

        if (definingOp == nullptr) {
          return;
        }

        if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(definingOp)) {
          auto classOp = op->getParentOfType<ClassInterface>();

          if (classOp == nullptr) {
            return;
          }

          mlir::BlockAndValueMapping derivatives;
          mapFullDerivatives(derivatives, classOp.getMembers());

          mlir::ValueRange ders = deriveTree(builder, derivableOp, derivatives);

          if (ders.size() != op->getNumResults()) {
            return;
          }

          op->replaceAllUsesWith(ders);
          op.erase();
        }
      });

      return mlir::success();
    }
  };
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass()
  {
    return std::make_unique<AutomaticDifferentiationPass>();
  }
}
