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

static std::string getPartialDerFunctionName(llvm::StringRef baseName)
{
  return "pder_" + baseName.str();
}

static std::string getPartialDerMemberName(llvm::StringRef baseName, unsigned int order)
{
  return "pder_" + std::to_string(order) + "_" + baseName.str();
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
      mlir::LogicalResult createFullDerFunction(
          mlir::OpBuilder& builder, FunctionOp functionOp);

      mlir::LogicalResult createPartialDerFunction(
          mlir::OpBuilder& builder, DerFunctionOp derFunctionOp);

    private:
      /// Check if an operation has already been already derived.
      bool isDerived(mlir::Operation* op) const;

      /// Set an operation as already derived.
      void setAsDerived(mlir::Operation* op);

      FunctionOp createPartialDerTemplateFunction(
          mlir::OpBuilder& builder, DerFunctionOp derFunctionOp);

      FunctionOp createPartialDerTemplateFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          FunctionOp functionOp,
          llvm::StringRef derivedFunctionName);

      std::pair<FunctionOp, unsigned int> getPartialDerBaseFunction(FunctionOp functionOp);

      mlir::LogicalResult deriveFunctionBody(
          mlir::OpBuilder& builder,
          FunctionOp functionOp,
          mlir::BlockAndValueMapping& derivatives);

    private:
      // Keeps track of the operations that have already been derived
      std::set<mlir::Operation*> derivedOps;

      // Map each partial derivative template function to its base function
      llvm::StringMap<FunctionOp> partialDerTemplates;

      llvm::StringMap<FunctionOp> partialDersTemplateCallers;

      llvm::StringMap<mlir::ArrayAttr> partialDerTemplatesIndependentVars;
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

      builder.create<MemberCreateOp>(
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

mlir::LogicalResult ForwardAD::createPartialDerFunction(
    mlir::OpBuilder& builder, DerFunctionOp derFunctionOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(derFunctionOp);

  auto templateFunction = createPartialDerTemplateFunction(builder, derFunctionOp);

  if (templateFunction == nullptr) {
    return mlir::failure();
  }

  FunctionOp zeroOrderFunctionOp;
  unsigned int templateOrder;

  std::tie(zeroOrderFunctionOp, templateOrder) = getPartialDerBaseFunction(templateFunction);

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

  partialDersTemplateCallers[derivedFunctionOp.name()] = templateFunction;

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

  auto zeroOrderArgsNumber = zeroOrderFunctionOp.getType().getNumInputs();
  auto inputMemberNames = zeroOrderFunctionOp.inputMemberNames();

  std::vector<mlir::Attribute> allIndependentVars;

  if (auto previousTemplateIt = partialDerTemplates.find(templateFunction.name()); previousTemplateIt != partialDerTemplates.end()) {
    auto previousTemplateVarsIt = partialDerTemplatesIndependentVars.find(previousTemplateIt->second.name());

    if (previousTemplateVarsIt != partialDerTemplatesIndependentVars.end()) {
      for (const auto& independentVar : previousTemplateVarsIt->second) {
        allIndependentVars.push_back(independentVar.cast<mlir::StringAttr>());
      }
    }
  }

  for (const auto& independentVariable : derFunctionOp.independent_vars()) {
    allIndependentVars.push_back(independentVariable.cast<mlir::StringAttr>());
  }

  partialDerTemplatesIndependentVars[templateFunction.name()] = builder.getArrayAttr(allIndependentVars);

  assert(templateOrder == allIndependentVars.size());
  unsigned int numberOfSeeds = zeroOrderArgsNumber;

  for (const auto& independentVariable : llvm::enumerate(allIndependentVars)) {
    auto independentVarName = independentVariable.value().cast<mlir::StringAttr>().getValue();
    unsigned int memberIndex = zeroOrderArgsNumber;

    for (unsigned int i = 0; i < zeroOrderArgsNumber; ++i) {
      if (inputMemberNames[i] == independentVarName) {
        memberIndex = i;
        break;
      }
    }

    assert(memberIndex < zeroOrderArgsNumber);

    for (unsigned int i = 0; i < numberOfSeeds; ++i) {
      float seed = i == memberIndex ? 1 : 0;
      // TODO handle arrays
      args.push_back(builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), seed)));
    }

    numberOfSeeds *= 2;
  }

  auto callOp = builder.create<CallOp>(
      loc,
      templateFunction.name(),
      resultsTypes, args);

  assert(callOp->getNumResults() == outputMembers.size());

  for (auto [outputMember, result] : llvm::zip(outputMembers, callOp->getResults())) {
    builder.create<MemberStoreOp>(loc, outputMember, result);
  }

  return mlir::success();
}

FunctionOp ForwardAD::createPartialDerTemplateFunction(
    mlir::OpBuilder& builder, DerFunctionOp derFunctionOp)
{
  auto module = derFunctionOp->getParentOfType<mlir::ModuleOp>();
  FunctionOp functionOp = module.lookupSymbol<FunctionOp>(derFunctionOp.derived_function());

  if (auto it = partialDersTemplateCallers.find(functionOp.name()); it != partialDersTemplateCallers.end()) {
    functionOp = it->second;
  }

  std::string derivedFunctionName = getPartialDerFunctionName(derFunctionOp.name());

  for (size_t i = 0; i < derFunctionOp.independent_vars().size(); ++i) {
    auto derTemplate = createPartialDerTemplateFunction(
        builder, derFunctionOp.getLoc(), functionOp, derivedFunctionName);

    if (derTemplate == nullptr) {
      return nullptr;
    }

    partialDerTemplates[derTemplate.name()] = functionOp;
    functionOp = derTemplate;
    derivedFunctionName = getPartialDerFunctionName(functionOp.name());
  }

  return functionOp;
}

FunctionOp ForwardAD::createPartialDerTemplateFunction(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    FunctionOp functionOp,
    llvm::StringRef derivedFunctionName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(functionOp);

  // Determine the characteristics of the base function
  FunctionOp baseFunctionOp;
  unsigned int currentOrder;

  std::tie(baseFunctionOp, currentOrder) = getPartialDerBaseFunction(functionOp);

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

std::pair<FunctionOp, unsigned int> ForwardAD::getPartialDerBaseFunction(FunctionOp functionOp)
{
  unsigned int order = 0;
  FunctionOp baseFunction = functionOp;

  while (partialDerTemplates.find(baseFunction.name()) != partialDerTemplates.end()) {
    ++order;
    baseFunction = partialDerTemplates.lookup(baseFunction.name());
  }

  return std::make_pair(baseFunction, order);
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

      while (findDerFunctions()) {
        // Sort the functions so that a function derivative is computed only
        // when the base function already has its body determined.

        llvm::sort(toBeProcessed, [](DerFunctionOp first, DerFunctionOp second) {
          return first.name() == second.derived_function();
        });

        for (auto& function : toBeProcessed) {
          if (auto res = forwardAD.createPartialDerFunction(builder, function); mlir::failed(res)) {
            return res;
          }

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

      std::vector<DerOp> ops;

      module.walk([&](DerOp op) {
        ops.push_back(op);
      });

      for (auto derOp : ops) {
        mlir::Value operand = derOp.operand();
        mlir::Operation* definingOp = operand.getDefiningOp();

        if (definingOp == nullptr) {
          continue;
        }

        if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(definingOp)) {
          auto classOp = derOp->getParentOfType<ClassInterface>();

          if (classOp == nullptr) {
            continue;
          }

          mlir::BlockAndValueMapping derivatives;
          mapFullDerivatives(derivatives, classOp.getMembers());

          mlir::ValueRange ders = deriveTree(builder, derivableOp, derivatives);

          if (ders.size() != derOp->getNumResults()) {
            continue;
          }

          derOp->replaceAllUsesWith(ders);
          derOp.erase();
        }
      }

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
