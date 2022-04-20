#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/Common.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/STLExtras.h"
#include <queue>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

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

namespace marco::codegen
{
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
    return deriveFunctionBody(
        builder, derivedFunctionOp, derivatives,
        [this](mlir::OpBuilder& builder, mlir::Operation* op, mlir::BlockAndValueMapping& derivatives) -> mlir::ValueRange {
          return createOpFullDerivative(builder, op, derivatives);
        });
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
        auto argType = templateFunction.getType().getInput(i);
        assert(!(seed == 1 && argType.isa<ArrayType>()));

        if (auto arrayType = argType.dyn_cast<ArrayType>()) {
          // TODO dynamic sizes
          assert(arrayType.hasConstantShape());

          mlir::Value array = builder.create<AllocOp>(loc, arrayType, llvm::None);
          args.push_back(array);
          mlir::Value seedValue = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), seed));
          builder.create<ArrayFillOp>(loc, array, seedValue);
        } else {
          args.push_back(builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), seed)));
        }
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
      auto realType = RealType::get(builder.getContext());

      if (auto arrayType = type.dyn_cast<ArrayType>()) {
        argsTypes.push_back(arrayType.toElementType(realType));
      } else {
        argsTypes.push_back(RealType::get(builder.getContext()));
      }
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
    auto res = deriveFunctionBody(
        builder, derivedFunctionOp, derivatives,
        [&](mlir::OpBuilder& nestedBuilder, mlir::Operation* op, mlir::BlockAndValueMapping& derivatives) -> mlir::ValueRange {
          return createOpPartialDerivative(nestedBuilder, op, derivatives);
        });

    if (mlir::failed(res)) {
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
      mlir::BlockAndValueMapping& derivatives,
      std::function<mlir::ValueRange(mlir::OpBuilder&, mlir::Operation*, mlir::BlockAndValueMapping&)> deriveFn)
  {
    // Determine the list of the derivable operations. We can't just derive as
    // we find them, as we would invalidate the operation walk's iterator.
    std::queue<mlir::Operation*> ops;

    for (auto& op : functionOp.body().getOps()) {
      ops.push(&op);
    }

    while (!ops.empty()) {
      auto op = ops.front();
      ops.pop();

      builder.setInsertionPointAfter(op);

      if (isDerivable(op) && !isDerived(op)) {
        mlir::ValueRange derivedValues = deriveFn(builder, op, derivatives);
        assert(op->getNumResults() == derivedValues.size());
        setAsDerived(op);

        if (!derivedValues.empty()) {
          for (const auto& [base, derived] : llvm::zip(op->getResults(), derivedValues)) {
            derivatives.map(base, derived);
          }
        }
      }

      if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(op)) {
        llvm::SmallVector<mlir::Region*, 3> regions;
        derivableOp.getDerivableRegions(regions);

        for (auto& region : regions) {
          for (auto& childOp : region->getOps()) {
            ops.push(&childOp);
          }
        }
      }
    }

    return mlir::success();
  }

  bool ForwardAD::isDerivable(mlir::Operation* op) const
  {
    return mlir::isa<CallOp, TimeOp, DerivableOpInterface>(op);
  }

  mlir::ValueRange ForwardAD::createOpFullDerivative(
      mlir::OpBuilder& builder,
      mlir::Operation* op,
      mlir::BlockAndValueMapping& derivatives)
  {
    if (auto callOp = mlir::dyn_cast<CallOp>(op)) {
      return createCallOpFullDerivative(builder, callOp, derivatives);
    }

    if (auto timeOp = mlir::dyn_cast<TimeOp>(op)) {
      return createTimeOpFullDerivative(builder, timeOp, derivatives);
    }

    if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(op)) {
      return derivableOp.derive(builder, derivatives);
    }

    llvm_unreachable("Can't derive a non-derivable operation");
    return llvm::None;
  }

  mlir::ValueRange ForwardAD::createOpPartialDerivative(
      mlir::OpBuilder& builder,
      mlir::Operation* op,
      mlir::BlockAndValueMapping& derivatives)
  {
    if (auto callOp = mlir::dyn_cast<CallOp>(op)) {
      return createCallOpPartialDerivative(builder, callOp, derivatives);
    }

    if (auto timeOp = mlir::dyn_cast<TimeOp>(op)) {
      return createTimeOpPartialDerivative(builder, timeOp, derivatives);
    }

    if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(op)) {
      return derivableOp.derive(builder, derivatives);
    }

    llvm_unreachable("Can't derive a non-derivable operation");
    return llvm::None;
  }

  mlir::ValueRange ForwardAD::createCallOpFullDerivative(
      mlir::OpBuilder& builder,
      CallOp callOp,
      mlir::BlockAndValueMapping& derivatives)
  {
    llvm_unreachable("CallOp full derivative is not implemented");
    return llvm::None;
  }

  mlir::ValueRange ForwardAD::createCallOpPartialDerivative(
      mlir::OpBuilder& builder,
      CallOp callOp,
      mlir::BlockAndValueMapping& derivatives)
  {
    auto loc = callOp.getLoc();
    auto module = callOp->getParentOfType<mlir::ModuleOp>();
    auto callee = module.lookupSymbol<FunctionOp>(callOp.callee());

    std::string derivedFunctionName = "call_pder_" + callOp.callee().str();

    llvm::SmallVector<mlir::Value, 3> args;

    for (auto arg : callOp.args()) {
      args.push_back(arg);
    }

    for (auto arg : callOp.args()) {
      args.push_back(derivatives.lookup(arg));
    }

    if (auto derTemplate = module.lookupSymbol<FunctionOp>(derivedFunctionName)) {
      return builder.create<CallOp>(loc, derTemplate, args)->getResults();
    }

    auto derTemplate = createPartialDerTemplateFunction(builder, loc, callee, derivedFunctionName);
    return builder.create<CallOp>(loc, derTemplate, args)->getResults();
  }

  mlir::ValueRange ForwardAD::createTimeOpFullDerivative(
      mlir::OpBuilder& builder,
      mlir::modelica::TimeOp timeOp,
      mlir::BlockAndValueMapping& derivatives)
  {
    return builder.create<ConstantOp>(timeOp.getLoc(), RealAttr::get(timeOp.getContext(), 1))->getResults();
  }

  mlir::ValueRange ForwardAD::createTimeOpPartialDerivative(
      mlir::OpBuilder& builder,
      mlir::modelica::TimeOp timeOp,
      mlir::BlockAndValueMapping& derivatives)
  {
    return builder.create<ConstantOp>(timeOp.getLoc(), RealAttr::get(timeOp.getContext(), 0))->getResults();
  }

  mlir::ValueRange ForwardAD::deriveTree(
      mlir::OpBuilder& builder, DerivableOpInterface op, mlir::BlockAndValueMapping& derivatives)
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
}
