#include "marco/Codegen/Transforms/Model/IDA.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "mlir/IR/AffineMap.h"
#include <queue>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static FunctionOp createPartialDerTemplateFromEquation(
    mlir::OpBuilder& builder,
    Equation& equation,
    mlir::ValueRange originalVariables,
    llvm::StringRef templateName)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(equation.getOperation()->getParentOfType<mlir::ModuleOp>().getBody());
  auto loc = equation.getOperation().getLoc();

  std::string functionOpName = templateName.str() + "_base";

  // The arguments of the base function contain both the variables and the inductions
  llvm::SmallVector<mlir::Type, 6> argsTypes;

  for (auto type : originalVariables.getTypes()) {
    argsTypes.push_back(type);
  }

  for (size_t i = 0; i < equation.getNumOfIterationVars(); ++i) {
    argsTypes.push_back(builder.getIndexType());
  }

  // Create the function to be derived
  auto functionOp = builder.create<FunctionOp>(
      loc,
      functionOpName,
      builder.getFunctionType(argsTypes, RealType::get(builder.getContext())));

  // Start the body of the function
  mlir::Block* entryBlock = builder.createBlock(&functionOp.body());
  builder.setInsertionPointToStart(entryBlock);

  // Create the input members and map them to the original variables (and inductions)
  mlir::BlockAndValueMapping mapping;

  for (auto originalVar : llvm::enumerate(originalVariables)) {
    auto memberType = MemberType::wrap(originalVar.value().getType(), false, IOProperty::input);
    auto memberOp = builder.create<MemberCreateOp>(loc, "var" + std::to_string(originalVar.index()), memberType, llvm::None);
    auto mappedVar = builder.create<MemberLoadOp>(loc, memberOp);
    mapping.map(originalVar.value(), mappedVar);
  }

  llvm::SmallVector<mlir::Value, 3> inductions;

  for (size_t i = 0; i < equation.getNumOfIterationVars(); ++i) {
    auto memberType = MemberType::wrap(builder.getIndexType(), false, IOProperty::input);
    auto memberOp = builder.create<MemberCreateOp>(loc, "ind" + std::to_string(i), memberType, llvm::None);
    inductions.push_back(builder.create<MemberLoadOp>(loc, memberOp));
  }

  auto explicitEquationInductions = equation.getInductionVariables();

  for (const auto& originalInduction : llvm::enumerate(explicitEquationInductions)) {
    assert(originalInduction.index() < inductions.size());
    mapping.map(originalInduction.value(), inductions[originalInduction.index()]);
  }

  // Create the output member, that is the difference between its equation right-hand side value and its
  // left-hand side value.
  auto originalTerminator = mlir::cast<EquationSidesOp>(equation.getOperation().bodyBlock()->getTerminator());
  assert(originalTerminator.lhsValues().size() == 1);
  assert(originalTerminator.rhsValues().size() == 1);

  auto outputMember = builder.create<MemberCreateOp>(
      loc, "out",
      MemberType::wrap(RealType::get(builder.getContext()), false, IOProperty::output),
      llvm::None);

  // Clone the original operations
  for (auto& op : equation.getOperation().bodyBlock()->getOperations()) {
    builder.clone(op, mapping);
  }

  auto terminator = mlir::cast<EquationSidesOp>(functionOp.bodyBlock()->getTerminator());
  assert(terminator.lhsValues().size() == 1);
  assert(terminator.rhsValues().size() == 1);

  mlir::Value lhs = terminator.lhsValues()[0];
  mlir::Value rhs = terminator.rhsValues()[0];

  if (auto arrayType = lhs.getType().dyn_cast<ArrayType>()) {
    assert(rhs.getType().isa<ArrayType>());
    assert(arrayType.getRank() + explicitEquationInductions.size() == inductions.size());
    auto implicitInductions = llvm::makeArrayRef(inductions).take_back(arrayType.getRank());

    lhs = builder.create<LoadOp>(loc, lhs, implicitInductions);
    rhs = builder.create<LoadOp>(loc, rhs, implicitInductions);
  }

  auto result = builder.create<SubOp>(loc, RealType::get(builder.getContext()), rhs, lhs);
  builder.create<MemberStoreOp>(loc, outputMember, result);

  auto lhsOp = terminator.lhs().getDefiningOp<EquationSideOp>();
  auto rhsOp = terminator.rhs().getDefiningOp<EquationSideOp>();
  terminator.erase();
  lhsOp.erase();
  rhsOp.erase();

  // Create the derivative template
  ForwardAD forwardAD;
  auto derTemplate = forwardAD.createPartialDerTemplateFunction(builder, loc, functionOp, templateName);
  functionOp.erase();
  return derTemplate;
}

namespace marco::codegen
{
  IDASolver::IDASolver() = default;

  bool IDASolver::isEnabled() const
  {
    // TODO should depend on the compiler's flags
    return true;
  }

  mlir::Type IDASolver::getSolverInstanceType(mlir::MLIRContext* context) const
  {
    return mlir::ida::InstanceType::get(context);
  }

  bool IDASolver::hasVariable(mlir::Value variable) const
  {
    return llvm::find(variables, variable) != variables.end();
  }

  void IDASolver::addVariable(mlir::Value variable)
  {
    assert(variable.isa<mlir::BlockArgument>());

    if (!hasVariable(variable)) {
      variables.push_back(variable);
    }
  }

  bool IDASolver::hasEquation(ScheduledEquation* equation) const
  {
    return llvm::find(equations, equation) != equations.end();
  }

  void IDASolver::addEquation(ScheduledEquation* equation)
  {
    equations.emplace(equation);
  }

  mlir::LogicalResult IDASolver::init(mlir::OpBuilder& builder, mlir::FuncOp initFunction)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&initFunction.getBody().front());

    // Create the IDA instance.
    // To do so, we need to first compute the total number of scalar variables that IDA
    // has to manage. Such number is equal to the number of scalar equations.
    size_t numberOfScalarEquations = 0;

    for (const auto& equation : equations) {
      numberOfScalarEquations += equation->getIterationRanges().flatSize();
    }

    idaInstance = builder.create<mlir::ida::CreateOp>(
        initFunction.getLoc(), builder.getI64IntegerAttr(numberOfScalarEquations));

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::processVariables(mlir::OpBuilder& builder, mlir::FuncOp initFunction, const mlir::BlockAndValueMapping& derivatives)
  {
    mlir::BlockAndValueMapping inverseDerivatives = derivatives.getInverse();
    auto terminator = mlir::cast<YieldOp>(initFunction.getBody().back().getTerminator());

    // Change the ownership of the variables managed to IDA

    for (const auto& variable : variables) {
      mlir::Value value = terminator.values()[variable.cast<mlir::BlockArgument>().getArgNumber()];
      auto memberCreateOp = value.getDefiningOp<MemberCreateOp>();
      builder.setInsertionPoint(memberCreateOp);

      auto dimensions = memberCreateOp.getMemberType().toArrayType().getShape();

      if (derivatives.contains(variable)) {
        // State variable
        auto idaVariable = builder.create<mlir::ida::AddVariableOp>(
            memberCreateOp.getLoc(), idaInstance,
            builder.getI64ArrayAttr(dimensions),
            builder.getBoolAttr(true));

        mappedVariables.map(variable, idaVariable);
        mlir::Value derivative = derivatives.lookup(variable);
        mappedVariables.map(derivative, idaVariable);

      } else if (!inverseDerivatives.contains(variable)) {
        // Algebraic variable
        auto idaVariable = builder.create<mlir::ida::AddVariableOp>(
            memberCreateOp.getLoc(), idaInstance,
            builder.getI64ArrayAttr(dimensions),
            builder.getBoolAttr(false));

        mappedVariables.map(variable, idaVariable);
      }
    }

    for (const auto& variable : variables) {
      mlir::Value value = terminator.values()[variable.cast<mlir::BlockArgument>().getArgNumber()];
      auto memberCreateOp = value.getDefiningOp<MemberCreateOp>();
      builder.setInsertionPoint(memberCreateOp);
      auto arrayType = memberCreateOp.getMemberType().toArrayType();
      mlir::Value idaVariable = mappedVariables.lookup(variable);

      if (inverseDerivatives.contains(variable)) {
        auto castedVariable = builder.create<mlir::ida::GetDerivativeOp>(
            memberCreateOp.getLoc(),
            arrayType,
            idaInstance, idaVariable);

        memberCreateOp->replaceAllUsesWith(castedVariable);
      } else {
        auto castedVariable = builder.create<mlir::ida::GetVariableOp>(
            memberCreateOp.getLoc(),
            arrayType,
            idaInstance, idaVariable);

        memberCreateOp->replaceAllUsesWith(castedVariable);
      }

      memberCreateOp.erase();
    }

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::processEquations(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model,
      mlir::FuncOp initFunction,
      mlir::TypeRange variableTypes,
      const mlir::BlockAndValueMapping& derivatives)
  {
    mlir::BlockAndValueMapping inverseDerivatives = derivatives.getInverse();

    // Substitute the accesses to non-IDA variables with the equations writing in such variables
    std::vector<std::unique_ptr<ScheduledEquation>> independentEquations;
    std::multimap<unsigned int, std::pair<modeling::MultidimensionalRange, ScheduledEquation*>> writesMap;

    for (const auto& equationsBlock : model.getScheduledBlocks()) {
      for (const auto& equation : *equationsBlock) {
        if (equations.find(equation.get()) == equations.end()) {
          const auto& write = equation->getWrite();
          auto varPosition = write.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
          auto writtenIndices = write.getAccessFunction().map(equation->getIterationRanges());
          writesMap.emplace(varPosition, std::make_pair(writtenIndices, equation.get()));
        }
      }
    }

    std::queue<std::unique_ptr<ScheduledEquation>> processedEquations;

    for (const auto& equation : equations) {
      auto clone = Equation::build(equation->cloneIR(), equation->getVariables());

      auto matchedClone = std::make_unique<MatchedEquation>(
          std::move(clone), equation->getIterationRanges(), equation->getWrite().getPath());

      auto scheduledClone = std::make_unique<ScheduledEquation>(
          std::move(matchedClone), equation->getIterationRanges(), equation->getSchedulingDirection());

      processedEquations.push(std::move(scheduledClone));
    }

    while (!processedEquations.empty()) {
      auto& equation = processedEquations.front();
      bool atLeastOneAccessReplaced = false;

      for (const auto& access : equation->getReads()) {
        if (atLeastOneAccessReplaced) {
          // Avoid unnecessary duplicates
          break;
        }

        auto readIndices = access.getAccessFunction().map(equation->getIterationRanges());
        auto varPosition = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
        auto writingEquations = llvm::make_range(writesMap.equal_range(varPosition));

        for (const auto& entry : writingEquations) {
          ScheduledEquation* writingEquation = entry.second.second;
          auto writtenVariableIndices = entry.second.first;

          if (!writtenVariableIndices.overlaps(readIndices)) {
            continue;
          }

          atLeastOneAccessReplaced = true;

          auto clone = Equation::build(equation->cloneIR(), equation->getVariables());

          auto explicitWritingEquation = writingEquation->cloneIRAndExplicitate(builder);
          TemporaryEquationGuard guard(*explicitWritingEquation);
          auto res = explicitWritingEquation->replaceInto(builder, *clone, access.getAccessFunction(), access.getPath());
          assert(mlir::succeeded(res));

          // Add the equation with the replaced access
          auto readAccessIndices = access.getAccessFunction().inverseMap(
              modeling::IndexSet(writtenVariableIndices),
              modeling::IndexSet(equation->getIterationRanges()));

          auto newEquationIndices = readAccessIndices.intersect(equation->getIterationRanges());

          for (const auto& range : newEquationIndices) {
            auto matchedEquation = std::make_unique<MatchedEquation>(
                clone->clone(), range, equation->getWrite().getPath());

            auto scheduledEquation = std::make_unique<ScheduledEquation>(
                std::move(matchedEquation), range, equation->getSchedulingDirection());

            processedEquations.push(std::move(scheduledEquation));
          }
        }
      }

      if (atLeastOneAccessReplaced) {
        equation->eraseIR();
      } else {
        independentEquations.push_back(std::move(equation));
      }

      processedEquations.pop();
    }

    // Set inside IDA the information about the equations
    auto terminator = mlir::cast<YieldOp>(initFunction.getBody().back().getTerminator());
    builder.setInsertionPoint(terminator);

    size_t residualFunctionsCounter = 0;
    size_t jacobianFunctionsCounter = 0;
    size_t partialDerTemplatesCounter = 0;

    for (const auto& equation : independentEquations) {
      equation->dumpIR();
      auto ranges = equation->getIterationRanges();
      std::vector<mlir::Attribute> rangesAttr;

      for (size_t i = 0; i < ranges.rank(); ++i) {
        rangesAttr.push_back(builder.getI64ArrayAttr({ ranges[i].getBegin(), ranges[i].getEnd() }));
      }

      auto idaEquation = builder.create<mlir::ida::AddEquationOp>(
          equation->getOperation().getLoc(),
          idaInstance,
          builder.getArrayAttr(rangesAttr));

      for (const auto& access : equation->getAccesses()) {
        mlir::Value idaVariable = mappedVariables.lookup(access.getVariable()->getValue());
        const auto& accessFunction = access.getAccessFunction();

        std::vector<mlir::AffineExpr> expressions;

        for (const auto& dimensionAccess : accessFunction) {
          if (dimensionAccess.isConstantAccess()) {
            expressions.push_back(mlir::getAffineConstantExpr(dimensionAccess.getPosition(), builder.getContext()));
          } else {
            auto baseAccess = mlir::getAffineDimExpr(dimensionAccess.getInductionVariableIndex(), builder.getContext());
            auto withOffset = baseAccess + dimensionAccess.getOffset();
            expressions.push_back(withOffset);
          }
        }

        builder.create<mlir::ida::AddVariableAccessOp>(
            equation->getOperation().getLoc(),
            idaInstance, idaEquation, idaVariable,
            mlir::AffineMap::get(accessFunction.size(), 0, expressions, builder.getContext()));
      }

      {
        // Create and populate the residual function
        auto residualFunctionName = "residualFunction" + std::to_string(residualFunctionsCounter++);

        auto insertionPoint = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(equation->getOperation()->getParentOfType<mlir::ModuleOp>().getBody());

        auto residualFunction = builder.create<mlir::ida::ResidualFunctionOp>(
            equation->getOperation().getLoc(),
            residualFunctionName,
            RealType::get(builder.getContext()),
            variableTypes,
            equation->getNumOfIterationVars(),
            RealType::get(builder.getContext()));

        assert(residualFunction.bodyRegion().empty());
        mlir::Block* bodyBlock = residualFunction.addEntryBlock();
        builder.setInsertionPointToStart(bodyBlock);

        mlir::BlockAndValueMapping mapping;

        // Map the model variables
        auto originalVars = model.getOperation().bodyRegion().getArguments();
        auto mappedVars = residualFunction.getArguments().slice(1, originalVars.size());
        assert(originalVars.size() == mappedVars.size());

        for (const auto& [original, mapped] : llvm::zip(originalVars, mappedVars)) {
          mapping.map(original, mapped);
        }

        // Map the iteration variables
        auto originalInductions = equation->getInductionVariables();
        auto mappedInductions = residualFunction.getArguments().slice(1 + originalVars.size());
        assert(originalInductions.size() == mappedInductions.size());

        for (const auto& [original, mapped] : llvm::zip(originalInductions, mappedInductions)) {
          mapping.map(original, mapped);
        }

        for (auto& op : equation->getOperation().bodyBlock()->getOperations()) {
          if (auto timeOp = mlir::dyn_cast<TimeOp>(op)) {
            mapping.map(timeOp.getResult(), residualFunction.getArguments()[0]);
          } else {
            builder.clone(op, mapping);
          }
        }

        auto clonedTerminator = mlir::cast<EquationSidesOp>(residualFunction.bodyRegion().back().getTerminator());

        assert(clonedTerminator.lhsValues().size() == 1);
        assert(clonedTerminator.rhsValues().size() == 1);

        mlir::Value lhs = clonedTerminator.lhsValues()[0];
        mlir::Value rhs = clonedTerminator.rhsValues()[0];

        if (lhs.getType().isa<ArrayType>()) {
          std::vector<mlir::Value> indices(
              std::next(mappedInductions.begin(), originalInductions.size()),
              mappedInductions.end());

          lhs = builder.create<LoadOp>(lhs.getLoc(), lhs, indices);
          assert((lhs.getType().isa<mlir::IndexType, BooleanType, IntegerType, RealType>()));
        }

        if (rhs.getType().isa<ArrayType>()) {
          std::vector<mlir::Value> indices(
              std::next(mappedInductions.begin(), originalInductions.size()),
              mappedInductions.end());

          rhs = builder.create<LoadOp>(rhs.getLoc(), rhs, indices);
          assert((rhs.getType().isa<mlir::IndexType, BooleanType, IntegerType, RealType>()));
        }

        mlir::Value difference = builder.create<SubOp>(residualFunction.getLoc(), RealType::get(builder.getContext()), rhs, lhs);
        builder.create<mlir::ida::ReturnOp>(difference.getLoc(), difference);
        clonedTerminator.erase();

        builder.restoreInsertionPoint(insertionPoint);
        builder.create<mlir::ida::AddResidualOp>(equation->getOperation().getLoc(), idaInstance, idaEquation, residualFunctionName);
      }

      std::string templateName = "ida_pder_" + std::to_string(partialDerTemplatesCounter++);

      {
        // Create the derivative template
        auto originalVars = model.getOperation().bodyRegion().getArguments();

        auto partialDerTemplate = createPartialDerTemplateFromEquation(
            builder, *equation, originalVars, templateName);

        // TODO add time in front to the partialDerTemplate signature
      }

      {
        for (const auto& variable : variables) {
          if (inverseDerivatives.contains(variable)) {
            continue;
          }

          auto insertionPoint = builder.saveInsertionPoint();

          // Create the Jacobian function
          auto jacobianFunctionName = "jacobianFunction" + std::to_string(jacobianFunctionsCounter++);
          builder.setInsertionPointToStart(equation->getOperation()->getParentOfType<mlir::ModuleOp>().getBody());

          auto jacobianFunction = builder.create<mlir::ida::JacobianFunctionOp>(
              equation->getOperation().getLoc(),
              jacobianFunctionName,
              RealType::get(builder.getContext()),
              variableTypes,
              equation->getNumOfIterationVars(),
              variable.getType().cast<ArrayType>().getRank(),
              RealType::get(builder.getContext()),
              RealType::get(builder.getContext()));

          assert(jacobianFunction.bodyRegion().empty());
          mlir::Block* bodyBlock = jacobianFunction.addEntryBlock();

          // Create the call to the derivative template
          builder.setInsertionPointToStart(bodyBlock);

          std::vector<mlir::Value> args;

          for (auto var : jacobianFunction.getVariables()) {
            args.push_back(var);
          }

          for (auto equationIndex : jacobianFunction.getEquationIndices()) {
            args.push_back(equationIndex);
          }

          unsigned int oneSeedPosition = variable.cast<mlir::BlockArgument>().getArgNumber();
          unsigned int alphaSeedPosition = jacobianFunction.getVariables().size();

          if (derivatives.contains(variable)) {
            alphaSeedPosition = derivatives.lookup(variable).cast<mlir::BlockArgument>().getArgNumber();
          }

          mlir::Value zero = builder.create<ConstantOp>(jacobianFunction.getLoc(), RealAttr::get(builder.getContext(), 0));
          mlir::Value one = builder.create<ConstantOp>(jacobianFunction.getLoc(), RealAttr::get(builder.getContext(), 1));

          for (auto var : llvm::enumerate(jacobianFunction.getVariables())) {
            if (auto arrayType = var.value().getType().dyn_cast<ArrayType>()) {
              assert(arrayType.hasConstantShape());

              auto array = builder.create<AllocOp>(
                  jacobianFunction.getLoc(),
                  arrayType.toElementType(RealType::get(builder.getContext())),
                  llvm::None);

              args.push_back(array);

              builder.create<ArrayFillOp>(jacobianFunction.getLoc(), array, zero);

              if (var.index() == oneSeedPosition) {
                builder.create<StoreOp>(jacobianFunction.getLoc(), one, array, jacobianFunction.getVariableIndices());
              } else if (var.index() == alphaSeedPosition) {
                builder.create<StoreOp>(jacobianFunction.getLoc(), jacobianFunction.getAlpha(), array, jacobianFunction.getVariableIndices());
              }

            } else {
              if (var.index() == oneSeedPosition) {
                args.push_back(one);
              } else if (var.index() == alphaSeedPosition) {
                args.push_back(jacobianFunction.getAlpha());
              } else {
                args.push_back(zero);
              }
            }
          }

          for (size_t i = 0; i < jacobianFunction.getEquationIndices().size(); ++i) {
            args.push_back(zero);
          }

          auto templateCall = builder.create<CallOp>(
              jacobianFunction.getLoc(), templateName, RealType::get(builder.getContext()), args);

          builder.create<mlir::ida::ReturnOp>(jacobianFunction.getLoc(), templateCall.getResult(0));

          // Add the Jacobian function to the IDA instance
          builder.restoreInsertionPoint(insertionPoint);

          builder.create<mlir::ida::AddJacobianOp>(
              equation->getOperation().getLoc(),
              idaInstance,
              idaEquation,
              mappedVariables.lookup(variable),
              jacobianFunctionName);
        }
      }

      model.getOperation()->getParentOfType<mlir::ModuleOp>().dump();
    }

    return mlir::success();
  }
}
