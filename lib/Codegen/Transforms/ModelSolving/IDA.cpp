#include "marco/Codegen/Transforms/Model/IDA.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "mlir/IR/AffineMap.h"
#include <queue>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;

namespace marco::codegen
{
  IDASolver::IDASolver() = default;

  bool IDASolver::isEnabled() const
  {
    // TODO should depend on the compiler's flags
    return true;
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

  void IDASolver::processInitFunction(mlir::OpBuilder& builder, const Model<ScheduledEquationsBlock>& model, mlir::FuncOp initFunction, const mlir::BlockAndValueMapping& derivatives)
  {
    auto modelOp = model.getOperation();
    auto loc = initFunction.getLoc();
    auto terminator = mlir::cast<YieldOp>(initFunction.getBody().back().getTerminator());

    // Create the IDA instance.
    // To do so, we need to first compute the total number of scalar variables that IDA
    // has to manage. Such number is equal to the number of scalar equations.
    size_t numberOfScalarEquations = 0;

    for (const auto& equation : equations) {
      numberOfScalarEquations += equation->getIterationRanges().flatSize();
    }

    mlir::Value ida = builder.create<mlir::ida::CreateOp>(loc, builder.getI64IntegerAttr(numberOfScalarEquations));

    // Change the ownership of the variables managed to IDA
    mlir::BlockAndValueMapping idaVariables;

    for (const auto& variable : variables) {
      mlir::Value value = terminator.values()[variable.cast<mlir::BlockArgument>().getArgNumber()];
      auto memberCreateOp = value.getDefiningOp<MemberCreateOp>();
      auto isState = derivatives.contains(variable);
      builder.setInsertionPoint(memberCreateOp);

      auto memberType = memberCreateOp.resultType().cast<MemberType>();
      auto dimensions = memberType.toArrayType().getShape();

      auto idaVariable = builder.create<mlir::ida::AddVariableOp>(
          memberCreateOp.getLoc(), ida,
          builder.getI64ArrayAttr(dimensions),
          builder.getBoolAttr(isState));

      idaVariables.map(variable, idaVariable);

      auto castedVariable = builder.create<mlir::ida::GetVariableOp>(
          memberCreateOp.getLoc(),
          memberType.toArrayType(),
          ida, idaVariable);

      memberCreateOp->replaceAllUsesWith(castedVariable);
      memberCreateOp.erase();
    }

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
    builder.setInsertionPoint(terminator);

    size_t residualFunctionsCounter = 0;
    size_t jacobianFunctionsCounter = 0;

    for (const auto& equation : independentEquations) {
      equation->dumpIR();
      auto ranges = equation->getIterationRanges();
      std::vector<mlir::Attribute> rangesAttr;

      for (size_t i = 0; i < ranges.rank(); ++i) {
        rangesAttr.push_back(builder.getI64ArrayAttr({ ranges[i].getBegin(), ranges[i].getEnd() }));
      }

      auto idaEquation = builder.create<mlir::ida::AddEquationOp>(
          equation->getOperation().getLoc(),
          ida,
          builder.getArrayAttr(rangesAttr));

      for (const auto& access : equation->getAccesses()) {
        mlir::Value idaVariable = idaVariables.lookup(access.getVariable()->getValue());
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
            ida, idaEquation, idaVariable,
            mlir::AffineMap::get(accessFunction.size(), 0, expressions, builder.getContext()));

        auto residualFunctionName = "residualFunction" + std::to_string(residualFunctionsCounter++);
        auto jacobianFunctionName = "jacobianFunction" + std::to_string(jacobianFunctionsCounter++);

        {
          mlir::OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(equation->getOperation()->getParentOfType<mlir::ModuleOp>().getBody());
          builder.create<mlir::ida::ResidualFunctionOp>(equation->getOperation().getLoc(), residualFunctionName);
          builder.create<mlir::ida::JacobianFunctionOp>(equation->getOperation().getLoc(), jacobianFunctionName);
        }

        builder.create<mlir::ida::AddResidualOp>(equation->getOperation().getLoc(), ida, idaEquation, residualFunctionName);
        builder.create<mlir::ida::AddJacobianOp>(equation->getOperation().getLoc(), ida, idaEquation, jacobianFunctionName);
      }
    }
  }
}
