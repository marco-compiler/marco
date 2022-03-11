#include "marco/Codegen/Transforms/Model/IDA.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "mlir/IR/AffineMap.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;

namespace marco::codegen
{
  IDASolver::IDASolver() = default;

  bool IDASolver::isEnabled() const
  {
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

  void IDASolver::addEquation(ScheduledEquation* equation)
  {
    equations.emplace(equation);
  }

  void IDASolver::processInitFunction(mlir::OpBuilder& builder, Model<ScheduledEquationsBlock> model, mlir::FuncOp initFunction, const mlir::BlockAndValueMapping& derivatives)
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

    // Set inside IDA the information about the equations
    builder.setInsertionPoint(terminator);

    size_t residualFunctionsCounter = 0;
    size_t jacobianFunctionsCounter = 0;

    for (const auto& equation : equations) {
      auto ranges = equation->getIterationRanges();
      std::vector<mlir::Attribute> rangesAttr;

      for (size_t i = 0; i < ranges.rank(); ++i) {
        rangesAttr.push_back(builder.getI64ArrayAttr({ ranges[i].getBegin(), ranges[i].getEnd() }));
      }

      auto idaEquation = builder.create<mlir::ida::AddEquationOp>(
          equation->getOperation().getLoc(),
          ida,
          builder.getArrayAttr(rangesAttr));

      // TODO sostituire variabili non IDA

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
