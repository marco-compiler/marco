#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "marco/Codegen/Conversion/Modelica/TypeConverter.h"
#include "marco/Codegen/dialects/modelica/ModelicaBuilder.h"
#include "marco/Codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving.h"
#include "marco/Codegen/Transforms/Model/Cycles.h"
#include "marco/Codegen/Transforms/Model/Equation.h"
#include "marco/Codegen/Transforms/Model/EquationImpl.h"
#include "marco/Codegen/Transforms/Model/Matching.h"
#include "marco/Codegen/Transforms/Model/Model.h"
#include "marco/Codegen/Transforms/Model/ModelConverter.h"
#include "marco/Codegen/Transforms/Model/Scheduling.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation.h"
#include "marco/Utils/VariableFilter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>
#include <map>
#include <memory>
#include <set>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;
using namespace ::marco::modeling;

/// Remove the derivative operations by replacing them with appropriate
/// buffers, and set the derived variables as state variables.
static mlir::LogicalResult removeDerivatives(
    mlir::OpBuilder& builder, Model<Equation>& model, mlir::BlockAndValueMapping& derivatives)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  auto appendIndexesFn = [](llvm::SmallVectorImpl<mlir::Value>& destination, mlir::ValueRange indexes) {
    for (size_t i = 0, e = indexes.size(); i < e; ++i) {
      mlir::Value index = indexes[e - 1 - i];
      destination.push_back(index);
    }
  };

  auto derivativeOrder = [&](mlir::Value value) -> unsigned int {
    auto inverseMap = derivatives.getInverse();
    unsigned int result = 0;

    while (inverseMap.contains(value)) {
      ++result;
      value = inverseMap.lookup(value);
    }

    return result;
  };

  model.getOperation().walk([&](DerOp op) {
    mlir::Location loc = op->getLoc();
    mlir::Value operand = op.operand();

    // If the value to be derived belongs to an array, then also the derived
    // value is stored within an array. Thus, we need to store its position.

    llvm::SmallVector<mlir::Value, 3> subscriptions;

    while (!operand.isa<mlir::BlockArgument>()) {
      mlir::Operation* definingOp = operand.getDefiningOp();
      assert(mlir::isa<LoadOp>(definingOp) || mlir::isa<SubscriptionOp>(definingOp));

      if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
        appendIndexesFn(subscriptions, loadOp.indexes());
        operand = loadOp.memory();
      } else {
        auto subscriptionOp = mlir::cast<SubscriptionOp>(definingOp);
        appendIndexesFn(subscriptions, subscriptionOp.indexes());
        operand = subscriptionOp.source();
      }
    }

    if (!derivatives.contains(operand)) {
      auto model = op->getParentOfType<ModelOp>();
      auto terminator = mlir::cast<YieldOp>(model.init().back().getTerminator());
      builder.setInsertionPoint(terminator);

      size_t index = operand.cast<mlir::BlockArgument>().getArgNumber();
      auto memberCreateOp = terminator.values()[index].getDefiningOp<MemberCreateOp>();
      auto nextDerName = getNextFullDerVariableName(memberCreateOp.name(), derivativeOrder(operand) + 1);

      assert(operand.getType().isa<ArrayType>());
      auto arrayType = operand.getType().cast<ArrayType>().toElementType(RealType::get(builder.getContext()));

      // Create the member and initialize it
      auto memberType = MemberType::get(arrayType.toAllocationScope(BufferAllocationScope::heap));
      mlir::Value memberDer = builder.create<MemberCreateOp>(loc, nextDerName, memberType, memberCreateOp.dynamicDimensions(), memberCreateOp.isConstant());
      mlir::Value zero = builder.create<ConstantOp>(loc, RealAttribute::get(builder.getContext(), 0));
      mlir::Value array = builder.create<MemberLoadOp>(loc, memberType.unwrap(), memberDer);
      builder.create<FillOp>(loc, zero, array);

      // Update the terminator values
      llvm::SmallVector<mlir::Value, 3> args(terminator.values().begin(), terminator.values().end());
      args.push_back(memberDer);
      builder.create<YieldOp>(loc, args);
      terminator.erase();

      // Add the new argument to the body of the model
      auto bodyArgument = model.body().addArgument(arrayType);
      derivatives.map(operand, bodyArgument);
    }

    builder.setInsertionPoint(op);
    mlir::Value derVar = derivatives.lookup(operand);

    llvm::SmallVector<mlir::Value, 3> reverted(subscriptions.rbegin(), subscriptions.rend());

    if (!subscriptions.empty()) {
      derVar = builder.create<SubscriptionOp>(loc, derVar, reverted);
    }

    if (auto arrayType = derVar.getType().cast<ArrayType>(); arrayType.getRank() == 0) {
      derVar = builder.create<LoadOp>(loc, derVar);
    }

    op.replaceAllUsesWith(derVar);
    op.erase();
  });

  return mlir::success();
}

/// Get all the variables that are declared inside the Model operation, independently
/// from their nature (state variables, constants, etc.).
static Variables discoverVariables(ModelOp model)
{
  Variables result;

  mlir::ValueRange vars = model.body().getArguments();

  for (size_t i = 1; i < vars.size(); ++i) {
    result.add(std::make_unique<Variable>(vars[i]));
  }

  return result;
}

/// Get the equations that are declared inside the Model operation.
static Equations<Equation> discoverEquations(ModelOp model, const Variables& variables)
{
  Equations<Equation> result;

  model.walk([&](EquationOp equationOp) {
    result.add(Equation::build(equationOp, variables));
  });

  return result;
}

/// Model solving pass.
/// Its objective is to convert a descriptive (and thus not sequential) model into an
/// algorithmic one and to create the functions to be called during the simulation.
class SolveModelPass: public mlir::PassWrapper<SolveModelPass, mlir::OperationPass<ModelOp>>
{
	public:
    explicit SolveModelPass(SolveModelOptions options, unsigned int bitWidth)
        : options(std::move(options)),
          bitWidth(std::move(bitWidth))
    {
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override
    {
      registry.insert<ModelicaDialect>();
      registry.insert<mlir::scf::SCFDialect>();
      registry.insert<mlir::LLVM::LLVMDialect>();
    }

    void runOnOperation() override
    {
      auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
      Model<Equation> model(getOperation());
      mlir::OpBuilder builder(model.getOperation());

      // Remove the derivative operations and allocate the appropriate memory buffers
      mlir::BlockAndValueMapping derivatives;

      if (mlir::failed(removeDerivatives(builder, model, derivatives))) {
        model.getOperation().emitError("Derivatives could not be converted to variables");
        return signalPassFailure();
      }

      // Now that the additional variables have been created, we can start a discovery process
      model.setVariables(discoverVariables(model.getOperation()));
      model.setEquations(discoverEquations(model.getOperation(), model.getVariables()));

      // Matching process
      Model<MatchedEquation> matchedModel(model.getOperation());

      if (mlir::failed(match(matchedModel, model, derivatives))) {
        return signalPassFailure();
      }

      // Resolve the algebraic loops
      if (mlir::failed(solveCycles(matchedModel, builder))) {
        // TODO Check if the selected solver can deal with cycles. If not, fail.
        return signalPassFailure();
      }

      // Schedule the equations
      Model<ScheduledEquationsBlock> scheduledModel(matchedModel.getOperation());

      if (mlir::failed(schedule(scheduledModel, matchedModel))) {
        return signalPassFailure();
      }

      // Create the simulation functions
      mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
      TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);
      ModelConverter modelConverter(options, typeConverter);

      if (auto status = modelConverter.convert(builder, scheduledModel, derivatives); mlir::failed(status)) {
        return signalPassFailure();
      }
    }

	private:
	  SolveModelOptions options;
	  unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> marco::codegen::createSolveModelPass(SolveModelOptions options, unsigned int bitWidth)
{
	return std::make_unique<SolveModelPass>(options, bitWidth);
}
