#include "marco/Codegen/Transforms/ModelSolving/ModelSolving.h"
#include "marco/Codegen/Transforms/ModelSolving/Cycles.h"
#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/ModelConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/TypeConverter.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/Common.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Utils/VariableFilter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <map>
#include <memory>
#include <set>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

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
        operand = loadOp.array();
      } else {
        auto subscriptionOp = mlir::cast<SubscriptionOp>(definingOp);
        appendIndexesFn(subscriptions, subscriptionOp.indices());
        operand = subscriptionOp.source();
      }
    }

    if (!derivatives.contains(operand)) {
      auto model = op->getParentOfType<ModelOp>();
      auto terminator = mlir::cast<YieldOp>(model.initRegion().back().getTerminator());
      builder.setInsertionPoint(terminator);

      size_t index = operand.cast<mlir::BlockArgument>().getArgNumber();
      auto memberCreateOp = terminator.values()[index].getDefiningOp<MemberCreateOp>();
      auto nextDerName = getNextFullDerVariableName(memberCreateOp.name(), derivativeOrder(operand) + 1);

      assert(operand.getType().isa<ArrayType>());
      auto arrayType = operand.getType().cast<ArrayType>().toElementType(RealType::get(builder.getContext()));

      // Create the member and initialize it
      auto memberType = MemberType::wrap(arrayType);
      mlir::Value memberDer = builder.create<MemberCreateOp>(loc, nextDerName, memberType, memberCreateOp.dynamicSizes());
      mlir::Value zero = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));
      mlir::Value array = builder.create<MemberLoadOp>(loc, memberType.unwrap(), memberDer);
      builder.create<ArrayFillOp>(loc, array, zero);

      // Update the terminator values
      llvm::SmallVector<mlir::Value, 3> args(terminator.values().begin(), terminator.values().end());
      args.push_back(memberDer);
      builder.create<YieldOp>(loc, args);
      terminator.erase();

      // Add the new argument to the body of the model
      auto bodyArgument = model.bodyRegion().addArgument(arrayType);
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

  for (const auto& var : model.bodyRegion().getArguments()) {
    result.add(std::make_unique<Variable>(var));
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

namespace
{
  /// Model solving pass.
  /// Its objective is to convert a descriptive (and thus not sequential) model into an
  /// algorithmic one and to create the functions to be called during the simulation.
  class ModelSolvingPass: public mlir::PassWrapper<ModelSolvingPass, mlir::OperationPass<ModelOp>>
  {
    public:
      explicit ModelSolvingPass(ModelSolvingOptions options, unsigned int bitWidth)
          : options(std::move(options)),
            bitWidth(std::move(bitWidth))
      {
      }

      void getDependentDialects(mlir::DialectRegistry& registry) const override
      {
        registry.insert<ModelicaDialect>();
        registry.insert<mlir::ida::IDADialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
      }

      void runOnOperation() override
      {
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
          if (options.solver != Solver::ida) {
            // Check if the selected solver can deal with cycles. If not, fail.
            return signalPassFailure();
          }
        }

        // Schedule the equations
        Model<ScheduledEquationsBlock> scheduledModel(matchedModel.getOperation());

        if (mlir::failed(schedule(scheduledModel, matchedModel))) {
          return signalPassFailure();
        }

        // Create the simulation functions
        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        marco::codegen::TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);
        ModelConverter modelConverter(options, typeConverter);

        if (auto status = modelConverter.convert(builder, scheduledModel, derivatives); mlir::failed(status)) {
          return signalPassFailure();
        }
      }

    private:
      ModelSolvingOptions options;
      unsigned int bitWidth;
  };
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createModelSolvingPass(ModelSolvingOptions options, unsigned int bitWidth)
  {
    return std::make_unique<ModelSolvingPass>(options, bitWidth);
  }
}