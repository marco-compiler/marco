#include "marco/Codegen/Transforms/ModelSolving/ModelSolving.h"
#include "marco/Codegen/Transforms/ModelSolving/Cycles.h"
#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/ModelConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/TypeConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/VariablesMap.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/Common.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/VariableFilter/VariableFilter.h"
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
#include <queue>
#include <set>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  struct EquationOpMultipleValuesPattern : public mlir::OpRewritePattern<EquationOp>
  {
    using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());

      if (terminator.lhsValues().size() != terminator.rhsValues().size()) {
        return rewriter.notifyMatchFailure(op, "Different amount of values in left-hand and right-hand sides of the equation");
      }

      auto amountOfValues = terminator.lhsValues().size();

      for (size_t i = 0; i < amountOfValues; ++i) {
        rewriter.setInsertionPointAfter(op);

        auto clone = rewriter.create<EquationOp>(loc);
        assert(clone.bodyRegion().empty());
        mlir::Block* cloneBodyBlock = rewriter.createBlock(&clone.bodyRegion());
        rewriter.setInsertionPointToStart(cloneBodyBlock);

        mlir::BlockAndValueMapping mapping;

        for (auto& originalOp : op.bodyBlock()->getOperations()) {
          if (mlir::isa<EquationSideOp>(originalOp)) {
            continue;
          }

          if (mlir::isa<EquationSidesOp>(originalOp)) {
            auto lhsOp = mlir::cast<EquationSideOp>(terminator.lhs().getDefiningOp());
            auto rhsOp = mlir::cast<EquationSideOp>(terminator.rhs().getDefiningOp());

            auto newLhsOp = rewriter.create<EquationSideOp>(lhsOp.getLoc(), mapping.lookup(terminator.lhsValues()[i]));
            auto newRhsOp = rewriter.create<EquationSideOp>(rhsOp.getLoc(), mapping.lookup(terminator.rhsValues()[i]));

            rewriter.create<EquationSidesOp>(terminator.getLoc(), newLhsOp, newRhsOp);
          } else {
            rewriter.clone(originalOp, mapping);
          }
        }
      }

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };
}

static void collectDerivedVariablesIndices(std::map<unsigned int, IndexSet>& derivedIndices, const Equations<Equation>& equations)
{
  for (const auto& equation : equations) {
    auto accesses = equation->getAccesses();

    equation->getOperation().walk([&](DerOp derOp) {
      auto it = llvm::find_if(accesses, [&](const auto& access) {
        auto value = equation->getValueAtPath(access.getPath());
        return value == derOp.operand();
      });

      assert(it != accesses.end());
      const auto& access = *it;
      auto indices = access.getAccessFunction().map(equation->getIterationRanges());
      auto argNumber = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
      derivedIndices[argNumber] += indices;
    });
  }
}

static void collectDerivedVariables(std::set<unsigned int>& derivedVariables, const Equations<Equation>& equations)
{
  for (const auto& equation : equations) {
    auto accesses = equation->getAccesses();

    equation->getOperation().walk([&](DerOp derOp) {
      auto it = llvm::find_if(accesses, [&](const auto& access) {
        auto value = equation->getValueAtPath(access.getPath());
        return value == derOp.operand();
      });

      assert(it != accesses.end());
      const auto& access = *it;
      auto argNumber = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
      derivedVariables.insert(argNumber);
    });
  }
}

static unsigned int addArgumentToRegions(llvm::ArrayRef<mlir::Region*> regions, mlir::Type type)
{
  assert(!regions.empty());

  assert(llvm::all_of(regions, [&](const auto& region) {
    return region->getNumArguments() == regions[0]->getNumArguments();
  }));

  unsigned int newArgNumber = regions[0]->getNumArguments();

  for (auto& region : regions) {
    region->addArgument(type);
  }

  return newArgNumber;
}

static mlir::modelica::EquationOp cloneEquationWithNewIndices(
    mlir::OpBuilder& builder,
    const Equation& equation,
    const MultidimensionalRange& indices,
    mlir::BlockAndValueMapping& mapping)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto equationOp = equation.getOperation();
  builder.setInsertionPointAfter(equationOp.getOperation());
  ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();

  while (parent != nullptr) {
    builder.setInsertionPointAfter(parent.getOperation());
    parent = parent->getParentOfType<ForEquationOp>();
  }

  auto oldIterationVariables = equation.getInductionVariables();
  assert(indices.rank() >= oldIterationVariables.size());

  for (size_t i = 0; i < oldIterationVariables.size(); ++i) {
    auto loop = builder.create<ForEquationOp>(oldIterationVariables[i].getLoc(), indices[i].getBegin(), indices[i].getEnd() - 1);
    builder.setInsertionPointToStart(loop.bodyBlock());
    mapping.map(oldIterationVariables[i], loop.induction());
  }

  return mlir::cast<EquationOp>(builder.clone(*equationOp.getOperation(), mapping));
}

static std::vector<long> variableRangeToShape(ArrayType arrayType, MultidimensionalRange range)
{
  std::vector<long> result;

  for (size_t i = 0, e = std::min(range.rank(), arrayType.getRank()); i < e; ++i) {
    auto dimension = range[i].size();
    result.push_back(dimension);
  }

  return result;
}

static void eraseValueInsideEquation(mlir::Value value)
{
  std::queue<mlir::Value> queue;
  queue.push(value);

  while (!queue.empty()) {
    std::vector<mlir::Value> valuesWithUses;
    mlir::Value current = queue.front();

    while (current != nullptr && !current.use_empty()) {
      valuesWithUses.push_back(current);
      queue.pop();

      if (queue.empty()) {
        current = nullptr;
      } else {
        current = queue.front();
      }
    }

    for (const auto& valueWithUses : valuesWithUses) {
      queue.push(valueWithUses);
    }

    if (current != nullptr) {
      assert(current.use_empty());

      if (auto op = current.getDefiningOp()) {
        for (auto operand : op->getOperands()) {
          queue.push(operand);
        }

        op->erase();
      }
    }

    queue.pop();
  }
}

static mlir::LogicalResult splitVariables(
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    VariablesMap& variablesMap,
    const std::map<unsigned int, IndexSet>& derivedIndices)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Collect the regions to be modified
  llvm::SmallVector<mlir::Region*, 2> regions;
  regions.push_back(&modelOp.equationsRegion());
  regions.push_back(&modelOp.initialEquationsRegion());

  // Set the insertion point to the end of the init block, as we will need to declare new members
  auto terminator = mlir::cast<YieldOp>(modelOp.initRegion().back().getTerminator());
  builder.setInsertionPoint(terminator);

  llvm::SmallVector<mlir::Value, 3> initRegionTerminatorValues;

  std::map<unsigned int, std::vector<SplitVariable>> splitVariables;
  auto variableNames = modelOp.variableNames();
  auto variablesTypes = regions[0]->getArgumentTypes();

  // Ensure that all regions have the same arguments
  assert(llvm::all_of(regions, [&](const auto& region) {
    return region->getArgumentTypes() == variablesTypes;
  }));

  for (const auto& name : llvm::enumerate(variableNames)) {
    auto arg = regions[0]->getArgument(name.index());
    unsigned int argNumber = arg.getArgNumber();
    auto variableArrayType = arg.getType().cast<ArrayType>();
    assert(variableArrayType.hasConstantShape());

    std::vector<Range> dimensions;

    if (variableArrayType.isScalar()) {
      dimensions.emplace_back(0, 1);
    } else {
      for (const auto& dimension : variableArrayType.getShape()) {
        dimensions.emplace_back(0, dimension);
      }
    }

    MultidimensionalRange allIndices(std::move(dimensions));
    IndexSet algebraicIndices(allIndices);

    size_t variableNameCounter = 0;

    auto addArgumentFn = [&](unsigned int baseArgNumber, const MultidimensionalRange& mappedIndices) -> unsigned int {
      builder.setInsertionPoint(terminator);

      auto memberCreateOp = terminator.values()[baseArgNumber].getDefiningOp<MemberCreateOp>();

      auto newArgArrayType = ArrayType::get(
          builder.getContext(), variableArrayType.getElementType(),
          variableRangeToShape(variableArrayType, mappedIndices));

      auto newArgNumber = addArgumentToRegions(regions, newArgArrayType);
      std::string newArgName = memberCreateOp.name().str() + "_" + std::to_string(variableNameCounter++);

      auto newMemberOp = builder.create<MemberCreateOp>(
          modelOp.getLoc(), newArgName,
          memberCreateOp.getMemberType().withShape(newArgArrayType.getShape()),
          llvm::None);

      initRegionTerminatorValues.push_back(newMemberOp);

      // Copy the original values
      mlir::Value source = builder.create<MemberLoadOp>(newMemberOp.getLoc(), memberCreateOp);

      if (newArgArrayType.isScalar()) {
        builder.create<MemberStoreOp>(newMemberOp.getLoc(), newMemberOp, source);
      } else {
        mlir::Value destination = builder.create<MemberLoadOp>(newMemberOp.getLoc(), newMemberOp);

        std::vector<mlir::Value> lowerBounds;
        std::vector<mlir::Value> upperBounds;
        std::vector<mlir::Value> steps;

        for (size_t i = 0; i < newArgArrayType.getRank(); ++i) {
          lowerBounds.push_back(builder.create<ConstantOp>(newMemberOp.getLoc(), builder.getIndexAttr(mappedIndices[i].getBegin())));
          upperBounds.push_back(builder.create<ConstantOp>(newMemberOp.getLoc(), builder.getIndexAttr(mappedIndices[i].getEnd())));
          steps.push_back(builder.create<ConstantOp>(newMemberOp.getLoc(), builder.getIndexAttr(1)));
        }

        mlir::scf::buildLoopNest(
            builder, newMemberOp.getLoc(), lowerBounds, upperBounds, steps,
            [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange indices) {
              std::vector<mlir::Value> destinationIndices;

              for (size_t i = 0; i < indices.size(); ++i) {
                mlir::Value offset = builder.create<ConstantOp>(loc, builder.getIndexAttr(-1 * mappedIndices[i].getBegin()));
                mlir::Value index = builder.create<mlir::AddIOp>(loc, builder.getIndexType(), indices[i], offset);
                destinationIndices.push_back(index);
              }

              mlir::Value sourceScalar = nestedBuilder.create<LoadOp>(loc, source, indices);
              builder.create<StoreOp>(loc, sourceScalar, destination, destinationIndices);
            });
      }

      return newArgNumber;
    };

    if (auto derivedVariableIndices = derivedIndices.find(argNumber); derivedVariableIndices != derivedIndices.end()) {
      for (const auto& derivedRange : derivedVariableIndices->second) {
        auto newArgNumber = addArgumentFn(argNumber, derivedRange);
        splitVariables[argNumber].push_back(SplitVariable(derivedRange, newArgNumber));
        algebraicIndices -= derivedRange;
      }
    }

    for (const auto& range : algebraicIndices) {
      auto newArgNumber = addArgumentFn(argNumber, range);
      splitVariables[argNumber].push_back(SplitVariable(range, newArgNumber));
    }
  }

  // Replace the old variables usages
  for (auto& region : regions) {
    Variables variables;

    // Consider only the variables to be replaced, so that we can identify right
    // just the accesses using them.

    for (const auto& argument : region->getArguments()) {
      if (splitVariables.find(argument.getArgNumber()) != splitVariables.end()) {
        variables.add(std::make_unique<Variable>(argument));
      }
    }

    for (auto& originalEquation : discoverEquations(*region, variables)) {
      // The equations to be processed
      std::queue<std::unique_ptr<Equation>> equations;

      // Add the original equation to the processing queue
      equations.push(Equation::build(originalEquation->cloneIR(), originalEquation->getVariables()));
      originalEquation->eraseIR();

      while (!equations.empty()) {
        auto& equation = equations.front();

        if (auto accesses = equation->getAccesses(); !accesses.empty()) {
          // We need to process the accesses one by one, because otherwise we would wrongly
          // create duplicates of the equation itself.
          const auto& access = accesses[0];

          auto equationIndices = IndexSet(equation->getIterationRanges());
          auto inductionVariables = equation->getInductionVariables();

          auto variable = access.getVariable()->getValue();
          auto requestedIndices = access.getAccessFunction().map(equationIndices);

          auto argNumber = variable.cast<mlir::BlockArgument>().getArgNumber();
          assert(splitVariables.find(argNumber) != splitVariables.end());

          for (const auto& splitVariable : splitVariables[argNumber]) {
            const auto& availableIndices = splitVariable.getIndices();

            if (!requestedIndices.overlaps(availableIndices)) {
              continue;
            }

            auto coveredIndices = requestedIndices.intersect(availableIndices);
            auto newEquationIndices = access.getAccessFunction().inverseMap(coveredIndices, equationIndices);

            for (const auto& newRange : newEquationIndices) {
              mlir::BlockAndValueMapping mapping;
              auto cloneOp = cloneEquationWithNewIndices(builder, *equation, newRange, mapping);
              auto clone = Equation::build(cloneOp, equation->getVariables());
              auto newIterationVariables = clone->getInductionVariables();

              mlir::Value mappedUsage = mapping.lookup(equation->getValueAtPath(access.getPath()));
              builder.setInsertionPointAfterValue(mappedUsage);

              mlir::Value replacement = region->getArgument(splitVariable.getArgNumber());
              size_t rank = replacement.getType().cast<ArrayType>().getRank();
              assert(rank <= access.getAccessFunction().size());
              std::vector<mlir::Value> indices;

              for (size_t i = 0; i < rank; ++i) {
                const auto& dimensionAccess = access.getAccessFunction()[i];

                if (dimensionAccess.isConstantAccess()) {
                  mlir::Value index = builder.create<ConstantOp>(
                      replacement.getLoc(),
                      builder.getIndexAttr(dimensionAccess.getPosition() - splitVariable.getIndices()[i].getBegin()));

                  indices.push_back(index);
                } else {
                  mlir::Value inductionVar = mapping.lookup(inductionVariables[dimensionAccess.getInductionVariableIndex()]);

                  mlir::Value offset = builder.create<ConstantOp>(
                      replacement.getLoc(),
                      builder.getIndexAttr(dimensionAccess.getOffset() - splitVariable.getIndices()[i].getBegin()));

                  mlir::Value index = builder.create<AddOp>(replacement.getLoc(), builder.getIndexType(), inductionVar, offset);
                  indices.push_back(index);
                }
              }

              replacement = builder.create<LoadOp>(replacement.getLoc(), replacement, indices);
              mappedUsage.replaceAllUsesWith(replacement);
              eraseValueInsideEquation(mappedUsage);

              equations.push(std::move(clone));
            }
          }

          equation->eraseIR();
        }

        equations.pop();
      }
    }
  }

  // Erase the original arguments that are not used anymore
  llvm::SmallVector<unsigned int> erasedArgs;
  std::map<unsigned int, unsigned int> mappedArgsAfterPruning;
  unsigned int lastPosition = 0;

  for (unsigned int i = 0; i < regions[0]->getNumArguments(); ++i) {
    if (splitVariables.find(i) == splitVariables.end()) {
      mappedArgsAfterPruning[i] = lastPosition++;
    } else {
      erasedArgs.push_back(i);
    }
  }

  for (const auto& name : llvm::enumerate(variableNames)) {
    if (auto it = splitVariables.find(name.index()); it != splitVariables.end()) {
      for (const auto& splitVariable : it->second) {
        auto newArgNumber = mappedArgsAfterPruning.find(splitVariable.getArgNumber());
        assert(newArgNumber != mappedArgsAfterPruning.end());
        SplitVariable variable(splitVariable.getIndices(), newArgNumber->second);
        variablesMap.add(name.value(), std::move(variable));
      }
    } else {
      auto newArgNumber = mappedArgsAfterPruning.find(name.index());
      assert(newArgNumber != mappedArgsAfterPruning.end());

      auto var = std::make_unique<Variable>(regions[0]->getArgument(name.index()));
      std::vector<Range> dimensions;

      for (size_t i = 0, e = var->getRank(); i < e; ++i) {
        dimensions.emplace_back(0, var->getDimensionSize(i));
      }

      SplitVariable splitVariable(MultidimensionalRange(std::move(dimensions)), newArgNumber->second);
      variablesMap.add(name.value(), std::move(splitVariable));
    }
  }

  // Erase in reverse order so that the arguments numbers don't get invalidated
  for (auto it = erasedArgs.rbegin(); it != erasedArgs.rend(); ++it) {
    for (auto& region : regions) {
      region->eraseArgument(*it);
    }
  }

  // Update the terminator with the new values
  builder.setInsertionPoint(terminator);
  builder.create<YieldOp>(terminator.getLoc(), initRegionTerminatorValues);
  terminator.erase();

  return mlir::success();
}

static mlir::LogicalResult createDerivativeVariables(
    mlir::OpBuilder& builder,
    ModelOp modelOp,
    DerivativesMap& derivativesMap,
    const std::set<unsigned int>& derivedVariables)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Collect the regions to be modified
  llvm::SmallVector<mlir::Region*, 2> regions;
  regions.push_back(&modelOp.equationsRegion());
  regions.push_back(&modelOp.initialEquationsRegion());

  // Ensure that all regions have the same arguments
  assert(llvm::all_of(regions, [&](const auto& region) {
    return region->getArgumentTypes() == regions[0]->getArgumentTypes();
  }));

  // The list of the new variables
  std::vector<mlir::Value> variables;
  auto terminator = mlir::cast<YieldOp>(modelOp.initRegion().back().getTerminator());

  for (auto variable : terminator.values()) {
    variables.push_back(variable);
  }

  // Create the new variables for the derivatives
  llvm::SmallVector<unsigned int> derivedVariablesOrdered;

  for (const auto& argNumber : derivedVariables) {
    derivedVariablesOrdered.push_back(argNumber);
  }

  llvm::sort(derivedVariablesOrdered);

  for (const auto& argNumber : derivedVariablesOrdered) {
    auto variable = terminator.values()[argNumber];
    auto memberCreateOp = variable.getDefiningOp<MemberCreateOp>();
    auto variableMemberType = memberCreateOp.getMemberType();

    auto derType = ArrayType::get(builder.getContext(), RealType::get(builder.getContext()), variableMemberType.getShape());
    assert(derType.hasConstantShape());
    auto derArgNumber = addArgumentToRegions(regions, derType);
    derivativesMap.setDerivative(argNumber, derArgNumber);

    // Create the variable and initialize it at zero
    builder.setInsertionPoint(terminator);

    auto derivativeName = getNextFullDerVariableName(memberCreateOp.name(), 1);

    auto derMemberOp = builder.create<MemberCreateOp>(
        memberCreateOp.getLoc(), derivativeName, MemberType::wrap(derType), llvm::None);

    variables.push_back(derMemberOp);

    mlir::Value zero = builder.create<ConstantOp>(derMemberOp.getLoc(), RealAttr::get(builder.getContext(), 0));
    mlir::Value derivative = builder.create<MemberLoadOp>(derMemberOp.getLoc(), derMemberOp);
    builder.create<ArrayFillOp>(derMemberOp.getLoc(), derivative, zero);
  }

  builder.create<YieldOp>(terminator.getLoc(), variables);
  terminator.erase();

  return mlir::success();
}

static mlir::LogicalResult removeDerOps(mlir::OpBuilder& builder, Model<Equation>& model)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto& derivativesMap = model.getDerivativesMap();

  std::vector<mlir::Value> variables;
  variables.resize(model.getVariables().size());

  for (const auto& variable : model.getVariables()) {
    variables[variable->getValue().cast<mlir::BlockArgument>().getArgNumber()] = variable->getValue();
  }

  for (auto& equation : model.getEquations()) {
    std::vector<DerOp> derOps;

    equation->getOperation().walk([&](DerOp derOp) {
      derOps.push_back(derOp);
    });

    if (!derOps.empty()) {
      auto inductionVariables = equation->getInductionVariables();
      auto accesses = equation->getAccesses();

      for (auto& derOp : derOps) {
        builder.setInsertionPoint(derOp);

        auto access = llvm::find_if(accesses, [&](const auto& access) {
          auto value = equation->getValueAtPath(access.getPath());
          return value == derOp.operand();
        });

        assert(access != accesses.end());
        auto variable = access->getVariable()->getValue();
        auto varArgNumber = variable.cast<mlir::BlockArgument>().getArgNumber();
        const auto& accessFunction = access->getAccessFunction();

        auto derArgNumber = derivativesMap.getDerivative(varArgNumber);
        mlir::Value replacement = model.getVariables()[derArgNumber]->getValue();

        // Check that all the requested indices are contained within the derivative variable
        assert(llvm::all_of(accessFunction.map(equation->getIterationRanges()), [&](const auto& indices) {
          auto arrayType = replacement.getType().cast<ArrayType>();

          if (arrayType.isScalar()) {
            return llvm::all_of(indices, [](const auto& index) {
              return index == 0;
            });
          } else {
            auto shape = arrayType.getShape();

            for (size_t i = 0; i < arrayType.getRank(); ++i) {
              if (indices[i] >= shape[i]) {
                return false;
              }
            }
          }

          return true;
        }));

        size_t rank = replacement.getType().cast<ArrayType>().getRank();
        assert(rank <= access->getAccessFunction().size());
        std::vector<mlir::Value> indices;

        for (size_t i = 0; i < rank; ++i) {
          const auto& dimensionAccess = access->getAccessFunction()[i];

          if (dimensionAccess.isConstantAccess()) {
            mlir::Value index = builder.create<ConstantOp>(replacement.getLoc(), builder.getIndexAttr(dimensionAccess.getPosition()));
            indices.push_back(index);
          } else {
            mlir::Value inductionVar = inductionVariables[dimensionAccess.getInductionVariableIndex()];
            mlir::Value offset = builder.create<ConstantOp>(replacement.getLoc(), builder.getIndexAttr(dimensionAccess.getOffset()));
            mlir::Value index = builder.create<AddOp>(replacement.getLoc(), builder.getIndexType(), inductionVar, offset);
            indices.push_back(index);
          }
        }

        replacement = builder.create<LoadOp>(replacement.getLoc(), replacement, indices);
        derOp.replaceAllUsesWith(replacement);
        eraseValueInsideEquation(derOp.getResult());
      }
    }
  }

  return mlir::success();
}

namespace
{
  /// Model solving pass.
  /// Its objective is to convert a descriptive (and thus not sequential) model into an
  /// algorithmic one and to create the functions to be called during the simulation.
  class ModelSolvingPass: public mlir::PassWrapper<ModelSolvingPass, mlir::OperationPass<mlir::ModuleOp>>
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
        llvm::SmallVector<ModelOp, 1> models;

        getOperation().walk([&](ModelOp op) {
          models.push_back(op);
        });

        if (models.size() > 1) {
          // There must be at most one ModelOp inside the module
          return signalPassFailure();
        }

        mlir::OpBuilder builder(models[0]);

        // Store the list of the original variable names, as it will be needed when
        // printing the values but new additional members will have been created by that time.
        auto variableNames = models[0].variableNames();

        // Copy the equations into the initial equations' region, in order to use
        // them when computing the initial values of the variables.

        if (mlir::failed(copyEquationsAmongInitialEquations(builder, models[0]))) {
          return signalPassFailure();
        }

        if (mlir::failed(convertEquationsWithMultipleValues())) {
          return signalPassFailure();
        }

        // Split the loops containing more than one operation within their bodies
        if (mlir::failed(convertToSingleEquationBody(models[0]))) {
          return signalPassFailure();
        }

        // The initial conditions are determined by resolving a separate model, with
        // indeed more equations than the model used during the simulation loop.
        Model<Equation> initialModel(models[0]);
        Model<Equation> model(models[0]);

        // Set the names of the original variables, so that we can modify the 'init'
        // without losing information about the original names.
        initialModel.setVariableNames(variableNames);
        model.setVariableNames(variableNames);

        initialModel.setVariables(discoverVariables(initialModel.getOperation().initialEquationsRegion()));
        initialModel.setEquations(discoverEquations(initialModel.getOperation().initialEquationsRegion(), initialModel.getVariables()));

        model.setVariables(discoverVariables(model.getOperation().equationsRegion()));
        model.setEquations(discoverEquations(model.getOperation().equationsRegion(), model.getVariables()));

        // Split variables according to their type.
        // Determine which scalar variables do appear as argument to the derivative operation.
        std::map<unsigned int, IndexSet> derivedIndices;
        collectDerivedVariablesIndices(derivedIndices, initialModel.getEquations());
        collectDerivedVariablesIndices(derivedIndices, model.getEquations());

        VariablesMap variablesMap;

        if (mlir::failed(splitVariables(builder, models[0], variablesMap, derivedIndices))) {
          return signalPassFailure();
        }

        initialModel.setVariablesMap(variablesMap);
        model.setVariablesMap(variablesMap);

        // The variable splitting may have caused a variable list change and equations splitting.
        // For this reason we need to perform again the discovery process.

        initialModel.setVariables(discoverVariables(initialModel.getOperation().initialEquationsRegion()));
        initialModel.setEquations(discoverEquations(initialModel.getOperation().initialEquationsRegion(), initialModel.getVariables()));

        model.setVariables(discoverVariables(model.getOperation().equationsRegion()));
        model.setEquations(discoverEquations(model.getOperation().equationsRegion(), model.getVariables()));

        // Create the variables for the derivatives
        std::set<unsigned int> derivedVariables;
        collectDerivedVariables(derivedVariables, initialModel.getEquations());
        collectDerivedVariables(derivedVariables, model.getEquations());

        DerivativesMap derivativesMap;

        if (mlir::failed(createDerivativeVariables(builder, models[0], derivativesMap, derivedVariables))) {
          return signalPassFailure();
        }

        // The derivatives mapping is now complete, thus we can set the derivatives map inside the models
        initialModel.setDerivativesMap(derivativesMap);
        model.setDerivativesMap(derivativesMap);

        // Now that the derivatives have been converted to variables, we need perform a new scan
        // of the variables so that they become available inside the model.
        initialModel.setVariables(discoverVariables(initialModel.getOperation().initialEquationsRegion()));
        model.setVariables(discoverVariables(model.getOperation().equationsRegion()));

        // Remove the derivative operations
        if (mlir::failed(removeDerOps(builder, initialModel))) {
          return signalPassFailure();
        }

        if (mlir::failed(removeDerOps(builder, model))) {
          return signalPassFailure();
        }

        // Erasing the derivative operations may have caused equations splitting.
        // For this reason we need to perform again the discovery process.

        initialModel.setEquations(discoverEquations(initialModel.getOperation().initialEquationsRegion(), initialModel.getVariables()));
        model.setEquations(discoverEquations(model.getOperation().equationsRegion(), model.getVariables()));

        // Convert both the models to scheduled ones
        Model<ScheduledEquationsBlock> scheduledInitialModel(initialModel.getOperation());
        Model<ScheduledEquationsBlock> scheduledModel(model.getOperation());

        /*
        auto initialModelVariableMatchableFn = [](const Variable& variable) -> bool {
          return !variable.isConstant();
        };

        if (mlir::failed(convertToScheduledModel(builder, scheduledInitialModel, initialModel, initialModelVariableMatchableFn))) {
          scheduledInitialModel.getOperation().emitError("Can't solve the initialization problem");
          return signalPassFailure();
        }
         */

        auto modelVariableMatchableFn = [&](const Variable& variable) -> bool {
          mlir::Value var = variable.getValue();
          auto argNumber = var.cast<mlir::BlockArgument>().getArgNumber();

          return !model.getDerivativesMap().hasDerivative(argNumber) && !variable.isConstant();
        };

        if (mlir::failed(convertToScheduledModel(builder, scheduledModel, model, modelVariableMatchableFn))) {
          scheduledInitialModel.getOperation().emitError("Can't solve the model");
          return signalPassFailure();
        }

        // Create the simulation functions
        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        marco::codegen::TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);
        ModelConverter modelConverter(options, typeConverter);

        if (auto status = modelConverter.convert(builder, scheduledModel); mlir::failed(status)) {
          return signalPassFailure();
        }

        // Erase the model operation, which has been converted to algorithmic code
        models[0].erase();
      }

    private:
      /// Copy the equations declared into the 'equations' region into the 'initial equations' region.
      mlir::LogicalResult copyEquationsAmongInitialEquations(mlir::OpBuilder& builder, ModelOp modelOp)
      {
        if (!modelOp.hasEquationsBlock()) {
          // There is no equation to be copied
          return mlir::success();
        }

        mlir::OpBuilder::InsertionGuard guard(builder);

        if (!modelOp.hasInitialEquationsBlock()) {
          builder.createBlock(&modelOp.initialEquationsRegion(), {}, modelOp.equationsRegion().getArgumentTypes());
        }

        // Map the variables declared into the equations region to the ones declared into the initial equations' region
        mlir::BlockAndValueMapping mapping;
        auto originalVariables = modelOp.equationsRegion().getArguments();
        auto mappedVariables = modelOp.initialEquationsRegion().getArguments();
        assert(originalVariables.size() == mappedVariables.size());

        for (const auto& [original, mapped] : llvm::zip(originalVariables, mappedVariables)) {
          mapping.map(original, mapped);
        }

        // Clone the equations
        builder.setInsertionPointToEnd(modelOp.initialEquationsBlock());

        for (auto& op : modelOp.equationsBlock()->getOperations()) {
          builder.clone(op, mapping);
        }

        return mlir::success();
      }

      mlir::LogicalResult convertEquationsWithMultipleValues()
      {
        mlir::ConversionTarget target(getContext());

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        target.addDynamicallyLegalOp<EquationOp>([](EquationOp op) {
          auto terminator = mlir::cast<EquationSidesOp>(op.bodyBlock()->getTerminator());
          return terminator.lhsValues().size() == 1 && terminator.rhsValues().size() == 1;
        });

        mlir::OwningRewritePatternList patterns(&getContext());
        patterns.insert<EquationOpMultipleValuesPattern>(&getContext());

        return applyPartialConversion(getOperation(), target, std::move(patterns));
      }

      mlir::LogicalResult convertToSingleEquationBody(ModelOp modelOp)
      {
        llvm::SmallVector<EquationOp> equations;

        for (auto op : modelOp.equationsBlock()->getOps<EquationOp>()) {
          equations.push_back(op);
        }

        mlir::OpBuilder builder(modelOp);

        mlir::BlockAndValueMapping mapping;

        for (auto& equationOp : equations) {
          builder.setInsertionPointToEnd(modelOp.equationsBlock());
          std::vector<ForEquationOp> parents;

          ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();

          while (parent != nullptr) {
            parents.push_back(parent);
            parent = parent->getParentOfType<ForEquationOp>();
          }

          for (size_t i = 0, e = parents.size(); i < e; ++i) {
            auto clonedParent = mlir::cast<ForEquationOp>(builder.clone(*parents[e - i - 1].getOperation(), mapping));
            builder.setInsertionPointToEnd(clonedParent.bodyBlock());
          }

          builder.clone(*equationOp.getOperation(), mapping);
        }

        for (auto& equationOp : equations) {
          ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();
          equationOp.erase();

          while (parent != nullptr && parent.bodyBlock()->empty()) {
            ForEquationOp newParent = parent->getParentOfType<ForEquationOp>();
            parent.erase();
            parent = newParent;
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult splitEquations(mlir::OpBuilder& builder, Model<MatchedEquation>& model)
      {
        Equations<MatchedEquation> equations;

        for (const auto& equation : model.getEquations()) {
          auto write = equation->getWrite();
          auto iterationRanges = equation->getIterationRanges();
          auto writtenIndices = write.getAccessFunction().map(iterationRanges);

          IndexSet result;

          for (const auto& access : equation->getAccesses()) {
            if (access.getPath() == write.getPath()) {
              continue;
            }

            if (access.getVariable() != write.getVariable()) {
              continue;
            }

            auto accessedIndices = access.getAccessFunction().map(iterationRanges);

            if (!accessedIndices.overlaps(writtenIndices)) {
              continue;
            }

            result += write.getAccessFunction().inverseMap(
                IndexSet(accessedIndices.intersect(writtenIndices)),
                IndexSet(iterationRanges));
          }

          for (const auto& range : result) {
            auto clone = Equation::build(equation->getOperation(), equation->getVariables());

            auto matchedClone = std::make_unique<MatchedEquation>(
                std::move(clone), range, write.getPath());

            equations.add(std::move(matchedClone));
          }

          for (const auto& range : IndexSet(iterationRanges) - result) {
            auto clone = Equation::build(equation->getOperation(), equation->getVariables());

            auto matchedClone = std::make_unique<MatchedEquation>(
                std::move(clone), range, write.getPath());

            equations.add(std::move(matchedClone));
          }
        }

        model.setEquations(equations);
        return mlir::success();
      }

      mlir::LogicalResult convertToScheduledModel(
          mlir::OpBuilder& builder,
          Model<ScheduledEquationsBlock>& result,
          const Model<Equation>& model,
          std::function<bool(const Variable&)> isMatchableFn)
      {
        // Matching process
        Model<MatchedEquation> matchedModel(model.getOperation());
        matchedModel.setVariableNames(model.getVariableNames());
        matchedModel.setVariablesMap(model.getVariablesMap());
        matchedModel.setDerivativesMap(model.getDerivativesMap());

        if (auto res = match(matchedModel, model, isMatchableFn); mlir::failed(res)) {
          return res;
        }

        if (auto res = splitEquations(builder, matchedModel); mlir::failed(res)) {
          return res;
        }

        // Resolve the algebraic loops
        if (auto res = solveCycles(matchedModel, builder); mlir::failed(res)) {
          if (options.solver != Solver::ida) {
            // Check if the selected solver can deal with cycles. If not, fail.
            return res;
          }
        }

        // Schedule the equations
        if (auto res = schedule(result, matchedModel); mlir::failed(res)) {
          return res;
        }

        result.setVariableNames(model.getVariableNames());
        result.setVariablesMap(model.getVariablesMap());
        result.setDerivativesMap(model.getDerivativesMap());

        return mlir::success();
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
