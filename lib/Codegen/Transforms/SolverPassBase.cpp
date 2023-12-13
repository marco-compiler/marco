#include "marco/Codegen/Transforms/SolverPassBase.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace ::marco;
using namespace ::mlir::modelica;

static IndexSet getPrintableIndices(
    VariableType variableType,
    llvm::ArrayRef<VariableFilter::Filter> filters)
{
  assert(!variableType.isScalar());
  IndexSet result;

  for (const auto& filter : filters) {
    if (!filter.isVisible()) {
      continue;
    }

    auto filterRanges = filter.getRanges();
    llvm::SmallVector<Range, 3> ranges;

    assert(variableType.hasStaticShape());

    assert(static_cast<int64_t>(filterRanges.size()) ==
           variableType.getRank());

    for (const auto& range : llvm::enumerate(filterRanges)) {
      // In Modelica, arrays are 1-based. If present, we need to lower by 1 the
      // value given by the variable filter.

      auto lowerBound = range.value().hasLowerBound()
          ? range.value().getLowerBound() - 1 : 0;

      auto upperBound = range.value().hasUpperBound()
          ? range.value().getUpperBound()
          : variableType.getShape()[range.index()];

      ranges.emplace_back(lowerBound, upperBound);
    }

    result += MultidimensionalRange(std::move(ranges));
  }

  return std::move(result);
}

static void createIterationLoops(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    llvm::ArrayRef<mlir::Value> beginIndices,
    llvm::ArrayRef<mlir::Value> endIndices,
    llvm::ArrayRef<mlir::Value> steps,
    mlir::ArrayAttr iterationDirections,
    llvm::SmallVectorImpl<mlir::Value>& inductions)
{
  assert(beginIndices.size() == endIndices.size());
  assert(beginIndices.size() == steps.size());

  assert(llvm::all_of(
      iterationDirections.getAsRange<EquationScheduleDirectionAttr>(),
      [](EquationScheduleDirectionAttr direction) {
        return direction.getValue() == EquationScheduleDirection::Forward ||
            direction.getValue() == EquationScheduleDirection::Backward;
      }));

  auto conditionFn =
      [&](EquationScheduleDirectionAttr direction,
          mlir::Value index, mlir::Value end) -> mlir::Value {
    if (direction.getValue() == EquationScheduleDirection::Backward) {
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sgt, index, end);
    }

    return builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, index, end);
  };

  auto updateFn =
      [&](EquationScheduleDirectionAttr direction,
          mlir::Value index, mlir::Value step) -> mlir::Value {
    if (direction.getValue() == EquationScheduleDirection::Backward) {
      return builder.create<mlir::arith::SubIOp>(loc, index, step);
    }

    return builder.create<mlir::arith::AddIOp>(loc, index, step);
  };

  for (size_t i = 0; i < steps.size(); ++i) {
    auto whileOp = builder.create<mlir::scf::WhileOp>(
        loc, builder.getIndexType(), beginIndices[i]);

    // Check the condition.
    // A naive check can consist in the equality comparison. However, in
    // order to be future-proof with respect to steps greater than one, we
    // need to check if the current value is beyond the end boundary. This in
    // turn requires to know the iteration direction.

    mlir::Block* beforeBlock = builder.createBlock(
        &whileOp.getBefore(), {}, builder.getIndexType(), loc);

    builder.setInsertionPointToStart(beforeBlock);

    mlir::Value condition = conditionFn(
        iterationDirections[i].cast<EquationScheduleDirectionAttr>(),
        whileOp.getBefore().getArgument(0), endIndices[i]);

    builder.create<mlir::scf::ConditionOp>(
        loc, condition, whileOp.getBefore().getArgument(0));

    // Execute the loop body.
    mlir::Block* afterBlock = builder.createBlock(
        &whileOp.getAfter(), {}, builder.getIndexType(), loc);

    mlir::Value inductionVariable = afterBlock->getArgument(0);
    inductions.push_back(inductionVariable);
    builder.setInsertionPointToStart(afterBlock);

    // Update the induction variable.
    mlir::Value nextValue = updateFn(
        iterationDirections[i].cast<EquationScheduleDirectionAttr>(),
        inductionVariable, steps[i]);

    builder.create<mlir::scf::YieldOp>(loc, nextValue);
    builder.setInsertionPoint(nextValue.getDefiningOp());
  }
}

namespace
{
  class TimeOpLowering : public mlir::OpRewritePattern<TimeOp>
  {
    public:
      using mlir::OpRewritePattern<TimeOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          TimeOp op, mlir::PatternRewriter& rewriter) const override
      {
        auto timeType = RealType::get(rewriter.getContext());

        auto globalGetOp = rewriter.create<GlobalVariableGetOp>(
            op.getLoc(),
            ArrayType::get(std::nullopt, timeType),
            "time");

        rewriter.replaceOpWithNewOp<LoadOp>(
            op, timeType, globalGetOp, std::nullopt);

        return mlir::success();
      }
  };
}

namespace
{
  class EquationTemplateEquationSidesOpPattern
      : public mlir::OpRewritePattern<EquationSidesOp>
  {
    public:
      using mlir::OpRewritePattern<EquationSidesOp>::OpRewritePattern;

      EquationTemplateEquationSidesOpPattern(
          mlir::MLIRContext* context,
          uint64_t viewElementIndex,
          uint64_t numOfExplicitInductions,
          uint64_t numOfImplicitInductions,
          mlir::ArrayAttr iterationDirections,
          mlir::ValueRange functionArgs)
          : mlir::OpRewritePattern<EquationSidesOp>(context),
            viewElementIndex(viewElementIndex),
            numOfExplicitInductions(numOfExplicitInductions),
            numOfImplicitInductions(numOfImplicitInductions),
            iterationDirections(iterationDirections)
      {
        for (mlir::Value functionArg : functionArgs) {
          this->functionArgs.push_back(functionArg);
        }
      }

      mlir::LogicalResult matchAndRewrite(
          EquationSidesOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        mlir::Value lhsValue = op.getLhsValues()[viewElementIndex];
        mlir::Value rhsValue = op.getRhsValues()[viewElementIndex];

        if (mlir::failed(convertToAssignment(
                rewriter, loc, lhsValue, rhsValue))) {
          return mlir::failure();
        }

        // Erase the equation terminator and also the operations for its sides.
        auto lhsOp = op.getLhs().getDefiningOp<EquationSideOp>();
        auto rhsOp = op.getRhs().getDefiningOp<EquationSideOp>();
        rewriter.eraseOp(op);
        rewriter.eraseOp(lhsOp);
        rewriter.eraseOp(rhsOp);

        return mlir::success();
      }

    private:
      /// Convert the equality to an assignment.
      mlir::LogicalResult convertToAssignment(
          mlir::OpBuilder& builder, mlir::Location loc,
          mlir::Value lhsValue, mlir::Value rhsValue) const
      {
        if (auto arrayType = lhsValue.getType().dyn_cast<ArrayType>()) {
          // Vectorized assignment.
          // We need to turn the implicit inductions into explicit ones.
          assert(arrayType.getRank() ==
                 static_cast<int64_t>(numOfImplicitInductions));

          assert(rhsValue.getType().isa<ArrayType>());
          assert(rhsValue.getType().cast<ArrayType>().getRank() ==
                 static_cast<int64_t>(numOfImplicitInductions));

          llvm::SmallVector<mlir::Value, 3> implicitInductions;

          llvm::SmallVector<mlir::Value, 3> implicitIterationsBegin;
          llvm::SmallVector<mlir::Value, 3> implicitIterationsEnd;
          llvm::SmallVector<mlir::Value, 3> implicitIterationsStep;

          for (size_t i = 0; i < numOfImplicitInductions; ++i) {
            implicitIterationsBegin.push_back(
                functionArgs[numOfExplicitInductions * 3 + i * 3]);

            implicitIterationsEnd.push_back(
                functionArgs[numOfExplicitInductions * 3 + i * 3 + 1]);

            implicitIterationsStep.push_back(
                functionArgs[numOfExplicitInductions * 3 + i * 3 + 2]);
          }

          ::createIterationLoops(
              builder, loc,
              implicitIterationsBegin,
              implicitIterationsEnd,
              implicitIterationsStep,
              iterationDirections, implicitInductions);

          rhsValue = builder.create<LoadOp>(loc, rhsValue, implicitInductions);

          rhsValue = builder.create<CastOp>(
              loc, arrayType.getElementType(), rhsValue);

          builder.create<StoreOp>(loc, rhsValue, lhsValue, implicitInductions);
          return mlir::success();
        }

        if (auto loadOp = lhsValue.getDefiningOp<LoadOp>()) {
          // Left-hand side is a scalar element extract from an array variable.
          rhsValue = builder.create<CastOp>(loc, lhsValue.getType(), rhsValue);

          builder.create<StoreOp>(
              loc, rhsValue, loadOp.getArray(), loadOp.getIndices());

          return mlir::success();
        }

        if (auto variableGetOp = lhsValue.getDefiningOp<VariableGetOp>()) {
          // Left-hand side is a scalar variable.
          rhsValue = builder.create<CastOp>(loc, lhsValue.getType(), rhsValue);

          builder.create<VariableSetOp>(
              loc, variableGetOp.getVariable(), rhsValue);

          return mlir::success();
        }

        return mlir::failure();
      }

    private:
      uint64_t viewElementIndex;
      uint64_t numOfExplicitInductions;
      uint64_t numOfImplicitInductions;
      mlir::ArrayAttr iterationDirections;
      llvm::SmallVector<mlir::Value, 10> functionArgs;
  };

  class EquationTemplateVariableGetOpPattern
      : public mlir::OpRewritePattern<VariableGetOp>
  {
    public:
      EquationTemplateVariableGetOpPattern(
          mlir::MLIRContext* context,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap)
          : mlir::OpRewritePattern<VariableGetOp>(context),
            localToGlobalVariablesMap(&localToGlobalVariablesMap)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          VariableGetOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto globalVariableIt =
            localToGlobalVariablesMap->find(op.getVariable());

        if (globalVariableIt != localToGlobalVariablesMap->end()) {
          mlir::Value replacement = rewriter.create<GlobalVariableGetOp>(
              loc, globalVariableIt->getValue());

          if (auto arrayType = replacement.getType().dyn_cast<ArrayType>();
              arrayType && arrayType.isScalar()) {
            replacement = rewriter.create<LoadOp>(
                loc, replacement, std::nullopt);
          }

          rewriter.replaceOp(op, replacement);
          return mlir::success();
        }

        return mlir::failure();
      }

    private:
      const llvm::StringMap<GlobalVariableOp>* localToGlobalVariablesMap;
  };

  class EquationTemplateVariableSetOpPattern
      : public mlir::OpRewritePattern<VariableSetOp>
  {
    public:
      EquationTemplateVariableSetOpPattern(
          mlir::MLIRContext* context,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap)
          : mlir::OpRewritePattern<VariableSetOp>(context),
            localToGlobalVariablesMap(&localToGlobalVariablesMap)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          VariableSetOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto globalVariableIt =
            localToGlobalVariablesMap->find(op.getVariable());

        if (globalVariableIt == localToGlobalVariablesMap->end()) {
          return mlir::failure();
        }

        GlobalVariableOp globalVariableOp = globalVariableIt->getValue();

        mlir::Value globalVariable =
            rewriter.create<GlobalVariableGetOp>(loc, globalVariableOp);

        mlir::Value storedValue = op.getValue();
        auto arrayType = globalVariable.getType().cast<ArrayType>();

        if (!arrayType.isScalar()) {
          return mlir::failure();
        }

        if (mlir::Type expectedType = arrayType.getElementType();
            storedValue.getType() != expectedType) {
          storedValue = rewriter.create<CastOp>(
              loc, expectedType, storedValue);
        }

        rewriter.replaceOpWithNewOp<StoreOp>(
            op, storedValue, globalVariable, std::nullopt);

        return mlir::success();
      }

    private:
      const llvm::StringMap<GlobalVariableOp>* localToGlobalVariablesMap;
  };
}

namespace mlir::modelica::impl
{
  ModelSolver::ModelSolver() = default;

  ModelSolver::~ModelSolver() = default;

  mlir::LogicalResult ModelSolver::convert(
      ModelOp modelOp,
      const DerivativesMap& derivativesMap,
      const VariableFilter& variablesFilter,
      bool processICModel,
      bool processMainModel)
  {
    mlir::Location loc = modelOp.getLoc();
    mlir::SymbolTableCollection symbolTableCollection;

    mlir::IRRewriter rewriter(modelOp.getContext());
    auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();

    // Collect the variables once for faster processing.
    llvm::SmallVector<VariableOp> variableOps;

    for (VariableOp variableOp : modelOp.getVariables()) {
      variableOps.push_back(variableOp);
    }

    // Collect the SCCs once for faster processing.
    llvm::SmallVector<SCCOp> initialSCCs;
    llvm::SmallVector<SCCOp> SCCs;
    modelOp.collectSCCs(initialSCCs, SCCs);

    // Create the common functions.
    if (mlir::failed(createModelNameOp(
            rewriter, moduleOp, loc, modelOp))) {
      return mlir::failure();
    }

    if (mlir::failed(createNumOfVariablesOp(
            rewriter, moduleOp, loc, variableOps))) {
      return mlir::failure();
    }

    if (mlir::failed(createVariableNamesOp(
            rewriter, moduleOp, loc, variableOps))) {
      return mlir::failure();
    }

    if (mlir::failed(createVariableRanksOp(
            rewriter, moduleOp, loc, variableOps))) {
      return mlir::failure();
    }

    if (mlir::failed(createPrintableIndicesOp(
            rewriter, moduleOp, loc,
            variableOps, derivativesMap, variablesFilter))) {
      return mlir::failure();
    }

    if (mlir::failed(createDerivativesMapOp(
            rewriter, moduleOp, loc, variableOps, derivativesMap))) {
      return mlir::failure();
    }

    llvm::StringMap<GlobalVariableOp> localToGlobalVariablesMap;

    if (mlir::failed(createGlobalVariables(
            rewriter, moduleOp, symbolTableCollection,
            modelOp, variableOps,
            localToGlobalVariablesMap))) {
      return mlir::failure();
    }

    if (mlir::failed(createVariableGetters(
            rewriter, moduleOp, loc, variableOps, localToGlobalVariablesMap))) {
      return mlir::failure();
    }

    if (mlir::failed(createInitFunction(
            rewriter, moduleOp, modelOp, variableOps,
            localToGlobalVariablesMap))) {
      return mlir::failure();
    }

    if (mlir::failed(createDeinitFunction(rewriter, moduleOp, loc))) {
      return mlir::failure();
    }

    if (processICModel) {
      if (mlir::failed(solveICModel(
              rewriter, symbolTableCollection, modelOp, variableOps,
              localToGlobalVariablesMap, initialSCCs))) {
        return mlir::failure();
      }
    }

    // Process the 'main' model.
    if (processMainModel) {
      if (mlir::failed(solveMainModel(
              rewriter, symbolTableCollection, modelOp,
              variableOps, derivativesMap, localToGlobalVariablesMap,
              SCCs))) {
        return mlir::failure();
      }
    }

    // The model has been completely converted to the code composing the
    // simulation, thus it can now be erased.
    rewriter.eraseOp(modelOp);

    // Declare the time variable.
    GlobalVariableOp timeVariableOp =
        declareTimeVariable(rewriter, moduleOp, loc, symbolTableCollection);

    if (!timeVariableOp) {
      return mlir::failure();
    }

    // Declare the time getter.
    if (mlir::failed(createTimeGetterOp(
            rewriter, moduleOp, symbolTableCollection, timeVariableOp))) {
      return mlir::failure();
    }

    // Declare the time setter.
    if (mlir::failed(createTimeSetterOp(
            rewriter, moduleOp, symbolTableCollection, timeVariableOp))) {
      return mlir::failure();
    }

    // Convert the time operation.
    if (mlir::failed(convertTimeOp(moduleOp))) {
      return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createModelNameOp(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      ModelOp modelOp)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());
    builder.create<mlir::simulation::ModelNameOp>(loc, modelOp.getSymName());
    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createNumOfVariablesOp(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      llvm::ArrayRef<VariableOp> variableOps)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    builder.create<mlir::simulation::NumberOfVariablesOp>(
        loc, builder.getI64IntegerAttr(variableOps.size()));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createVariableNamesOp(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      llvm::ArrayRef<VariableOp> variableOps)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    llvm::SmallVector<llvm::StringRef> names;

    for (VariableOp variableOp : variableOps) {
      names.push_back(variableOp.getSymName());
    }

    builder.create<mlir::simulation::VariableNamesOp>(
        loc, builder.getStrArrayAttr(names));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createVariableRanksOp(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      llvm::ArrayRef<VariableOp> variableOps)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    llvm::SmallVector<int64_t> ranks;

    for (VariableOp variableOp : variableOps) {
      VariableType variableType = variableOp.getVariableType();
      ranks.push_back(variableType.getRank());
    }

    builder.create<mlir::simulation::VariableRanksOp>(
        loc, builder.getI64ArrayAttr(ranks));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createPrintableIndicesOp(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      llvm::ArrayRef<VariableOp> variableOps,
      const DerivativesMap& derivativesMap,
      const VariableFilter& variablesFilter)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    llvm::SmallVector<mlir::Attribute> printInformation;

    auto getFlatName = [](mlir::SymbolRefAttr symbolRef) -> std::string {
      std::string result = symbolRef.getRootReference().str();

      for (mlir::FlatSymbolRefAttr nested : symbolRef.getNestedReferences()) {
        result += "." + nested.getValue().str();
      }

      return result;
    };

    for (VariableOp variableOp : variableOps) {
      VariableType variableType = variableOp.getVariableType();
      std::vector<VariableFilter::Filter> filters;

      if (auto stateName = derivativesMap.getDerivedVariable(
              mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr()))) {
        // Derivative variable.
        filters = variablesFilter.getVariableDerInfo(
            getFlatName(*stateName), variableType.getRank());

      } else {
        // Non-derivative variable.
        filters = variablesFilter.getVariableInfo(
            variableOp.getSymName(), variableType.getRank());
      }

      if (variableType.isScalar()) {
        // Scalar variable.
        bool isVisible = llvm::any_of(
            filters, [](const VariableFilter::Filter& filter) {
              return filter.isVisible();
            });

        printInformation.push_back(builder.getBoolAttr(isVisible));
      } else {
        // Array variable.
        IndexSet printableIndices = getPrintableIndices(variableType, filters);
        printableIndices = printableIndices.getCanonicalRepresentation();

        printInformation.push_back(IndexSetAttr::get(
            builder.getContext(), printableIndices));
      }
    }

    builder.create<mlir::simulation::PrintableIndicesOp>(
        loc, builder.getArrayAttr(printInformation));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createDerivativesMapOp(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      llvm::ArrayRef<VariableOp> variableOps,
      const DerivativesMap& derivativesMap)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    // Map the position of the variables for faster lookups.
    llvm::DenseMap<mlir::SymbolRefAttr, int64_t> positionsMap;

    for (size_t i = 0, e = variableOps.size(); i < e; ++i) {
      VariableOp variableOp = variableOps[i];
      auto name = mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr());
      positionsMap[name] = i;
    }

    // Compute the positions of the derivatives.
    llvm::SmallVector<int64_t> derivatives;

    for (VariableOp variableOp : variableOps) {
      if (auto derivative = derivativesMap.getDerivative(
              mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr()))) {
        auto it = positionsMap.find(*derivative);

        if (it == positionsMap.end()) {
          return mlir::failure();
        }

        derivatives.push_back(it->getSecond());
      } else {
        derivatives.push_back(-1);
      }
    }

    builder.create<mlir::simulation::DerivativesMapOp>(
        loc, builder.getI64ArrayAttr(derivatives));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createGlobalVariables(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection& symbolTableCollection,
      ModelOp modelOp,
      llvm::ArrayRef<VariableOp> variableOps,
      llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    // Declare the global variables.
    for (size_t i = 0, e = variableOps.size(); i < e; ++i) {
      VariableOp variableOp = variableOps[i];

      auto globalVariableOp = builder.create<GlobalVariableOp>(
          variableOp.getLoc(), "var_" + std::to_string(i),
          variableOp.getVariableType().toArrayType());

      symbolTableCollection.getSymbolTable(moduleOp).insert(globalVariableOp);
      localToGlobalVariablesMap[variableOp.getSymName()] = globalVariableOp;
    }

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createVariableGetters(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      llvm::ArrayRef<VariableOp> variableOps,
      const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Create a getter for each variable.
    size_t variableGetterCounter = 0;
    llvm::SmallVector<mlir::Attribute> getterNames;

    for (VariableOp variableOp : variableOps) {
      builder.setInsertionPointToEnd(moduleOp.getBody());
      VariableType variableType = variableOp.getVariableType();

      auto getterOp = builder.create<mlir::simulation::VariableGetterOp>(
          variableOp.getLoc(),
          "var_getter_" + std::to_string(variableGetterCounter++),
          variableType.getRank());

      getterNames.push_back(
          mlir::FlatSymbolRefAttr::get(getterOp.getSymNameAttr()));

      mlir::Block* bodyBlock = getterOp.addEntryBlock();
      builder.setInsertionPointToStart(bodyBlock);

      mlir::Value variable = builder.create<GlobalVariableGetOp>(
          variableOp.getLoc(),
          localToGlobalVariablesMap.lookup(variableOp.getSymName()));

      mlir::Value result = builder.create<LoadOp>(
          variableOp.getLoc(), variable, getterOp.getIndices());

      result = builder.create<CastOp>(
          variableOp.getLoc(), getterOp.getResultTypes()[0], result);

      builder.create<mlir::simulation::ReturnOp>(variableOp.getLoc(), result);
    }

    // Create the operation collecting all the getters.
    builder.setInsertionPointToEnd(moduleOp.getBody());

    builder.create<mlir::simulation::VariableGettersOp>(
        loc, builder.getArrayAttr(getterNames));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createInitFunction(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      ModelOp modelOp,
      llvm::ArrayRef<VariableOp> variableOps,
      const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto initFunctionOp =
        builder.create<mlir::simulation::InitFunctionOp>(modelOp.getLoc());

    mlir::Block* entryBlock =
        builder.createBlock(&initFunctionOp.getBodyRegion());

    builder.setInsertionPointToStart(entryBlock);

    // Map the variables by name for faster lookup.
    llvm::StringMap<VariableOp> variablesMap;

    for (VariableOp variableOp : variableOps) {
      variablesMap[variableOp.getSymName()] = variableOp;
    }

    // Keep track of the variables for which a start value has been provided.
    llvm::DenseSet<llvm::StringRef> initializedVars;

    for (StartOp startOp : modelOp.getOps<StartOp>()) {
      // Set the variable as initialized.
      initializedVars.insert(startOp.getVariable());

      // Note that read-only variables must be set independently of the 'fixed'
      // attribute being true or false.

      VariableOp variableOp = variablesMap[startOp.getVariable()];

      if (startOp.getFixed() && !variableOp.getVariableType().isReadOnly()) {
        continue;
      }

      mlir::IRMapping startOpsMapping;

      for (auto& op : startOp.getBodyRegion().getOps()) {
        if (auto yieldOp = mlir::dyn_cast<YieldOp>(op)) {
          mlir::Value valueToBeStored =
              startOpsMapping.lookup(yieldOp.getValues()[0]);

          mlir::Value destination = builder.create<GlobalVariableGetOp>(
              startOp.getLoc(),
              localToGlobalVariablesMap.lookup(startOp.getVariable()));

          if (startOp.getEach()) {
            builder.create<ArrayFillOp>(
                startOp.getLoc(), destination, valueToBeStored);
          } else {
            auto destinationArrayType =
                destination.getType().cast<ArrayType>();

            auto valueType = valueToBeStored.getType();

            if (auto valueArrayType = valueType.dyn_cast<ArrayType>()) {
              builder.create<ArrayCopyOp>(
                  startOp.getLoc(), valueToBeStored, destination);
            } else {
              auto elementType = destinationArrayType.getElementType();

              if (elementType != valueToBeStored.getType()) {
                valueToBeStored = builder.create<CastOp>(
                    startOp.getLoc(), elementType, valueToBeStored);
              }

              builder.create<StoreOp>(
                  startOp.getLoc(), valueToBeStored, destination,
                  std::nullopt);
            }
          }
        } else {
          builder.clone(op, startOpsMapping);
        }
      }
    }

    // The variables without a 'start' attribute must be initialized to zero.
    for (VariableOp variableOp : modelOp.getOps<VariableOp>()) {
      if (initializedVars.contains(variableOp.getSymName())) {
        continue;
      }

      mlir::Value destination = builder.create<GlobalVariableGetOp>(
          variableOp.getLoc(),
          localToGlobalVariablesMap.lookup(variableOp.getSymName()));

      auto arrayType = destination.getType().cast<ArrayType>();

      mlir::Value zero = builder.create<ConstantOp>(
          destination.getLoc(), getZeroAttr(arrayType.getElementType()));

      if (arrayType.isScalar()) {
        builder.create<StoreOp>(
            destination.getLoc(), zero, destination, std::nullopt);
      } else {
        builder.create<ArrayFillOp>(destination.getLoc(), destination, zero);
      }
    }

    builder.setInsertionPointToEnd(&initFunctionOp.getBodyRegion().back());
    builder.create<mlir::simulation::YieldOp>(modelOp.getLoc(), std::nullopt);

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createDeinitFunction(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto deinitFunctionOp =
        builder.create<mlir::simulation::DeinitFunctionOp>(loc);

    mlir::Block* entryBlock =
        builder.createBlock(&deinitFunctionOp.getBodyRegion());

    builder.setInsertionPointToStart(entryBlock);

    builder.create<mlir::simulation::YieldOp>(loc, std::nullopt);

    return mlir::success();
  }

  GlobalVariableOp ModelSolver::declareTimeVariable(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto timeType = RealType::get(builder.getContext());

    auto globalVariableOp = builder.create<GlobalVariableOp>(
        loc, "time", ArrayType::get(std::nullopt, timeType));

    symbolTableCollection.getSymbolTable(moduleOp).insert(globalVariableOp);
    return globalVariableOp;
  }

  mlir::LogicalResult ModelSolver::createTimeGetterOp(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection& symbolTableCollection,
      GlobalVariableOp timeVariableOp)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    mlir::Location loc = timeVariableOp.getLoc();

    auto functionOp = builder.create<mlir::simulation::FunctionOp>(
        loc, "getTime",
        builder.getFunctionType(std::nullopt, builder.getF64Type()));

    symbolTableCollection.getSymbolTable(moduleOp).insert(functionOp);

    mlir::Block* entryBlock = functionOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value array =
        builder.create<GlobalVariableGetOp>(loc, timeVariableOp);

    mlir::Value result = builder.create<LoadOp>(
        loc, RealType::get(builder.getContext()), array);

    result = builder.create<CastOp>(loc, builder.getF64Type(), result);
    builder.create<mlir::simulation::ReturnOp>(loc, result);

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createTimeSetterOp(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection& symbolTableCollection,
      GlobalVariableOp timeVariableOp)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    mlir::Location loc = timeVariableOp.getLoc();

    auto functionOp = builder.create<mlir::simulation::FunctionOp>(
        loc, "setTime",
        builder.getFunctionType(builder.getF64Type(), std::nullopt));

    symbolTableCollection.getSymbolTable(moduleOp).insert(functionOp);

    mlir::Block* entryBlock = functionOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value array =
        builder.create<GlobalVariableGetOp>(loc, timeVariableOp);

    mlir::Value newTime = builder.create<CastOp>(
        loc, RealType::get(builder.getContext()), functionOp.getArgument(0));

    builder.create<StoreOp>(loc, newTime, array, std::nullopt);
    builder.create<mlir::simulation::ReturnOp>(loc, std::nullopt);

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::convertTimeOp(mlir::ModuleOp moduleOp)
  {
    mlir::RewritePatternSet patterns(moduleOp.getContext());

    patterns.add<
        TimeOpLowering>(moduleOp.getContext());

    return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
  }

  void ModelSolver::getEquationTemplatesUsageCount(
      llvm::ArrayRef<SCCOp> SCCs,
      llvm::DenseMap<EquationTemplateOp, size_t>& usages) const
  {
    for (SCCOp scc : SCCs) {
      for (ScheduledEquationInstanceOp equationOp :
           scc.getOps<ScheduledEquationInstanceOp>()) {
        usages[equationOp.getTemplate()]++;
      }
    }
  }

  RawFunctionOp ModelSolver::createEquationTemplateFunction(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection& symbolTableCollection,
      EquationTemplateOp equationTemplateOp,
      uint64_t viewElementIndex,
      mlir::ArrayAttr iterationDirections,
      llvm::StringRef functionName,
      const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    mlir::Location loc = equationTemplateOp.getLoc();

    size_t numOfExplicitInductions =
        equationTemplateOp.getInductionVariables().size();

    size_t numOfImplicitInductions =
        equationTemplateOp.getNumOfImplicitInductionVariables(
            viewElementIndex);

    llvm::SmallVector<mlir::Type, 9> argTypes(
        numOfExplicitInductions * 3 + numOfImplicitInductions * 3,
        builder.getIndexType());

    auto rawFunctionOp = builder.create<RawFunctionOp>(
        loc, functionName, builder.getFunctionType(argTypes, std::nullopt));

    symbolTableCollection.getSymbolTable(moduleOp).insert(rawFunctionOp);

    mlir::Block* entryBlock = rawFunctionOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::IRMapping mapping;

    // Create the explicit iteration loops.
    llvm::SmallVector<mlir::Value, 3> explicitInductions;

    llvm::SmallVector<mlir::Value, 3> explicitIterationsBegin;
    llvm::SmallVector<mlir::Value, 3> explicitIterationsEnd;
    llvm::SmallVector<mlir::Value, 3> explicitIterationsStep;

    for (size_t i = 0; i < numOfExplicitInductions; ++i) {
      explicitIterationsBegin.push_back(rawFunctionOp.getArgument(i * 3));
      explicitIterationsEnd.push_back(rawFunctionOp.getArgument(i * 3 + 1));
      explicitIterationsStep.push_back(rawFunctionOp.getArgument(i * 3 + 2));
    }

    ::createIterationLoops(
        builder, loc,
        explicitIterationsBegin, explicitIterationsEnd, explicitIterationsStep,
        iterationDirections, explicitInductions);

    // Map the induction variables.
    for (size_t i = 0; i < numOfExplicitInductions; ++i) {
      mapping.map(
          equationTemplateOp.getInductionVariables()[i],
          explicitInductions[i]);
    }

    // Clone the equation body.
    for (auto& op : equationTemplateOp.getOps()) {
      builder.clone(op, mapping);
    }

    mlir::ConversionTarget target(*builder.getContext());
    target.addLegalDialect<ModelicaDialect>();
    target.addIllegalOp<EquationSidesOp>();

    target.addDynamicallyLegalOp<VariableGetOp>([&](VariableGetOp op) {
      auto globalVariableIt = localToGlobalVariablesMap.find(op.getVariable());
      return globalVariableIt == localToGlobalVariablesMap.end();
    });

    target.addDynamicallyLegalOp<VariableSetOp>([&](VariableSetOp op) {
      auto globalVariableIt = localToGlobalVariablesMap.find(op.getVariable());
      return globalVariableIt == localToGlobalVariablesMap.end();
    });

    target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
      return true;
    });

    mlir::RewritePatternSet patterns(builder.getContext());

    patterns.insert<EquationTemplateEquationSidesOpPattern>(
        builder.getContext(), viewElementIndex,
        numOfExplicitInductions, numOfImplicitInductions,
        iterationDirections, rawFunctionOp.getArguments());

    patterns.insert<EquationTemplateVariableGetOpPattern>(
        builder.getContext(), localToGlobalVariablesMap);

    patterns.insert<EquationTemplateVariableSetOpPattern>(
        builder.getContext(), localToGlobalVariablesMap);

    if (mlir::failed(applyPartialConversion(
            rawFunctionOp, target, std::move(patterns)))) {
      return nullptr;
    }

    // Create the return operation.
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<RawReturnOp>(loc);

    return rawFunctionOp;
  }

  void ModelSolver::createIterationLoops(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      llvm::ArrayRef<mlir::Value> beginIndices,
      llvm::ArrayRef<mlir::Value> endIndices,
      llvm::ArrayRef<mlir::Value> steps,
      mlir::ArrayAttr iterationDirections,
      llvm::SmallVectorImpl<mlir::Value>& inductions)
  {
    ::createIterationLoops(builder, loc, beginIndices, endIndices, steps,
                           iterationDirections, inductions);
  }

  mlir::LogicalResult ModelSolver::callEquationFunction(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      ScheduledEquationInstanceOp equationOp,
      RawFunctionOp rawFunctionOp) const
  {
    llvm::SmallVector<mlir::Value, 3> args;

    // Explicit indices.
    if (auto indices = equationOp.getIndices()) {
      for (size_t i = 0, e = indices->getValue().rank(); i < e; ++i) {
        // Begin index.
        args.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(indices->getValue()[i].getBegin())));

        // End index.
        args.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(indices->getValue()[i].getEnd())));

        // Step.
        args.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(1)));
      }
    }

    // Implicit indices.
    if (auto indices = equationOp.getImplicitIndices()) {
      for (size_t i = 0, e = indices->getValue().rank(); i < e; ++i) {
        // Begin index.
        args.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(indices->getValue()[i].getBegin())));

        // End index.
        args.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(indices->getValue()[i].getEnd())));

        // Step.
        args.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(1)));
      }
    }

    builder.create<CallOp>(loc, rawFunctionOp, args);
    return mlir::success();
  }
}

//===---------------------------------------------------------------------===//
// ModelConversionTestPass

namespace mlir::modelica
{
#define GEN_PASS_DEF_MODELCONVERSIONTESTPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

namespace
{
  class TestSolver : public mlir::modelica::impl::ModelSolver
  {
    public:
      mlir::LogicalResult solveICModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::modelica::ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs) override
      {
        // Do nothing.
        return mlir::success();
      }

      mlir::LogicalResult solveMainModel(
          mlir::IRRewriter& rewriter,
          mlir::SymbolTableCollection& symbolTableCollection,
          mlir::modelica::ModelOp modelOp,
          llvm::ArrayRef<VariableOp> variableOps,
          const DerivativesMap& derivativesMap,
          const llvm::StringMap<GlobalVariableOp>& localToGlobalVariablesMap,
          llvm::ArrayRef<SCCOp> SCCs) override
      {
        // Do nothing.
        return mlir::success();
      }
  };

  class ModelConversionTestPass
      : public mlir::modelica::impl::ModelConversionTestPassBase<
          ModelConversionTestPass>
  {
    public:
      using ModelConversionTestPassBase::ModelConversionTestPassBase;

      void runOnOperation() override;

    private:
      DerivativesMap& getDerivativesMap(ModelOp modelOp);
  };
}

void ModelConversionTestPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ModelOp, 1> modelOps;

  for (ModelOp modelOp : moduleOp.getOps<ModelOp>()) {
      modelOps.push_back(modelOp);
  }

  TestSolver solver;

  auto expectedVariablesFilter =
      marco::VariableFilter::fromString(variablesFilter);

  std::unique_ptr<marco::VariableFilter> variablesFilterInstance;

  if (!expectedVariablesFilter) {
      getOperation().emitWarning(
          "Invalid variable filter string. No filtering will take place");

      variablesFilterInstance = std::make_unique<marco::VariableFilter>();
  } else {
      variablesFilterInstance = std::make_unique<marco::VariableFilter>(
          std::move(*expectedVariablesFilter));
  }

  for (ModelOp modelOp : modelOps) {
      DerivativesMap& derivativesMap = getDerivativesMap(modelOp);

      if (mlir::failed(solver.convert(
              modelOp, derivativesMap, *variablesFilterInstance,
              processICModel, processMainModel))) {
        return signalPassFailure();
      }
  }
}

DerivativesMap& ModelConversionTestPass::getDerivativesMap(ModelOp modelOp)
{
  if (auto analysis = getCachedChildAnalysis<DerivativesMap>(modelOp)) {
    return *analysis;
  }

  auto& analysis = getChildAnalysis<DerivativesMap>(modelOp);
  analysis.initialize();
  return analysis;
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createModelConversionTestPass()
  {
    return std::make_unique<ModelConversionTestPass>();
  }

  std::unique_ptr<mlir::Pass> createModelConversionTestPass(
      const ModelConversionTestPassOptions& options)
  {
    return std::make_unique<ModelConversionTestPass>(options);
  }
}
