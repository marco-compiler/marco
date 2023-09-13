#include "marco/Codegen/Transforms/ModelAlgorithmConversion.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include <stack>

namespace mlir::modelica
{
#define GEN_PASS_DEF_MODELALGORITHMCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  template<typename Op>
  class AlgorithmInterfacePattern : public mlir::OpRewritePattern<Op>
  {
    public:
      AlgorithmInterfacePattern(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTable)
          : mlir::OpRewritePattern<Op>(context),
            symbolTable(&symbolTable)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          Op op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto algorithmInt = mlir::cast<AlgorithmInterface>(op.getOperation());
        auto modelOp = algorithmInt->template getParentOfType<ModelOp>();

        // Determine the read and written variables.
        llvm::DenseSet<VariableOp> readVariables;
        llvm::DenseSet<VariableOp> writtenVariables;

        algorithmInt.walk([&](VariableGetOp getOp) {
          auto variableOp = symbolTable->lookupSymbolIn<VariableOp>(
              modelOp, getOp.getVariableAttr());

          if (variableOp.getVariableType().isScalar()) {
            readVariables.insert(variableOp);
          } else {
            bool isRead, isWritten;
            std::tie(isRead, isWritten) =
                determineReadWrite(getOp.getResult());

            if (isRead) {
              readVariables.insert(variableOp);
            } else if (isWritten) {
              writtenVariables.insert(variableOp);
            }
          }
        });

        algorithmInt.walk([&](VariableSetOp setOp) {
          auto variableOp = symbolTable->lookupSymbolIn<VariableOp>(
              modelOp, setOp.getVariableAttr());

          writtenVariables.insert(variableOp);
        });

        // Determine the input and output variables of the function.
        // If a variable is read, but not written, then it will be an argument
        // of the function. All written variables are results of the function.
        llvm::SmallVector<VariableOp> inputVariables;
        llvm::SmallVector<VariableOp> outputVariables(
            writtenVariables.begin(), writtenVariables.end());

        for (VariableOp readVariable : readVariables) {
          if (!writtenVariables.contains(readVariable)) {
            inputVariables.push_back(readVariable);
          }
        }

        // Obtain a unique name for the function to be created.
        std::string functionName = getFunctionName(modelOp);

        // Create the function.
        auto moduleOp = modelOp->template getParentOfType<mlir::ModuleOp>();
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        auto functionOp = rewriter.create<FunctionOp>(loc, functionName);

        // Declare the variables.
        rewriter.setInsertionPointToStart(functionOp.bodyBlock());
        mlir::IRMapping mapping;

        for (VariableOp variableOp : inputVariables) {
          auto clonedVariableOp = mlir::cast<VariableOp>(
              rewriter.clone(*variableOp.getOperation(), mapping));

          auto originalVariableType = variableOp.getVariableType();

          clonedVariableOp.setType(VariableType::get(
              originalVariableType.getShape(),
              originalVariableType.getElementType(),
              VariabilityProperty::none,
              IOProperty::input));
        }

        for (VariableOp variableOp : outputVariables) {
          auto clonedVariableOp = mlir::cast<VariableOp>(
              rewriter.clone(*variableOp.getOperation(), mapping));

          auto originalVariableType = variableOp.getVariableType();

          clonedVariableOp.setType(VariableType::get(
              originalVariableType.getShape(),
              originalVariableType.getElementType(),
              VariabilityProperty::none,
              IOProperty::output));
        }

        // Set the default value of the output variables.
        for (VariableOp variableOp : outputVariables) {
          StartOp startOp = getStartOp(modelOp, variableOp.getSymName());

          auto defaultOp = rewriter.create<DefaultOp>(
              startOp.getLoc(), variableOp.getSymName());

          rewriter.cloneRegionBefore(
              startOp.getBodyRegion(),
              defaultOp.getBodyRegion(),
              defaultOp.getBodyRegion().end());

          if (startOp.getEach()) {
            mlir::OpBuilder::InsertionGuard guard(rewriter);

            auto yieldOp = mlir::cast<YieldOp>(
                defaultOp.getBodyRegion().back().getTerminator());

            assert(yieldOp.getValues().size() == 1);
            rewriter.setInsertionPoint(yieldOp);

            mlir::Value array = rewriter.create<ArrayBroadcastOp>(
                yieldOp.getLoc(),
                variableOp.getVariableType().unwrap(),
                yieldOp.getValues()[0]);

            rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, array);
          }
        }

        // Create the algorithm inside the function and move the original body
        // into it.
        rewriter.setInsertionPointToEnd(functionOp.bodyBlock());
        auto algorithmOp = rewriter.create<AlgorithmOp>(loc);

        rewriter.inlineRegionBefore(
            algorithmInt->getRegion(0),
            algorithmOp.getBodyRegion(),
            algorithmOp.getBodyRegion().end());

        // Create the equation containing the call to the function.
        rewriter.setInsertionPointToEnd(modelOp.getBody());

        mlir::Region* equationRegion =
            createEquation(rewriter, loc, outputVariables);

        rewriter.setInsertionPointToStart(&equationRegion->front());

        llvm::SmallVector<mlir::Value> inputVariableGetOps;
        llvm::SmallVector<mlir::Value> outputVariableGetOps;

        for (VariableOp inputVariable : inputVariables) {
          inputVariableGetOps.push_back(rewriter.create<VariableGetOp>(
              loc,
              inputVariable.getVariableType().unwrap(),
              inputVariable.getSymName()));
        }

        for (VariableOp outputVariable : outputVariables) {
          outputVariableGetOps.push_back(rewriter.create<VariableGetOp>(
              loc,
              outputVariable.getVariableType().unwrap(),
              outputVariable.getSymName()));
        }

        auto callOp = rewriter.create<CallOp>(
            loc, functionOp, inputVariableGetOps);

        mlir::Value lhs = rewriter.create<EquationSideOp>(
            loc, outputVariableGetOps);

        mlir::Value rhs = rewriter.create<EquationSideOp>(
            loc, callOp.getResults());

        rewriter.create<EquationSidesOp>(loc, lhs, rhs);

        // Erase the algorithm.
        rewriter.eraseOp(op);

        return mlir::success();
      }

      virtual std::string getFunctionName(ModelOp modelOp) const = 0;

      StartOp getStartOp(ModelOp modelOp, llvm::StringRef variable) const
      {
        for (StartOp startOp : modelOp.getOps<StartOp>()) {
          if (startOp.getVariable() == variable) {
            return startOp;
          }
        }

        llvm_unreachable("StartOp not found");
        return nullptr;
      }

      virtual mlir::Region* createEquation(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::ArrayRef<VariableOp> outputVariables) const = 0;

    private:
      /// Determine if an array is read or written.
      /// The return value consists in pair of boolean values, respectively
      /// indicating whether the array is read and written.
      std::pair<bool, bool> determineReadWrite(mlir::Value array) const
      {
        assert(array.getType().isa<ArrayType>());

        bool read = false;
        bool write = false;

        std::stack<mlir::Value> aliases;
        aliases.push(array);

        auto shouldStopEarly = [&read, &write]() {
          // Stop early if both a read and write have been found.
          return read && write;
        };

        // Keep the vector outside the loop, in order to avoid a stack overflow
        llvm::SmallVector<mlir::SideEffects::EffectInstance<
            mlir::MemoryEffects::Effect>> effects;

        while (!aliases.empty() && !shouldStopEarly()) {
          mlir::Value alias = aliases.top();
          aliases.pop();

          std::stack<mlir::Operation*> ops;

          for (const auto& user : alias.getUsers()) {
            ops.push(user);
          }

          while (!ops.empty() && !shouldStopEarly()) {
            mlir::Operation* op = ops.top();
            ops.pop();

            effects.clear();

            if (auto memoryInterface =
                    mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
              memoryInterface.getEffectsOnValue(alias, effects);

              read |= llvm::any_of(effects, [](const auto& effect) {
                return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
              });

              write |= llvm::any_of(effects, [](const auto& effect) {
                return mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect());
              });
            }

            if (auto viewInterface =
                    mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
              if (viewInterface.getViewSource() == alias) {
                for (const auto& result : viewInterface->getResults()) {
                  aliases.push(result);
                }
              }
            }
          }
        }

        return std::make_pair(read, write);
      }

    private:
      mlir::SymbolTableCollection* symbolTable;
  };

  class AlgorithmOpPattern : public AlgorithmInterfacePattern<AlgorithmOp>
  {
    public:
      AlgorithmOpPattern(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTable,
          size_t& functionsCounter)
          : AlgorithmInterfacePattern(context, symbolTable),
            functionsCounter(&functionsCounter)
      {
      }

      std::string getFunctionName(ModelOp modelOp) const override
      {
        return modelOp.getSymName().str() +
            "_algorithm_" + std::to_string((*functionsCounter)++);
      }

      mlir::Region* createEquation(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::ArrayRef<VariableOp> outputVariables) const override
      {
        mlir::OpBuilder::InsertionGuard guard(builder);

        // Create the equation template.
        auto equationTemplateOp = builder.create<EquationTemplateOp>(loc);
        assert(equationTemplateOp.getBodyRegion().empty());
        builder.createBlock(&equationTemplateOp.getBodyRegion());

        // Create the equation instance.
        builder.setInsertionPointAfter(equationTemplateOp);

        for (uint64_t i = 0, e = outputVariables.size(); i < e; ++i) {
          auto equationInstanceOp = builder.create<EquationInstanceOp>(
              loc, equationTemplateOp, false);

          equationInstanceOp.setViewElementIndex(i);

          VariableOp variableOp = outputVariables[i];
          VariableType variableType = variableOp.getVariableType();

          if (!variableType.isScalar()) {
            llvm::SmallVector<Range> ranges;

            for (uint64_t dimension : variableType.getShape()) {
              ranges.push_back(Range(0, dimension));
            }

            equationInstanceOp.setImplicitIndicesAttr(
                MultidimensionalRangeAttr::get(
                    builder.getContext(), MultidimensionalRange(ranges)));
          }
        }

        return &equationTemplateOp.getBodyRegion();
      }

    private:
      size_t* functionsCounter;
  };

  class InitialAlgorithmOpPattern
      : public AlgorithmInterfacePattern<InitialAlgorithmOp>
  {
    public:
      InitialAlgorithmOpPattern(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTable,
          size_t& functionsCounter)
          : AlgorithmInterfacePattern(context, symbolTable),
            functionsCounter(&functionsCounter)
      {
      }

      std::string getFunctionName(ModelOp modelOp) const override
      {
        return modelOp.getSymName().str() +
            "_initial_algorithm_" + std::to_string((*functionsCounter)++);
      }

      mlir::Region* createEquation(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::ArrayRef<VariableOp> outputVariables) const override
      {
        mlir::OpBuilder::InsertionGuard guard(builder);

        // Create the equation template.
        auto equationTemplateOp = builder.create<EquationTemplateOp>(loc);
        assert(equationTemplateOp.getBodyRegion().empty());
        builder.createBlock(&equationTemplateOp.getBodyRegion());

        // Create the equation instance.
        builder.setInsertionPointAfter(equationTemplateOp);

        for (uint64_t i = 0, e = outputVariables.size(); i < e; ++i) {
          auto equationInstanceOp = builder.create<EquationInstanceOp>(
              loc, equationTemplateOp, true);

          equationInstanceOp.setViewElementIndex(i);

          VariableOp variableOp = outputVariables[i];
          VariableType variableType = variableOp.getVariableType();

          if (!variableType.isScalar()) {
            llvm::SmallVector<Range> ranges;

            for (uint64_t dimension : variableType.getShape()) {
              ranges.push_back(Range(0, dimension));
            }

            equationInstanceOp.setImplicitIndicesAttr(
                MultidimensionalRangeAttr::get(
                    builder.getContext(), MultidimensionalRange(ranges)));
          }
        }

        return &equationTemplateOp.getBodyRegion();
      }

    private:
      size_t* functionsCounter;
  };
}

namespace
{
  class ModelAlgorithmConversionPass
      : public mlir::modelica::impl::ModelAlgorithmConversionPassBase<
            ModelAlgorithmConversionPass>
  {
    public:
      using ModelAlgorithmConversionPassBase::ModelAlgorithmConversionPassBase;

      void runOnOperation() override
      {
        mlir::ModuleOp moduleOp = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addDynamicallyLegalOp<InitialAlgorithmOp>(
            [](InitialAlgorithmOp op) {
              return !mlir::isa<ModelOp>(op->getParentOp());
            });

        target.addDynamicallyLegalOp<AlgorithmOp>([](AlgorithmOp op) {
          return !mlir::isa<ModelOp>(op->getParentOp());
        });

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::RewritePatternSet patterns(&getContext());

        // Counters for uniqueness of functions.
        size_t algorithmsCounter = 0;
        size_t initialAlgorithmsCounter = 0;

        mlir::SymbolTableCollection symbolTable;

        patterns.insert<AlgorithmOpPattern>(
            &getContext(), symbolTable, algorithmsCounter);

        patterns.insert<InitialAlgorithmOpPattern>(
            &getContext(), symbolTable, initialAlgorithmsCounter);

        if (mlir::failed(applyPartialConversion(
                moduleOp, target, std::move(patterns)))) {
          return signalPassFailure();
        }
      }
  };
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createModelAlgorithmConversionPass()
  {
    return std::make_unique<ModelAlgorithmConversionPass>();
  }
}
