#include "marco/Codegen/Transforms/ModelAlgorithmConversion.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_MODELALGORITHMCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class AlgorithmOpPattern : public mlir::OpRewritePattern<AlgorithmOp>
  {
    public:
      AlgorithmOpPattern(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTable,
          size_t& initialAlgorithmsCounter,
          size_t& algorithmsCounter)
          : mlir::OpRewritePattern<AlgorithmOp>(context),
            symbolTable(&symbolTable),
            initialAlgorithmsCounter(&initialAlgorithmsCounter),
            algorithmsCounter(&algorithmsCounter)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          AlgorithmOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();
        auto modelOp = op->getParentOfType<ModelOp>();

        // Determine the read and written variables.
        llvm::DenseSet<VariableOp> readVariables;
        llvm::DenseSet<VariableOp> writtenVariables;

        op.walk([&](VariableGetOp getOp) {
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

        op.walk([&](VariableSetOp setOp) {
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
        std::string functionName = getFunctionName(modelOp, op);

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
            op.getBodyRegion(),
            algorithmOp.getBodyRegion(),
            algorithmOp.getBodyRegion().end());

        // Create the equation containing the call to the function.
        rewriter.setInsertionPointToEnd(modelOp.getBody());

        mlir::Region* equationRegion =
            createEquation(rewriter, loc, op, outputVariables);

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

    private:
      std::string getFunctionName(
          ModelOp modelOp, AlgorithmOp algorithmOp) const
      {
        std::string result = modelOp.getSymName().str();

        if (algorithmOp.getInitial()) {
          result += "_initial_algorithm_" +
              std::to_string((*initialAlgorithmsCounter)++);
        } else {
          result += "_algorithm_" + std::to_string((*algorithmsCounter)++);
        }

        return result;
      }

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

      mlir::Region* createEquation(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          AlgorithmOp algorithmOp,
          llvm::ArrayRef<VariableOp> outputVariables) const
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
              loc, equationTemplateOp, algorithmOp.getInitial());

          equationInstanceOp.setViewElementIndex(i);

          VariableOp variableOp = outputVariables[i];
          VariableType variableType = variableOp.getVariableType();

          if (!variableType.isScalar()) {
            llvm::SmallVector<Range> ranges;

            for (int64_t dimension : variableType.getShape()) {
              ranges.push_back(Range(0, dimension));
            }

            equationInstanceOp.setImplicitIndicesAttr(
                MultidimensionalRangeAttr::get(
                    builder.getContext(), MultidimensionalRange(ranges)));
          }
        }

        return &equationTemplateOp.getBodyRegion();
      }

      /// Determine if an array is read or written.
      /// The return value consists in pair of boolean values, respectively
      /// indicating whether the array is read and written.
      std::pair<bool, bool> determineReadWrite(mlir::Value array) const
      {
        assert(array.getType().isa<ArrayType>());

        bool read = false;
        bool write = false;

        llvm::SmallVector<mlir::Value> aliases;
        aliases.push_back(array);

        auto shouldStopEarly = [&read, &write]() {
          // Stop early if both a read and write have been found.
          return read && write;
        };

        // Keep the vector outside the loop, in order to avoid a stack overflow
        llvm::SmallVector<mlir::SideEffects::EffectInstance<
            mlir::MemoryEffects::Effect>> effects;

        while (!aliases.empty() && !shouldStopEarly()) {
          mlir::Value alias = aliases.pop_back_val();
          llvm::SmallVector<mlir::Operation*> ops;

          for (const auto& user : alias.getUsers()) {
            ops.push_back(user);
          }

          while (!ops.empty() && !shouldStopEarly()) {
            mlir::Operation* op = ops.pop_back_val();
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
                  aliases.push_back(result);
                }
              }
            }
          }
        }

        return std::make_pair(read, write);
      }

    private:
      mlir::SymbolTableCollection* symbolTable;
      size_t* initialAlgorithmsCounter;
      size_t* algorithmsCounter;
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
            &getContext(), symbolTable,
            initialAlgorithmsCounter, algorithmsCounter);

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
