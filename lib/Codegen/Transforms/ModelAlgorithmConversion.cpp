#include "marco/Codegen/Transforms/ModelAlgorithmConversion.h"
#include "marco/Dialect/BaseModelica/BaseModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_MODELALGORITHMCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

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
        rewriter.createBlock(&functionOp.getBodyRegion());

        // Declare the variables.
        rewriter.setInsertionPointToStart(functionOp.getBody());
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
        rewriter.setInsertionPointToEnd(functionOp.getBody());
        auto algorithmOp = rewriter.create<AlgorithmOp>(loc);

        rewriter.inlineRegionBefore(
            op.getBodyRegion(),
            algorithmOp.getBodyRegion(),
            algorithmOp.getBodyRegion().end());

        // Create the equation templates.
        llvm::SmallVector<EquationTemplateOp> templateOps;

        for (size_t i = 0, e = outputVariables.size(); i < e; ++i) {
          VariableOp outputVariable = outputVariables[i];
          VariableType variableType = outputVariable.getVariableType();
          int64_t rank = variableType.getRank();

          rewriter.setInsertionPointToEnd(modelOp.getBody());

          auto templateOp = rewriter.create<EquationTemplateOp>(loc);
          templateOps.push_back(templateOp);
          mlir::Block* templateBody = templateOp.createBody(rank);
          rewriter.setInsertionPointToStart(templateBody);

          llvm::SmallVector<mlir::Value> inputVariableGetOps;

          for (VariableOp variable : inputVariables) {
            inputVariableGetOps.push_back(
                rewriter.create<VariableGetOp>(loc, variable));
          }

          auto callOp = rewriter.create<CallOp>(
              loc, functionOp, inputVariableGetOps);

          mlir::Value lhs =
              rewriter.create<VariableGetOp>(loc, outputVariables[i]);

          mlir::Value rhs = callOp.getResult(i);

          if (auto inductions = templateOp.getInductionVariables();
                  !inductions.empty()) {
            lhs = rewriter.create<LoadOp>(
                lhs.getLoc(), lhs, templateOp.getInductionVariables());

            rhs = rewriter.create<LoadOp>(
                rhs.getLoc(), rhs, templateOp.getInductionVariables());
          }

          mlir::Value lhsOp = rewriter.create<EquationSideOp>(loc, lhs);
          mlir::Value rhsOp = rewriter.create<EquationSideOp>(loc, rhs);
          rewriter.create<EquationSidesOp>(loc, lhsOp, rhsOp);
        }

        // Create the equation instances.
        rewriter.setInsertionPointToEnd(modelOp.getBody());

        if (op.getInitial()) {
          auto initialModelOp = rewriter.create<InitialModelOp>(loc);
          rewriter.createBlock(&initialModelOp.getBodyRegion());
          rewriter.setInsertionPointToStart(initialModelOp.getBody());
        } else {
          auto dynamicOp = rewriter.create<DynamicOp>(loc);
          rewriter.createBlock(&dynamicOp.getBodyRegion());
          rewriter.setInsertionPointToStart(dynamicOp.getBody());
        }

        for (size_t i = 0, e = outputVariables.size(); i < e; ++i) {
          EquationTemplateOp templateOp = templateOps[i];
          IndexSet variableIndices = outputVariables[i].getIndices();

          if (variableIndices.empty()) {
            rewriter.create<EquationInstanceOp>(templateOp.getLoc(), templateOp);
          } else {
            for (const MultidimensionalRange& range : llvm::make_range(
                     variableIndices.rangesBegin(),
                     variableIndices.rangesEnd())) {
              auto instanceOp = rewriter.create<EquationInstanceOp>(
                  templateOp.getLoc(), templateOp);

              instanceOp.setIndicesAttr(MultidimensionalRangeAttr::get(
                  rewriter.getContext(), range));
            }
          }
        }

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
          assert(startOp.getVariable().getNestedReferences().empty());
          
          if (startOp.getVariable().getRootReference() == variable) {
            return startOp;
          }
        }

        llvm_unreachable("StartOp not found");
        return nullptr;
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
      : public mlir::bmodelica::impl::ModelAlgorithmConversionPassBase<
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

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createModelAlgorithmConversionPass()
  {
    return std::make_unique<ModelAlgorithmConversionPass>();
  }
}
