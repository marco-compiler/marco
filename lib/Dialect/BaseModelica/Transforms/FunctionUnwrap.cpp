#include "marco/Dialect/BaseModelica/Transforms/FunctionUnwrap.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_FUNCTIONUNWRAPPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class FunctionUnwrapPass
      : public mlir::bmodelica::impl::FunctionUnwrapPassBase<FunctionUnwrapPass>
  {
    public:
      using FunctionUnwrapPassBase<FunctionUnwrapPass>::FunctionUnwrapPassBase;

      void runOnOperation() override;
  };
}

using CallsMap = llvm::DenseMap<FunctionOp, llvm::SmallVector<CallOp>>;

namespace
{
  class FunctionOpPattern : public mlir::OpRewritePattern<FunctionOp>
  {
    public:
      FunctionOpPattern(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTableCollection,
          const CallsMap& callsMap)
          : mlir::OpRewritePattern<FunctionOp>(context),
            symbolTableCollection(&symbolTableCollection),
            callsMap(&callsMap)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          FunctionOp op, mlir::PatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        // Compute the new name.
        llvm::SmallVector<std::string> namesStack;
        namesStack.push_back(op.getSymName().str());
        mlir::Operation* parentOp = op->getParentOp();

        while (parentOp && !mlir::isa<mlir::ModuleOp>(parentOp)) {
          if (auto classInterface = mlir::dyn_cast<ClassInterface>(parentOp)) {
            auto symbolInt = mlir::cast<mlir::SymbolOpInterface>(
                classInterface.getOperation());

            namesStack.push_back(symbolInt.getName().str());
          }

          parentOp = parentOp->getParentOp();
        }

        assert(!namesStack.empty());
        std::string newName = namesStack.back();

        for (size_t i = 1, e = namesStack.size(); i < e; ++i) {
          newName += "_" + namesStack[e - i - 1];
        }

        auto newFunctionOp = rewriter.create<FunctionOp>(op.getLoc(), newName);

        rewriter.inlineRegionBefore(
            op.getBodyRegion(),
            newFunctionOp.getBodyRegion(),
            newFunctionOp.getBodyRegion().end());

        // Add the new function to the module and to the symbol table.
        // This ensures that no name clash happens.
        symbolTableCollection->getSymbolTable(moduleOp).insert(newFunctionOp);

        // Update the calls to the old function.
        for (CallOp callOp : getCallOps(op)) {
          rewriter.setInsertionPoint(callOp);

          rewriter.replaceOpWithNewOp<CallOp>(
              callOp, newFunctionOp, callOp.getArgs(), callOp.getArgNames());
        }

        // Erase the old function.
        mlir::Operation* oldParentSymbolTable =
            op->getParentWithTrait<mlir::OpTrait::SymbolTable>();

        if (oldParentSymbolTable) {
          auto& oldSymbolTable =
              symbolTableCollection->getSymbolTable(oldParentSymbolTable);

          oldSymbolTable.remove(op);
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }

    private:
      llvm::ArrayRef<CallOp> getCallOps(FunctionOp functionOp) const
      {
        auto it = callsMap->find(functionOp);

        if (it == callsMap->end()) {
          return std::nullopt;
        }

        return it->getSecond();
      }

    private:
      mlir::SymbolTableCollection* symbolTableCollection;
      const CallsMap* callsMap;
  };
}

void FunctionUnwrapPass::runOnOperation()
{
  auto moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTableCollection;
  CallsMap callsMap;

  moduleOp.walk([&](CallOp callOp) {
    mlir::Operation* callee =
        callOp.getFunction(moduleOp, symbolTableCollection);

    if (callee && mlir::isa<FunctionOp>(callee)) {
      auto functionOp = mlir::cast<FunctionOp>(callee);
      callsMap[functionOp].push_back(callOp);
    }
  });

  mlir::ConversionTarget target(getContext());

  target.addDynamicallyLegalOp<FunctionOp>([&](FunctionOp op) {
    return mlir::isa<mlir::ModuleOp>(op->getParentOp());
  });

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<FunctionOpPattern>(
      &getContext(), symbolTableCollection, callsMap);

  if (mlir::failed(applyPartialConversion(
          moduleOp, target, std::move(patterns)))) {
    return signalPassFailure();
  }
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createFunctionUnwrapPass()
  {
    return std::make_unique<FunctionUnwrapPass>();
  }
}
