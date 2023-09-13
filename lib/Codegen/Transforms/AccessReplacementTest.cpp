#include "marco/Codegen/Transforms/AccessReplacementTest.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_ACCESSREPLACEMENTTESTPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

using EquationsMap = llvm::StringMap<EquationInstanceOp>;

namespace
{
  class AccessReplacePattern
      : public mlir::OpRewritePattern<EquationInstanceOp>
  {
    public:
      AccessReplacePattern(
          mlir::MLIRContext* context,
          mlir::SymbolTableCollection& symbolTableCollection,
          const EquationsMap& equationsMap)
          : mlir::OpRewritePattern<EquationInstanceOp>(context),
            symbolTableCollection(&symbolTableCollection),
            equationsMap(&equationsMap)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          EquationInstanceOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        // Obtain the information about the access to be replaced.
        IndexSet indices;

        if (auto indicesAttr =
                op->getAttrOfType<IndexSetAttr>("replace_indices")) {
          indices = indicesAttr.getValue();
        }

        auto destinationPath =
            op->getAttrOfType<EquationPathAttr>("replace_destination_path");

        auto replacementEquationId =
            op->getAttrOfType<mlir::StringAttr>("replace_eq");

        auto sourcePath =
            op->getAttrOfType<EquationPathAttr>("replace_source_path");

        if (!destinationPath || !replacementEquationId || !sourcePath) {
          return mlir::failure();
        }

        std::optional<VariableAccess> destinationAccess = op.getAccessAtPath(
            *symbolTableCollection, destinationPath.getValue());

        if (!destinationAccess) {
          return mlir::failure();
        }

        EquationInstanceOp replacementEquation =
            equationsMap->lookup(replacementEquationId.getValue());

        if (!replacementEquation) {
          return mlir::failure();
        }

        std::optional<VariableAccess> sourceAccess =
            replacementEquation.getAccessAtPath(
                *symbolTableCollection, sourcePath.getValue());

        if (!sourceAccess) {
          return mlir::failure();
        }

        // Remove the attributes before cloning, in order to avoid infinite
        // loops.

        op->removeAttr("replace_indices");
        op->removeAttr("replace_destination_path");
        op->removeAttr("replace_eq");
        op->removeAttr("replace_source_path");

        // Perform the replacement.
        llvm::SmallVector<EquationInstanceOp> clones;

        if (mlir::failed(op.cloneWithReplacedAccess(
                rewriter, std::reference_wrapper<const IndexSet>(indices),
                *destinationAccess, replacementEquation.getTemplate(),
                *sourceAccess, clones))) {
          return mlir::failure();
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }

    private:
      mlir::SymbolTableCollection* symbolTableCollection;
      const EquationsMap* equationsMap;
  };
}

namespace
{
  class AccessReplacementTestPass
      : public mlir::modelica::impl::AccessReplacementTestPassBase<
            AccessReplacementTestPass>
  {
    public:
      using AccessReplacementTestPassBase::AccessReplacementTestPassBase;

      void runOnOperation() override
      {
        ModelOp modelOp = getOperation();

        llvm::SmallVector<EquationInstanceOp> equationOps;
        EquationsMap equationsMap;

        modelOp->walk([&](EquationInstanceOp equationOp) {
          equationOps.push_back(equationOp);

          if (auto idAttr =
                  equationOp->getAttrOfType<mlir::StringAttr>("id")) {
            equationsMap[idAttr.getValue()] = equationOp;
          }
        });

        mlir::RewritePatternSet patterns(&getContext());
        mlir::SymbolTableCollection symbolTableCollection;

        patterns.add<AccessReplacePattern>(
            &getContext(), symbolTableCollection, equationsMap);

        if (mlir::failed(applyPatternsAndFoldGreedily(
                modelOp, std::move(patterns)))) {
          return signalPassFailure();
        }
      }
  };
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createAccessReplacementTestPass()
  {
    return std::make_unique<AccessReplacementTestPass>();
  }
}
