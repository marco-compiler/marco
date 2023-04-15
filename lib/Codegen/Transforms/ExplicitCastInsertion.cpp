#include "marco/Codegen/Transforms/ExplicitCastInsertion.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_EXPLICITCASTINSERTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class CallOpScalarPattern : public mlir::OpRewritePattern<CallOp>
  {
    public:
      CallOpScalarPattern(
          mlir::MLIRContext* context, mlir::SymbolTableCollection& symbolTable)
          : mlir::OpRewritePattern<CallOp>(context),
            symbolTable(&symbolTable)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          CallOp op, mlir::PatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

        auto calleeFunctionOp = mlir::cast<FunctionOp>(
            op.getFunction(moduleOp, *symbolTable));

        assert(op.getArgs().size() ==
               calleeFunctionOp.getArgumentTypes().size());

        auto pairs = llvm::zip(
            op.getArgs(), calleeFunctionOp.getArgumentTypes());

        for (auto [ arg, type ] : pairs) {
          mlir::Type actualType = arg.getType();

          if (!actualType.isa<ArrayType>() && !type.isa<ArrayType>()) {
            continue;
          }

          if (!actualType.isa<ArrayType>() && type.isa<ArrayType>()) {
            return mlir::failure();
          }

          if (actualType.isa<ArrayType>() && !type.isa<ArrayType>()) {
            return mlir::failure();
          }

          if (actualType.cast<ArrayType>().getRank() !=
              type.cast<ArrayType>().getRank()) {
            return mlir::failure();
          }
        }

        mlir::Location location = op->getLoc();
        llvm::SmallVector<mlir::Value, 3> args;

        for (auto [ arg, type ] : llvm::zip(
                 op.getArgs(), calleeFunctionOp.getArgumentTypes())) {
          if (arg.getType() != type) {
            if (arg.getType().isa<ArrayType>()) {
              arg = rewriter.create<ArrayCastOp>(location, type, arg);
            } else {
              arg = rewriter.create<CastOp>(location, type, arg);
            }
          }

          args.push_back(arg);
        }

        rewriter.replaceOpWithNewOp<CallOp>(
            op, op.getCallee(), op.getResultTypes(), args);

        return mlir::success();
      }

    private:
      mlir::SymbolTableCollection* symbolTable;
  };

  struct SubscriptionOpPattern : public mlir::OpRewritePattern<SubscriptionOp>
  {
    using mlir::OpRewritePattern<SubscriptionOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        SubscriptionOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location location = op->getLoc();
      llvm::SmallVector<mlir::Value, 3> indexes;

      for (mlir::Value index : op.getIndices()) {
        if (!index.getType().isa<mlir::IndexType>()) {
          index = rewriter.create<CastOp>(
              location, rewriter.getIndexType(), index);
        }

        indexes.push_back(index);
      }

      rewriter.replaceOpWithNewOp<SubscriptionOp>(op, op.getSource(), indexes);
      return mlir::success();
    }
  };

  struct ConditionOpPattern : public mlir::OpRewritePattern<ConditionOp>
  {
    using mlir::OpRewritePattern<ConditionOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        ConditionOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location location = op->getLoc();
      mlir::Value condition = rewriter.create<CastOp>(
          location, BooleanType::get(op.getContext()), op.getCondition());
      rewriter.replaceOpWithNewOp<ConditionOp>(op, condition);
      return mlir::success();
    }
  };
}

static void populateExplicitCastInsertionPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::MLIRContext* context,
    mlir::SymbolTableCollection& symbolTable)
{
	patterns.insert<CallOpScalarPattern>(context, symbolTable);

  patterns.insert<
      SubscriptionOpPattern,
      ConditionOpPattern>(context);
}

namespace
{
  class ExplicitCastInsertionPass
      : public impl::ExplicitCastInsertionPassBase<ExplicitCastInsertionPass>
  {
    public:
      using ExplicitCastInsertionPassBase::ExplicitCastInsertionPassBase;

      void runOnOperation() override
      {
        auto moduleOp = getOperation();

        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<ModelicaDialect>();
        target.addLegalDialect<mlir::scf::SCFDialect>();

        // Create an instance of the symbol table in order to reduce the cost
        // of looking for symbols.
        mlir::SymbolTableCollection symbolTable;

        target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
          auto calleeFunctionOp = mlir::cast<FunctionOp>(
              op.getFunction(moduleOp, symbolTable));

          if (calleeFunctionOp == nullptr) {
            return true;
          }

          assert(op.getArgs().size() ==
                 calleeFunctionOp.getArgumentTypes().size());

          auto pairs = llvm::zip(
              op.getArgs(), calleeFunctionOp.getArgumentTypes());

          return llvm::all_of(pairs, [&](const auto& pair) {
            return std::get<0>(pair).getType() == std::get<1>(pair);
          });
        });

        target.addDynamicallyLegalOp<SubscriptionOp>([](SubscriptionOp op) {
          auto indexes = op.getIndices();

          return llvm::all_of(indexes, [](mlir::Value index) {
            return index.getType().isa<mlir::IndexType>();
          });
        });

        target.addDynamicallyLegalOp<ConditionOp>([](ConditionOp op) {
          mlir::Type conditionType = op.getCondition().getType();
          return conditionType.isa<BooleanType>();
        });

        mlir::RewritePatternSet patterns(&getContext());

        populateExplicitCastInsertionPatterns(
            patterns, &getContext(), symbolTable);

        if (mlir::failed(applyPartialConversion(
                moduleOp, target, std::move(patterns)))) {
          mlir::emitError(
              moduleOp.getLoc(), "Error in inserting the explicit casts");

          return signalPassFailure();
        }
      }
  };
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createExplicitCastInsertionPass()
  {
    return std::make_unique<ExplicitCastInsertionPass>();
  }
}
