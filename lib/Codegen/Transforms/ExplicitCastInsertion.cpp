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
  struct CallOpScalarPattern : public mlir::OpRewritePattern<CallOp>
  {
    using mlir::OpRewritePattern<CallOp>::OpRewritePattern;

    mlir::LogicalResult match(CallOp op) const override
    {
      auto module = op->getParentOfType<mlir::ModuleOp>();
      auto callee = module.lookupSymbol<FunctionOp>(op.getCallee());
      assert(op.getArgs().size() == callee.getArgumentTypes().size());
      auto pairs = llvm::zip(op.getArgs(), callee.getArgumentTypes());

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

        if (actualType.cast<ArrayType>().getRank() != type.cast<ArrayType>().getRank()) {
          return mlir::failure();
        }
      }

      return mlir::success();
    }

    void rewrite(CallOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location location = op->getLoc();

      auto module = op->getParentOfType<mlir::ModuleOp>();
      auto callee = module.lookupSymbol<FunctionOp>(op.getCallee());

      llvm::SmallVector<mlir::Value, 3> args;

      for (auto [ arg, type ] : llvm::zip(op.getArgs(), callee.getArgumentTypes())) {
        if (arg.getType() != type) {
          if (arg.getType().isa<ArrayType>()) {
            arg = rewriter.create<ArrayCastOp>(location, type, arg);
          } else {
            arg = rewriter.create<CastOp>(location, type, arg);
          }
        }

        args.push_back(arg);
      }

      auto movedResults = op.getMovedResults();

      if (movedResults.has_value()) {
        rewriter.replaceOpWithNewOp<CallOp>(op, op.getResultTypes(), op.getCallee(), args, rewriter.getI64IntegerAttr(movedResults->getSExtValue()));
      } else {
        rewriter.replaceOpWithNewOp<CallOp>(op, op.getCallee(), op.getResultTypes(), args);
      }
    }
  };

  struct SubscriptionOpPattern : public mlir::OpRewritePattern<SubscriptionOp>
  {
    using mlir::OpRewritePattern<SubscriptionOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SubscriptionOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location location = op->getLoc();
      llvm::SmallVector<mlir::Value, 3> indexes;

      for (mlir::Value index : op.getIndices()) {
        if (!index.getType().isa<mlir::IndexType>()) {
          index = rewriter.create<CastOp>(location, rewriter.getIndexType(), index);
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

    mlir::LogicalResult matchAndRewrite(ConditionOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location location = op->getLoc();
      mlir::Value condition = rewriter.create<CastOp>(location, BooleanType::get(op.getContext()), op.getCondition());
      rewriter.replaceOpWithNewOp<ConditionOp>(op, condition);
      return mlir::success();
    }
  };
}

static void populateExplicitCastInsertionPatterns(
    mlir::RewritePatternSet& patterns, mlir::MLIRContext* context)
{
	patterns.insert<
	    CallOpScalarPattern,
			SubscriptionOpPattern,
			ConditionOpPattern>(context);
}

namespace
{
  struct ExplicitCastInsertionTarget : public mlir::ConversionTarget {
    ExplicitCastInsertionTarget(mlir::MLIRContext& context) : mlir::ConversionTarget(context)
    {
      addDynamicallyLegalOp<CallOp>([](CallOp op) {
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto callee = module.lookupSymbol<FunctionOp>(op.getCallee());

        if (callee == nullptr) {
          return true;
        }

        assert(op.getArgs().size() == callee.getArgumentTypes().size());
        auto pairs = llvm::zip(op.getArgs(), callee.getArgumentTypes());

        return llvm::all_of(pairs, [&](const auto& pair) {
          return std::get<0>(pair).getType() == std::get<1>(pair);
        });
      });

      addDynamicallyLegalOp<SubscriptionOp>([](SubscriptionOp op) {
        auto indexes = op.getIndices();

        return llvm::all_of(indexes, [](mlir::Value index) {
          return index.getType().isa<mlir::IndexType>();
        });
      });

      addDynamicallyLegalOp<ConditionOp>([](ConditionOp op) {
        mlir::Type conditionType = op.getCondition().getType();
        return conditionType.isa<BooleanType>();
      });
    }
  };

  class ExplicitCastInsertionPass: public impl::ExplicitCastInsertionPassBase<ExplicitCastInsertionPass>
  {
    public:
      using ExplicitCastInsertionPassBase::ExplicitCastInsertionPassBase;

      void runOnOperation() override
      {
        auto module = getOperation();

        ExplicitCastInsertionTarget target(getContext());
        target.addLegalDialect<ModelicaDialect>();
        target.addLegalDialect<mlir::scf::SCFDialect>();

        mlir::RewritePatternSet patterns(&getContext());
        populateExplicitCastInsertionPatterns(patterns, &getContext());

        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
          mlir::emitError(module.getLoc(), "Error in inserting the explicit casts");
          signalPassFailure();
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
