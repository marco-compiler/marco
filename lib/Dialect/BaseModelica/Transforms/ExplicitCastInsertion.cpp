#include "marco/Dialect/BaseModelica/Transforms/ExplicitCastInsertion.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EXPLICITCASTINSERTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

static void getArgumentTypes(mlir::Operation *function,
                             llvm::SmallVectorImpl<mlir::Type> &argumentTypes) {
  if (auto functionOp = mlir::dyn_cast<FunctionOp>(function)) {
    argumentTypes.clear();

    for (mlir::Type type : functionOp.getArgumentTypes()) {
      argumentTypes.push_back(type);
    }

    return;
  }

  if (auto rawFunctionOp = mlir::dyn_cast<RawFunctionOp>(function)) {
    for (mlir::Type type : rawFunctionOp.getArgumentTypes()) {
      argumentTypes.push_back(type);
    }

    return;
  }

  if (auto equationFunctionOp = mlir::dyn_cast<EquationFunctionOp>(function)) {
    for (mlir::Type type : equationFunctionOp.getArgumentTypes()) {
      argumentTypes.push_back(type);
    }

    return;
  }
}

namespace {
class CallOpScalarPattern : public mlir::OpRewritePattern<CallOp> {
public:
  CallOpScalarPattern(mlir::MLIRContext *context,
                      mlir::SymbolTableCollection &symbolTable)
      : mlir::OpRewritePattern<CallOp>(context), symbolTable(&symbolTable) {}

  mlir::LogicalResult
  matchAndRewrite(CallOp op, mlir::PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    mlir::Operation *calleeOp = op.getFunction(moduleOp, *symbolTable);
    llvm::SmallVector<mlir::Type, 3> argumentTypes;
    getArgumentTypes(calleeOp, argumentTypes);

    assert(op.getArgs().size() == argumentTypes.size());

    for (auto [arg, type] : llvm::zip(op.getArgs(), argumentTypes)) {
      mlir::Type actualType = arg.getType();

      if (!mlir::isa<ArrayType>(actualType) && !mlir::isa<ArrayType>(type)) {
        continue;
      }

      if (!mlir::isa<ArrayType>(actualType) && mlir::isa<ArrayType>(type)) {
        return mlir::failure();
      }

      if (mlir::isa<ArrayType>(actualType) && !mlir::isa<ArrayType>(type)) {
        return mlir::failure();
      }

      if (mlir::cast<ArrayType>(actualType).getRank() !=
          mlir::cast<ArrayType>(type).getRank()) {
        return mlir::failure();
      }
    }

    mlir::Location location = op->getLoc();
    llvm::SmallVector<mlir::Value, 3> args;

    for (auto [arg, type] : llvm::zip(op.getArgs(), argumentTypes)) {
      if (arg.getType() != type) {
        if (mlir::isa<ArrayType>(arg.getType())) {
          arg = rewriter.create<ArrayCastOp>(location, type, arg);
        } else {
          arg = rewriter.create<CastOp>(location, type, arg);
        }
      }

      args.push_back(arg);
    }

    rewriter.replaceOpWithNewOp<CallOp>(op, op.getCallee(), op.getResultTypes(),
                                        args);

    return mlir::success();
  }

private:
  mlir::SymbolTableCollection *symbolTable;
};

struct SubscriptionOpPattern : public mlir::OpRewritePattern<SubscriptionOp> {
  using mlir::OpRewritePattern<SubscriptionOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SubscriptionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location location = op->getLoc();
    llvm::SmallVector<mlir::Value, 3> indexes;

    for (mlir::Value index : op.getIndices()) {
      if (!mlir::isa<mlir::IndexType>(index.getType())) {
        index =
            rewriter.create<CastOp>(location, rewriter.getIndexType(), index);
      }

      indexes.push_back(index);
    }

    rewriter.replaceOpWithNewOp<SubscriptionOp>(op, op.getSource(), indexes);
    return mlir::success();
  }
};

struct ConditionOpPattern : public mlir::OpRewritePattern<ConditionOp> {
  using mlir::OpRewritePattern<ConditionOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConditionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location location = op->getLoc();
    mlir::Value condition = rewriter.create<CastOp>(
        location, BooleanType::get(op.getContext()), op.getCondition());
    rewriter.replaceOpWithNewOp<ConditionOp>(op, condition);
    return mlir::success();
  }
};
} // namespace

static void populateExplicitCastInsertionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::SymbolTableCollection &symbolTable) {
  patterns.insert<CallOpScalarPattern>(context, symbolTable);

  patterns.insert<SubscriptionOpPattern, ConditionOpPattern>(context);
}

namespace {
class ExplicitCastInsertionPass
    : public impl::ExplicitCastInsertionPassBase<ExplicitCastInsertionPass> {
public:
  using ExplicitCastInsertionPassBase<
      ExplicitCastInsertionPass>::ExplicitCastInsertionPassBase;

  void runOnOperation() override;
};
} // namespace

void ExplicitCastInsertionPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();

  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<BaseModelicaDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();

  // Create an instance of the symbol table in order to reduce the cost
  // of looking for symbols.
  mlir::SymbolTableCollection symbolTable;

  target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
    auto calleeOp = op.getFunction(moduleOp, symbolTable);

    if (calleeOp == nullptr) {
      return true;
    }

    llvm::SmallVector<mlir::Type, 3> argumentTypes;
    getArgumentTypes(calleeOp, argumentTypes);

    assert(op.getArgs().size() == argumentTypes.size());

    return llvm::all_of(
        llvm::zip(op.getArgs(), argumentTypes), [&](const auto &pair) {
          return std::get<0>(pair).getType() == std::get<1>(pair);
        });
  });

  target.addDynamicallyLegalOp<SubscriptionOp>([](SubscriptionOp op) {
    auto indexes = op.getIndices();

    return llvm::all_of(indexes, [](mlir::Value index) {
      return mlir::isa<mlir::IndexType>(index.getType());
    });
  });

  target.addDynamicallyLegalOp<ConditionOp>([](ConditionOp op) {
    mlir::Type conditionType = op.getCondition().getType();
    return mlir::isa<BooleanType>(conditionType);
  });

  mlir::RewritePatternSet patterns(&getContext());

  populateExplicitCastInsertionPatterns(patterns, &getContext(), symbolTable);

  if (mlir::failed(
          applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    mlir::emitError(moduleOp.getLoc(), "Error in inserting the explicit casts");

    return signalPassFailure();
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createExplicitCastInsertionPass() {
  return std::make_unique<ExplicitCastInsertionPass>();
}
} // namespace mlir::bmodelica
