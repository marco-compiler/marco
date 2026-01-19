#include "marco/Frontend/Passes/EquationOffloading.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Frontend/Passes/EquationTargets/CPUEquationTarget.h"
#include "marco/Frontend/Passes/EquationTargets/EquationTarget.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinOps.h"

using namespace ::marco::frontend;

namespace {
class EquationOffloadingPass
    : public mlir::PassWrapper<EquationOffloadingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  EquationOffloadingPassOptions options;

public:
  explicit EquationOffloadingPass(EquationOffloadingPassOptions options);

  std::unique_ptr<Pass> clonePass() const override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  void runOnOperation() override;

  llvm::SmallVector<EquationTarget *>
  getTargetsByCost(mlir::func::FuncOp equationFunction);

  /// Attach the candidate targets as an attribute for debugging purposes.
  void
  attachCandidateTargetsAsAttribute(mlir::func::FuncOp equationFunction,
                                    llvm::ArrayRef<EquationTarget *> targets);

  /// Generate the sequence of transforms for offloading to one of the given
  /// targets.
  mlir::transform::SequenceOp
  generateTransform(mlir::func::FuncOp equationFunction,
                    llvm::ArrayRef<EquationTarget *> targets);
};
} // namespace

EquationOffloadingPass::EquationOffloadingPass(
    EquationOffloadingPassOptions options)
    : options(std::move(options)) {
  if (options.targets.empty()) {
    options.targets.push_back(std::make_unique<CPUEquationTarget>("generic"));
  }
}

std::unique_ptr<mlir::Pass> EquationOffloadingPass::clonePass() const {
  return std::make_unique<EquationOffloadingPass>(options);
}

void EquationOffloadingPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<mlir::transform::TransformDialect>();
}

static constexpr llvm::StringLiteral targetCandidatesAttrName =
    "target_candidates";

void EquationOffloadingPass::runOnOperation() {
  // Collect all the requested targets.
  llvm::SetVector<std::string> targets;

  for (mlir::func::FuncOp funcOp :
       getOperation().getOps<mlir::func::FuncOp>()) {
    if (funcOp->hasAttr(
            mlir::bmodelica::BaseModelicaDialect::kEquationFunctionAttrName)) {
      for (auto target :
           funcOp->getAttrOfType<mlir::ArrayAttr>(targetCandidatesAttrName)) {
        targets.insert(mlir::cast<mlir::StringAttr>(target).getValue().str());
      }
    }
  }

  getOperation()->setAttr(
      mlir::transform::TransformDialect::kWithNamedSequenceAttrName,
      mlir::UnitAttr::get(&getContext()));

  llvm::SmallVector<mlir::func::FuncOp> equationFunctions;

  for (mlir::func::FuncOp funcOp :
       getOperation().getOps<mlir::func::FuncOp>()) {
    if (funcOp->hasAttr(
            mlir::bmodelica::BaseModelicaDialect::kEquationFunctionAttrName)) {
      equationFunctions.push_back(funcOp);
    }
  }

  for (mlir::func::FuncOp funcOp : equationFunctions) {
    llvm::SmallVector<EquationTarget *> targets = getTargetsByCost(funcOp);

    if (options.attachCandidateTargetsAsAttribute) {
      attachCandidateTargetsAsAttribute(funcOp, targets);
    }

    auto transformOp = generateTransform(funcOp, targets);
    mlir::RaggedArray<mlir::transform::MappedValue> extraMappings;
    mlir::transform::TransformOptions transformOptions;

#ifdef NDEBUG
    transformOptions.enableExpensiveChecks(false);
#else
    transformOptions.enableExpensiveChecks(true);
#endif

    /*
        if (mlir::failed(mlir::transform::applyTransforms(
                funcOp, transformOp, extraMappings, transformOptions))) {
          signalPassFailure();
          return;
        }
        */

    // transformOp.erase();
  }
}

llvm::SmallVector<EquationTarget *>
EquationOffloadingPass::getTargetsByCost(mlir::func::FuncOp equationFunction) {
  llvm::SmallVector<EquationTarget *> targets;

  for (const auto &target : options.targets) {
    if (target->isCompatible(equationFunction)) {
      targets.push_back(target.get());
    }
  }

  llvm::sort(targets, [&](EquationTarget *a, EquationTarget *b) {
    return a->getCost(equationFunction) < b->getCost(equationFunction);
  });

  return targets;
}

void EquationOffloadingPass::attachCandidateTargetsAsAttribute(
    mlir::func::FuncOp equationFunction,
    llvm::ArrayRef<EquationTarget *> targets) {
  llvm::SmallVector<mlir::Attribute> targetNames;

  for (EquationTarget *target : targets) {
    targetNames.push_back(
        mlir::StringAttr::get(&getContext(), target->getName()));
  }

  equationFunction->setAttr(targetCandidatesAttrName,
                            mlir::ArrayAttr::get(&getContext(), targetNames));
}

mlir::transform::SequenceOp EquationOffloadingPass::generateTransform(
    mlir::func::FuncOp equationFunction,
    llvm::ArrayRef<EquationTarget *> targets) {
  // Temporarily restrict the possible targets to just one choice, accepting the
  // target that succeeds in code generation.
  // TODO In the long term, we should keep all the attached targets, and decide
  // and dynamically decide the offloading target at runtime. This would require
  // emitting the code for all the targets, but would enable online tuning. This
  // should be done as part of a bigger effort aiming to communicate to the
  // whole dependency graph to the runtime environment.

  mlir::OpBuilder builder(equationFunction);
  builder.setInsertionPointAfter(equationFunction);

  auto funcType = mlir::transform::OperationType::get(
      &getContext(), equationFunction.getOperationName());

  return builder.create<mlir::transform::SequenceOp>(
      equationFunction.getLoc(), mlir::TypeRange{},
      mlir::transform::FailurePropagationMode::Propagate, funcType,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc,
          mlir::BlockArgument arg) {
        auto alternativesOp =
            nestedBuilder.create<mlir::transform::AlternativesOp>(
                loc, mlir::TypeRange{}, arg, targets.size() + 1);

        for (auto target : llvm::enumerate(targets)) {
          // Create target-specific transform sequence.
          mlir::Region &alternativeRegion =
              alternativesOp.getAlternatives()[target.index()];

          nestedBuilder.setInsertionPointToStart(
              nestedBuilder.createBlock(&alternativeRegion, {}, funcType, loc));

          auto targetTransform = target.value()->createTransformSequence(
              nestedBuilder, equationFunction,
              alternativeRegion.getArguments().front());

          targetTransform.setFailurePropagationMode(
              mlir::transform::FailurePropagationMode::Suppress);

          nestedBuilder.create<mlir::transform::YieldOp>(loc);
        }

        // Add the fallback alternative that does not perform any modification.
        nestedBuilder.setInsertionPointToStart(nestedBuilder.createBlock(
            &alternativesOp.getAlternatives().back(), {}, funcType, loc));

        nestedBuilder.create<mlir::transform::YieldOp>(loc);

        nestedBuilder.setInsertionPointAfter(alternativesOp);
        nestedBuilder.create<mlir::transform::YieldOp>(loc);
      });
}

namespace marco::frontend {
EquationOffloadingPassOptions::EquationOffloadingPassOptions() = default;

EquationOffloadingPassOptions::EquationOffloadingPassOptions(
    const EquationOffloadingPassOptions &other) {
  for (const auto &target : other.targets) {
    targets.push_back(target->clone());
  }
}

EquationOffloadingPassOptions::EquationOffloadingPassOptions(
    EquationOffloadingPassOptions &&other) noexcept {
  targets = std::move(other.targets);
}

EquationOffloadingPassOptions &EquationOffloadingPassOptions::operator=(
    const EquationOffloadingPassOptions &other) {
  if (this != &other) {
    targets.clear();

    for (const auto &target : other.targets) {
      targets.push_back(target->clone());
    }
  }

  return *this;
}

EquationOffloadingPassOptions &EquationOffloadingPassOptions::operator=(
    EquationOffloadingPassOptions &&other) noexcept {
  targets = std::move(other.targets);
  return *this;
}

std::unique_ptr<mlir::Pass>
createEquationOffloadingPass(EquationOffloadingPassOptions options) {
  return std::make_unique<EquationOffloadingPass>(std::move(options));
}
} // namespace marco::frontend
