//===- DataRecomputation.cpp - Handles recomputation candidates -----------===//
//
//===----------------------------------------------------------------------===//
//
// This file contains the DataRecomputation pass (pending a better name).
//===----------------------------------------------------------------------===//

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Transforms/DataRecomputation.h"

#include "marco/Dialect/Runtime/IR/Ops.h"
#include "marco/Modeling/GraphDumper.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Transforms/DataRecomputation/IndexExpression.h"
#include "marco/Transforms/DataRecomputation/OpTypeVariant.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Vector/IR/VectorDialect.h.inc"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

#define DRDBG() llvm::dbgs() << "DRCOMP: "

namespace mlir {
#define GEN_PASS_DEF_DATARECOMPUTATIONPASS
#include "marco/Transforms/Passes.h.inc"
} // namespace mlir

using namespace ::mlir::memref;
using namespace ::mlir::func;

using AccessSet = llvm::DenseSet<mlir::Operation *>;

namespace {


template <class... OpTys>
struct OpTypePack {};

template <class... OpTys>
struct OpTypeHelper {
  static bool anyOf(mlir::Operation *op) {
    return mlir::isa<OpTys...>(op);
  }
};

template <class... OpTys>
struct OpTypeHelper<OpTypePack<OpTys...>> {
  static bool anyOf(mlir::Operation *op) {
    return OpTypeHelper<OpTys...>::anyOf(op);
  }
};

using GlobalStaticAllocOps =
    OpTypePack<mlir::memref::GlobalOp, mlir::LLVM::GlobalOp>;
using ViewLike =
    OpTypePack<mlir::memref::SubViewOp, mlir::memref::ReinterpretCastOp,
               mlir::memref::ViewOp, mlir::memref::CastOp>;
using AllocLike = OpTypePack<mlir::memref::AllocOp, mlir::memref::AllocaOp>;
using LoadLike = OpTypePack<mlir::memref::LoadOp>;
using StoreLike = OpTypePack<mlir::memref::StoreOp>;

struct NumberedStore {
  mlir::Operation *store;
  mlir::Operation *allocatingOp;
};

void collectBaseMemrefs(mlir::Value v, llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor, llvm::SmallVectorImpl<mlir::Value> &bases)
{
  llvm::SmallVector<mlir::Value, 4> worklist{v};
  llvm::SmallDenseSet<mlir::Value> visited{};

  while ( ! worklist.empty() ) {
    auto current = worklist.pop_back_val();
    if ( ! visited.insert(current).second ) {
      continue;
    }

    if ( allocRootFor.contains(current) || mlir::isa<mlir::BlockArgument>(current) ) {
      bases.push_back(current);
      continue;
    }

    if ( mlir::Operation *def = current.getDefiningOp() ) {
      if ( OpTypeHelper<ViewLike>::anyOf(def) ) {
        // TODO: Verify that view ops always have the base memref as first operand
        worklist.push_back(def->getOperand(0));
        continue;
      }
    }

    // Fallback: Treat as a base if no other method of determining base
    bases.push_back(current);
  }
}

struct AccessProvenance {
  mlir::Operation *accessOp;
  llvm::SmallVector<mlir::Operation *, 4> roots;
};


bool isGlobalStaticAllocationOp(mlir::Operation *op) {
  if (mlir::isa<mlir::SymbolOpInterface>(op)) {
    return OpTypeHelper<GlobalStaticAllocOps>::anyOf(op);
  }
  return false;
}

using namespace mlir::detail;

class DataRecomputationPass final
    : public mlir::impl::DataRecomputationPassBase<DataRecomputationPass> {

public:
  class PassContext {
  public:
    PassContext(mlir::MLIRContext *context,
                mlir::SymbolTableCollection &symTabCollection,
                mlir::ModuleOp moduleOp)
        : context{context}, symTabCollection{symTabCollection},
          moduleOp{moduleOp} {}

    mlir::MLIRContext *getContext() { return context; }

    mlir::SymbolTableCollection &getSymTabCollection() {
      return symTabCollection;
    }

    mlir::ModuleOp getModuleOp() { return moduleOp; }

  private:
    mlir::MLIRContext *context;
    std::reference_wrapper<mlir::SymbolTableCollection> symTabCollection;
    mlir::ModuleOp moduleOp;

  public:
    PassContext(const PassContext &) = default;
    PassContext(PassContext &&) = default;
    PassContext &operator=(const PassContext &) = default;
    PassContext &operator=(PassContext &&) = default;
  };

public:
  // Use the base class' constructor
  using DataRecomputationPassBase<
      DataRecomputationPass>::DataRecomputationPassBase;

  void runOnOperation() override;

private:
  void handleFunc(PassContext &ctx, mlir::func::FuncOp);

private:
  mlir::ModuleOp moduleOp;
  /// A map of functions in the module
  llvm::DenseMap<mlir::StringRef, FuncOp> functionMap;
  mlir::func::FuncOp entrypointFuncOp;
};
} // namespace

void DataRecomputationPass::runOnOperation() {
  /// Whether to output diagnostics for tests
  const bool outputDiagnostics = drTestDiagnostics.getValue();

  // Gather all global statics
  moduleOp = this->getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::SymbolTableCollection symTabCollection{};
  auto &diagnostics = context->getDiagEngine();

  PassContext passCtx{context, symTabCollection, moduleOp};

  llvm::DenseSet<mlir::Operation *> globalSymbolAllocations{};

  mlir::SymbolUserMap userMap(passCtx.getSymTabCollection(), moduleOp);

  // Collect global allocating ops on symbols
  llvm::for_each(moduleOp.getOps(), [&](mlir::Operation &op) {
    bool isGSA = isGlobalStaticAllocationOp(&op);
    bool allocatesOnSymbol = mlir::isa<mlir::SymbolOpInterface>(&op);
    if (isGSA && allocatesOnSymbol) {
      globalSymbolAllocations.insert(&op);
    }
  });

  // DenseMaps of Sets for:
  // MemLiveIn
  // MemLiveOut
  // MemKill
  //    - Offset?
  //    - OffsetExpr?

  // We only want to forward stuff. So just find all calls, take their
  // arguments, trace them back to their provenance

  llvm::SmallVector<mlir::func::FuncOp> funcOps{};

  // Collect all functions
  moduleOp.walk(
      [&funcOps](mlir::func::FuncOp funcOp) { funcOps.emplace_back(funcOp); });

  for (auto &funcOp : funcOps) {
    handleFunc(passCtx, funcOp);
  }
}

using PassContext = DataRecomputationPass::PassContext;


void DataRecomputationPass::handleFunc(PassContext &ctx,
                                       mlir::func::FuncOp funcOp) {
  llvm::SmallVector<mlir::Operation *> writeOps{};
  llvm::SmallVector<mlir::Operation *> readOps{};
  llvm::DenseSet<mlir::Operation *> allocatingOps{};

  llvm::SmallVector<AccessProvenance, 16> writesWithProvenance;
  llvm::SmallVector<AccessProvenance, 16> readsWithProvenance;

  llvm::DenseMap<mlir::Value, mlir::Operation *> allocRootFor{};


  auto collectAllocRootsForOp = [&](mlir::Operation *op) {
    if (mlir::hasEffect<mlir::MemoryEffects::Allocate>(op)) {
      allocatingOps.insert(op);
      for (mlir::Value res : op->getResults()) {
        if (mlir::isa<mlir::MemRefType>(res.getType())) {
          allocRootFor.try_emplace(res, op);
        }
      }
    }

    // Also collect get_globals
    if (auto getGlobalOp = mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
      auto *symOp = ctx.getSymTabCollection().lookupNearestSymbolFrom(
          ctx.getModuleOp(), getGlobalOp.getNameAttr());

      for (mlir::Value res : op->getResults()) {
        if (mlir::isa<mlir::MemRefType>(res.getType())) {
          allocRootFor.try_emplace(res, symOp);
        }
      }
    }
  };

  // Collect allocation roots
  funcOp.walk(collectAllocRootsForOp);

  funcOp.walk([&](mlir::Operation *op) {

    /*
     * Explanation: Allocation roots found above used to trace back operands of
     * view-like ops to root allocations.
     */

    auto attachProvenance = [&](bool isWrite) {
      AccessProvenance ap{op, {}};

      auto memEffectIface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
      ::llvm::SmallVector< ::mlir::MemoryEffects::EffectInstance> effects{};
      memEffectIface.getEffects(effects);


      for ( auto &effect : effects ) {
        const bool isWriteEffect = mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect());
        const bool isReadEffect = mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());

        const bool selectedWrite = isWrite && isWriteEffect;
        const bool selectedRead = (!isWrite) && isReadEffect;
        const bool validSelection = selectedRead || selectedWrite;

        if ( ! validSelection ) {
          continue;
        }

        // Only handle memref types
        mlir::Value value = effect.getValue();
        if ( ! value || !mlir::isa<mlir::MemRefType>(value.getType()) ) {
          continue;
        }

        // Collect MemRef Base Value
        // Tie it to an alloc root *or* a BlockArgument
        llvm::SmallVector<mlir::Value> bases{};
        collectBaseMemrefs(value, allocRootFor, bases);

        for ( mlir::Value b : bases ) {
          if ( auto it = allocRootFor.find(b); it != allocRootFor.end()){
            ap.roots.push_back(it->second);
          }
        }

        if (!ap.roots.empty()) {
          if (isWrite)
            writesWithProvenance.push_back(std::move(ap));
          else
            readsWithProvenance.push_back(std::move(ap));
        }
      }
    }; // attachProvenance

    if ( mlir::hasEffect<mlir::MemoryEffects::Write>(op) ) {
      writeOps.emplace_back(op);
      attachProvenance(/*isWrite=*/true);
    }

    if ( mlir::hasEffect<mlir::MemoryEffects::Read>(op) ) {
      readOps.emplace_back(op);
      attachProvenance(/*isWrite=*/false);
    }
  });

  for ( auto &x : writesWithProvenance ) {
    DRDBG() << *x.accessOp << ", could stem from: \n";
    for ( auto &root : x.roots ) {
      DRDBG() << "\t" << *root << "\n";
    }
  }



}

namespace mlir {
std::unique_ptr<Pass> createDataRecomputationPass() {
  return std::make_unique<DataRecomputationPass>();
}
} // namespace mlir
