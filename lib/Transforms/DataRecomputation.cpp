//===- DataRecomputation.cpp - Handles recomputation candidates -----------===//
//
//===----------------------------------------------------------------------===//
//
// This file contains the DataRecomputation pass (pending a better name).
//===----------------------------------------------------------------------===//

#include "marco/Transforms/DataRecomputation.h"
#include "marco/Transforms/DataRecomputationIndexing.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorDialect.h.inc"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#define DRDBG() llvm::dbgs() << "DRCOMP: "

namespace mlir {
#define GEN_PASS_DEF_DATARECOMPUTATIONPASS
#include "marco/Transforms/Passes.h.inc"
} // namespace mlir

using namespace ::mlir::memref;
using namespace ::mlir::func;

namespace {

class DRPassContext {
public:
  DRPassContext(mlir::MLIRContext *context,
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
  DRPassContext(const DRPassContext &) = default;
  DRPassContext(DRPassContext &&) = default;
  DRPassContext &operator=(const DRPassContext &) = default;
  DRPassContext &operator=(DRPassContext &&) = default;
};

template <class... OpTys>
struct OpTypePack {};

template <class... OpTys>
struct OpTypeHelper {
  static bool anyOf(mlir::Operation *op) { return mlir::isa<OpTys...>(op); }
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
bool isGlobalStaticAllocationOp(mlir::Operation *op) {
  if (mlir::isa<mlir::SymbolOpInterface>(op)) {
    return OpTypeHelper<GlobalStaticAllocOps>::anyOf(op);
  }
  return false;
}

llvm::DenseSet<mlir::Operation *>
collectGlobalStaticAllocations(DRPassContext &passCtx) {
  llvm::DenseSet<mlir::Operation *> result{};
  mlir::ModuleOp moduleOp = passCtx.getModuleOp();
  llvm::for_each(moduleOp.getOps(), [&](mlir::Operation &op) {
    bool isGSA = isGlobalStaticAllocationOp(&op);
    bool allocatesOnSymbol = mlir::isa<mlir::SymbolOpInterface>(&op);
    if (isGSA && allocatesOnSymbol) {
      result.insert(&op);
    }
  });

  return result;
}

/// Maps each value that stems from an allocation to the allocating operation.
using AllocationRoots = llvm::DenseMap<mlir::Value, mlir::Operation *>;

/// Walk \p rootOp and build a map from each allocated memref Value to the
/// operation that created it (alloc ops and memref.get_global).
static AllocationRoots collectAllocationRoots(DRPassContext &passCtx,
                                               mlir::Operation *rootOp) {
  AllocationRoots allocRootFor;

  rootOp->walk([&](mlir::Operation *op){
    if (mlir::hasEffect<mlir::MemoryEffects::Allocate>(op)) {
      for (mlir::Value res : op->getResults()) {
        if (mlir::isa<mlir::MemRefType>(res.getType())) {
          allocRootFor.try_emplace(res, op);
        }
      }
    }

    // Also collect get_globals
    if (auto getGlobalOp = mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
      auto *symOp = passCtx.getSymTabCollection().lookupNearestSymbolFrom(
        passCtx.getModuleOp(), getGlobalOp.getNameAttr());

      for (mlir::Value res : op->getResults()) {
        if (mlir::isa<mlir::MemRefType>(res.getType())) {
          allocRootFor.try_emplace(res, symOp);
        }
      }
    }
  });

  return allocRootFor;
}

//< Take a value and trace it back via ViewLike Ops to find base memrefs.
//< Base Memref = The closest to the alloc result as possible.
//
// A dense map is used to memoize traces for earlier termination.
//< Example
/*
```mlir
%base = memref.alloc ...
%vv = memref.subview %base ...
%vv2 = memref.subview %vv
// vv2 will yield %base
```
*/
void collectBaseMemrefs(
  mlir::Value v, llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
  llvm::SmallVectorImpl<mlir::Value> &bases) {

  llvm::SmallVector<mlir::Value, 4> worklist{v};
  llvm::SmallDenseSet<mlir::Value> visited{};

  while (!worklist.empty()) {
    auto current = worklist.pop_back_val();
    if (!visited.insert(current).second) {
      continue;
    }

    // Base memref found
    if (allocRootFor.contains(current) ||
      mlir::isa<mlir::BlockArgument>(current)) {
      bases.push_back(current);
      continue;
    }

    // If the current value has a defining operation, and the operation
    // is a view-taking operation, we get the first operand and add it to the
    // worklist.
    if (mlir::Operation *def = current.getDefiningOp()) {
      if (OpTypeHelper<ViewLike>::anyOf(def)) {
        // TODO: Verify that view ops always have the base memref as first
        // operand
        worklist.push_back(def->getOperand(0));
        continue;
      }
    }

    // Fallback: Treat as a base if no other method of determining base
    bases.push_back(current);
  }
}

/// Maps each store op to the allocation roots its stored value depends on
/// (via loads in the SSA operand chain).
using StoreValueDeps =
    llvm::DenseMap<mlir::Operation *,
                   llvm::SmallDenseSet<mlir::Operation *, 4>>;

/// Trace each store's value operand through SSA to find all allocation roots
/// the value depends on via memory loads.
static StoreValueDeps computeStoreValueDeps(
    mlir::Operation *rootOp,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor) {
  StoreValueDeps result;

  rootOp->walk([&](mlir::memref::StoreOp storeOp) {
    llvm::SmallDenseSet<mlir::Operation *, 4> deps;
    llvm::SmallVector<mlir::Value, 8> worklist;
    llvm::SmallDenseSet<mlir::Value> visited;

    worklist.push_back(storeOp.getValueToStore());

    while (!worklist.empty()) {
      mlir::Value current = worklist.pop_back_val();
      if (!visited.insert(current).second)
        continue;

      // BlockArgument cases
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
        if (mlir::isa<mlir::MemRefType>(blockArg.getType())) {
          llvm::SmallVector<mlir::Value, 2> bases;
          collectBaseMemrefs(blockArg, allocRootFor, bases);
          for (mlir::Value base : bases) {
            auto it = allocRootFor.find(base);
            if (it != allocRootFor.end())
              deps.insert(it->second);
          }
        }
        continue;  // stop recursing for any block argument
      }

      mlir::Operation *defOp = current.getDefiningOp();
      if (!defOp)
        continue;

      // LoadOp: record the load's memref root as a dependency
      if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(defOp)) {
        llvm::SmallVector<mlir::Value, 2> bases;
        collectBaseMemrefs(loadOp.getMemRef(), allocRootFor, bases);
        for (mlir::Value base : bases) {
          auto it = allocRootFor.find(base);
          if (it != allocRootFor.end())
            deps.insert(it->second);
        }
        continue;
      }

      // CallOpInterface: conservatively depend on all memref-typed operands
      if (mlir::isa<mlir::CallOpInterface>(defOp)) {
        for (mlir::Value operand : defOp->getOperands()) {
          if (!mlir::isa<mlir::MemRefType>(operand.getType()))
            continue;
          llvm::SmallVector<mlir::Value, 2> bases;
          collectBaseMemrefs(operand, allocRootFor, bases);
          for (mlir::Value base : bases) {
            auto it = allocRootFor.find(base);
            if (it != allocRootFor.end())
              deps.insert(it->second);
          }
        }
        continue;
      }

      // No operands (constants, etc.): stop
      if (defOp->getNumOperands() == 0)
        continue;

      // Otherwise: recurse into all operands
      for (mlir::Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }

    if (!deps.empty())
      result[storeOp.getOperation()] = std::move(deps);
  });

  return result;
}

// ===== Reaching Stores Analysis (Custom Region-Walking) =====

/// Set of memref.store operations (plus null for external writes) that may
/// have been the most recent write to a given allocation root.
using StoreSet = llvm::SmallDenseSet<mlir::Operation *, 4>;

/// An entry in the reaching-stores analysis: a store op plus offset coverage.
struct IndexedStore {
  mlir::Operation *storeOp;  // nullptr = external write
  /// Concrete indices this store must-write-to.
  /// nullopt = may-write-anywhere (dynamic indices, through views, rank-0).
  std::optional<PointSet> coverage;
};

/// Vector of indexed store entries for one allocation root.
using IndexedStoreVec = llvm::SmallVector<IndexedStore, 4>;

/// Maps each allocation root (the op that created the memory) to its
/// reaching indexed stores.
using StoreMap = llvm::DenseMap<mlir::Operation *, IndexedStoreVec>;

/// Maps each load op to the set of stores that may have been the last writer.
using LoadProvenanceMap = llvm::DenseMap<mlir::Operation *, StoreSet>;

/// Stores reaching one argument position at a call site.
using ArgStoreSet = StoreSet;

/// One directed call edge annotated with per-argument reaching stores.
struct EnrichedCallEdge {
  mlir::Operation *callSiteOp;
  mlir::FunctionOpInterface callee;
  llvm::SmallVector<ArgStoreSet> argStores;
};

/// Maps callee Operation* -> all incoming enriched call edges.
using EnrichedCallGraph =
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<EnrichedCallEdge>>;

/// Bundles all analysis state threaded through analyzeBlock/analyzeOp.
struct AnalysisContext {
  DRPassContext &passCtx;
  llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor;
  LoadProvenanceMap &loadProv;
  EnrichedCallGraph &callGraph;
  llvm::DenseSet<mlir::Operation *> inProgress;
  const StoreValueDeps &storeValueDeps;
  const llvm::DenseSet<mlir::Operation *> &globalAllocOps;
};

/// Union-join two StoreMaps. Returns a new map containing,
/// for each root, the merged entries from both maps.
/// If the same store op appears in both maps, their coverages are unioned.
StoreMap joinStoreMaps(StoreMap result, const StoreMap &other) {
  for (auto &[root, entries] : other) {
    auto &resultEntries = result[root];
    for (const auto &entry : entries) {
      // Deduplicate: if the same store op exists, union coverages.
      auto it = llvm::find_if(resultEntries, [&](const IndexedStore &e) {
        return e.storeOp == entry.storeOp;
      });
      if (it != resultEntries.end()) {
        if (!it->coverage || !entry.coverage)
          it->coverage = std::nullopt;
        else
          *it->coverage += *entry.coverage;
      } else {
        resultEntries.push_back(entry);
      }
    }
  }
  return result;
}

/// Remove from state any store whose value depends on clobberedRoot.
static void killDependentStores(
    mlir::Operation *clobberedRoot,
    StoreMap &state,
    const StoreValueDeps &storeValueDeps) {
  for (auto &[root, entries] : state) {
    llvm::erase_if(entries, [&](const IndexedStore &entry) {
      if (!entry.storeOp) return false;  // keep nullptr sentinel
      auto it = storeValueDeps.find(entry.storeOp);
      return it != storeValueDeps.end() && it->second.contains(clobberedRoot);
    });
  }
}

/// Try to extract concrete constant indices from a store/load's index operands.
/// Returns a PointSet containing the single accessed point, or nullopt if
/// any index is non-constant.
static std::optional<PointSet> computeAccessIndices(
    mlir::Operation::operand_range indices) {
  llvm::SmallVector<int64_t> coords;
  for (mlir::Value idx : indices) {
    auto constOp = idx.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constOp) return std::nullopt;
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue());
    if (!intAttr) return std::nullopt;
    coords.push_back(intAttr.getInt());
  }
  return PointSet::fromCoords(coords);
}

/// Update state to reflect a memref.store.
/// Rank-0 stores (no indices) kill all prior stores for that root.
/// Indexed stores with constant indices subtract their coverage from prior
/// entries and remove fully-killed entries. Dynamic-index stores are added
/// conservatively (nullopt coverage, cannot kill prior stores).
void applyStore(
    mlir::memref::StoreOp storeOp, StoreMap &state,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
    const StoreValueDeps &storeValueDeps) {
  llvm::SmallVector<mlir::Value, 2> bases;
  collectBaseMemrefs(storeOp.getMemRef(), allocRootFor, bases);
  for (mlir::Value base : bases) {
    auto it = allocRootFor.find(base);
    if (it == allocRootFor.end()) continue;

    // Kill stores whose values depended on the memory being overwritten
    killDependentStores(it->second, state, storeValueDeps);

    if (storeOp.getIndices().empty()) {
      // Rank-0 store: kill all prior entries and replace.
      state[it->second] = {{storeOp.getOperation(), std::nullopt}};
    } else {
      // Determine coverage: concrete if all indices are constant and the
      // store is not through a ViewLike op; nullopt otherwise.
      bool throughView = (base != storeOp.getMemRef());
      auto coverage = throughView
                          ? std::nullopt
                          : computeAccessIndices(storeOp.getIndices());

      if (coverage) {
        // Concrete coverage: subtract from prior entries, remove dead ones.
        llvm::erase_if(state[it->second], [&](IndexedStore &entry) {
          if (!entry.coverage) return false;  // can't subtract from universal
          *entry.coverage -= *coverage;
          return entry.coverage->empty();
        });
      }
      state[it->second].push_back({storeOp.getOperation(), coverage});
    }
  }
}

/// Analyze how a defined callee uses a specific memref argument.
/// Returns whether the callee may write through that argument, whether it
/// passes the argument to further calls, and the direct store operations.
struct CalleeArgEffect {
  bool mayWrite = false;
  bool passedToCall = false;
  llvm::SmallVector<mlir::Operation *, 4> storeOps;
};

static CalleeArgEffect analyzeCalleeArg(
    mlir::FunctionOpInterface callee, unsigned argIdx) {
  CalleeArgEffect effect;
  mlir::Region *body = callee.getCallableRegion();
  if (!body || body->empty()) {
    // External callee: conservatively assume writes.
    effect.mayWrite = true;
    effect.passedToCall = true;
    return effect;
  }

  if (argIdx >= body->getNumArguments()) {
    // Out of bounds (e.g., variadic): conservative.
    effect.mayWrite = true;
    effect.passedToCall = true;
    return effect;
  }

  mlir::BlockArgument arg = body->getArgument(argIdx);
  if (!mlir::isa<mlir::MemRefType>(arg.getType()))
    return effect;

  // Collect all values that are views/casts of this argument.
  llvm::SmallDenseSet<mlir::Value> argValues;
  llvm::SmallVector<mlir::Value> worklist;
  argValues.insert(arg);
  worklist.push_back(arg);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    for (mlir::OpOperand &use : current.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (OpTypeHelper<ViewLike>::anyOf(user)) {
        for (mlir::Value result : user->getResults()) {
          if (mlir::isa<mlir::MemRefType>(result.getType())) {
            if (argValues.insert(result).second)
              worklist.push_back(result);
          }
        }
      }
    }
  }

  // Check uses for stores and calls.
  for (mlir::Value v : argValues) {
    for (mlir::OpOperand &use : v.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(user)) {
        if (argValues.contains(storeOp.getMemRef())) {
          effect.mayWrite = true;
          effect.storeOps.push_back(storeOp.getOperation());
        }
      }
      if (mlir::isa<mlir::CallOpInterface>(user))
        effect.passedToCall = true;
    }
  }

  return effect;
}

/// Check if a callee function may access a given global variable.
/// External callees cannot access private globals.
/// Defined callees are scanned for direct memref.get_global usage.
static bool calleeMayAccessGlobal(
    mlir::FunctionOpInterface callee,
    mlir::Operation *globalOp) {
  auto globalSymbol = mlir::dyn_cast<mlir::SymbolOpInterface>(globalOp);
  if (!globalSymbol)
    return true;

  bool isPrivate =
      globalSymbol.getVisibility() == mlir::SymbolTable::Visibility::Private;

  mlir::Region *body = callee.getCallableRegion();
  if (!body || body->empty()) {
    // External callee: cannot access private globals.
    return !isPrivate;
  }

  // Defined callee: check for direct memref.get_global usage.
  llvm::StringRef globalName = globalSymbol.getName();
  bool directAccess = false;
  bool hasInnerCalls = false;

  body->walk([&](mlir::Operation *op) {
    if (auto getGlobal = mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
      if (getGlobal.getName() == globalName)
        directAccess = true;
    }
    if (mlir::isa<mlir::CallOpInterface>(op))
      hasInnerCalls = true;
  });

  if (directAccess) return true;
  if (!hasInnerCalls) return false;

  // Has inner calls but no direct access: conservatively assume
  // inner calls could access the global.
  return true;
}

/// Mark memref arguments of a call as clobbered, with refinements for
/// defined callees (read-only args skip clobbering, direct writes propagate
/// actual store ops) and global visibility (private globals unreachable by
/// external callees).
void applyCall(
    mlir::Operation *callOp, StoreMap &state,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
    const StoreValueDeps &storeValueDeps,
    const llvm::DenseSet<mlir::Operation *> &globalAllocOps,
    mlir::FunctionOpInterface callee) {
  // Track which roots we've already clobbered to avoid redundant work.
  llvm::SmallDenseSet<mlir::Operation *, 8> clobbered;

  // Handle explicit memref arguments.
  for (auto [i, operand] : llvm::enumerate(callOp->getOperands())) {
    if (!mlir::isa<mlir::MemRefType>(operand.getType())) continue;
    llvm::SmallVector<mlir::Value, 2> bases;
    collectBaseMemrefs(operand, allocRootFor, bases);
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end()) continue;

      if (callee) {
        CalleeArgEffect effect = analyzeCalleeArg(callee, i);
        if (!effect.mayWrite && !effect.passedToCall) {
          // Callee only reads this argument: no clobber needed.
          continue;
        }

        killDependentStores(it->second, state, storeValueDeps);

        if (effect.mayWrite && !effect.passedToCall) {
          // Direct writes only: propagate actual store ops.
          for (mlir::Operation *storeOp : effect.storeOps)
            state[it->second].push_back({storeOp, std::nullopt});
        } else {
          // Passed to further call or external: conservative.
          state[it->second].push_back({nullptr, std::nullopt});
        }
      } else {
        // Unresolved callee: conservative clobber.
        killDependentStores(it->second, state, storeValueDeps);
        state[it->second].push_back({nullptr, std::nullopt});
      }
      clobbered.insert(it->second);
    }
  }

  // Clobber globals that the callee may access.
  for (mlir::Operation *globalOp : globalAllocOps) {
    if (clobbered.contains(globalOp))
      continue;

    if (callee && !calleeMayAccessGlobal(callee, globalOp))
      continue;

    killDependentStores(globalOp, state, storeValueDeps);
    state[globalOp].push_back({nullptr, std::nullopt});
  }
}

// Forward declarations for the recursive analysis.
static void analyzeBlock(mlir::Block &block, StoreMap &state,
    AnalysisContext &ctx);

static void analyzeOp(mlir::Operation *op, StoreMap &state,
    AnalysisContext &ctx);

static void analyzeBlock(mlir::Block &block, StoreMap &state,
    AnalysisContext &ctx) {
  for (mlir::Operation &op : block)
    analyzeOp(&op, state, ctx);
}

static void analyzeOp(mlir::Operation *op, StoreMap &state,
    AnalysisContext &ctx) {
  auto &allocRootFor = ctx.allocRootFor;
  auto &loadProv = ctx.loadProv;

  // --- scf.if / affine.if: branch + join ---
  if (mlir::isa<mlir::scf::IfOp>(op) ||
      mlir::isa<mlir::affine::AffineIfOp>(op)) {
    // Then region (always present, region 0)
    StoreMap thenState = state;
    mlir::Region &thenRegion = op->getRegion(0);
    if (!thenRegion.empty())
      analyzeBlock(thenRegion.front(), thenState, ctx);

    // Else region (region 1, may be empty or absent)
    StoreMap elseState = state;
    if (op->getNumRegions() > 1) {
      mlir::Region &elseRegion = op->getRegion(1);
      if (!elseRegion.empty())
        analyzeBlock(elseRegion.front(), elseState, ctx);
    }

    state = joinStoreMaps(thenState, elseState);
    return;
  }

  // --- scf.for / affine.for: may-execute loop ---
  if (mlir::isa<mlir::scf::ForOp>(op) ||
      mlir::isa<mlir::affine::AffineForOp>(op)) {
    StoreMap bodyState = state;
    mlir::Region &bodyRegion = op->getRegion(0);
    if (!bodyRegion.empty())
      analyzeBlock(bodyRegion.front(), bodyState, ctx);

    // For affine.for with provably-positive trip count (static lb < ub),
    // the body always executes at least once — use only the body state.
    if (auto affineFor = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
      if (affineFor.hasConstantLowerBound() &&
          affineFor.hasConstantUpperBound() &&
          affineFor.getConstantLowerBound() <
              affineFor.getConstantUpperBound()) {
        state = bodyState;
        return;
      }
    }

    // Loop may not execute: join body-result with pre-loop state
    state = joinStoreMaps(state, bodyState);
    return;
  }

  // --- scf.while: may-execute while loop ---
  if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
    // before-region (condition)
    StoreMap condState = state;
    mlir::Region &beforeRegion = whileOp.getBefore();
    if (!beforeRegion.empty())
      analyzeBlock(beforeRegion.front(), condState, ctx);

    // after-region (body)
    StoreMap bodyState = condState;
    mlir::Region &afterRegion = whileOp.getAfter();
    if (!afterRegion.empty())
      analyzeBlock(afterRegion.front(), bodyState, ctx);

    // Join with pre-loop (may not execute)
    state = joinStoreMaps(state, bodyState);
    return;
  }

  // --- memref.store ---
  if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
    applyStore(storeOp, state, allocRootFor, ctx.storeValueDeps);
    return;
  }

  // --- memref.load: record provenance (offset-sensitive) ---
  if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
    llvm::SmallVector<mlir::Value, 2> bases;
    collectBaseMemrefs(loadOp.getMemRef(), allocRootFor, bases);
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end()) continue;
      auto rootIt = state.find(it->second);
      if (rootIt == state.end()) continue;

      // Ensure the load appears in provenance even if no stores match
      // (e.g., all stores were killed). An empty provenance set is
      // classified as MULTI by the downstream classifier.
      auto &provSet = loadProv[op];

      bool throughView = (base != loadOp.getMemRef());
      auto loadCoverage = (loadOp.getIndices().empty() || throughView)
                              ? std::nullopt
                              : computeAccessIndices(loadOp.getIndices());

      for (const auto &entry : rootIt->second) {
        if (loadCoverage && entry.coverage) {
          // Both have concrete coverage: only include if they overlap.
          if (loadCoverage->overlaps(*entry.coverage))
            provSet.insert(entry.storeOp);
        } else {
          // Either is nullopt: conservative, always overlaps.
          provSet.insert(entry.storeOp);
        }
      }
    }
    return;
  }

  // --- CallOpInterface: record enriched edge, then conservative clobber ---
  if (auto callIface = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
    // Resolve callee
    mlir::FunctionOpInterface callee = nullptr;
    auto callableRef = callIface.getCallableForCallee();
    if (auto symbol = mlir::dyn_cast<mlir::SymbolRefAttr>(callableRef)) {
      auto *calleeOp = ctx.passCtx.getSymTabCollection().lookupSymbolIn(
          ctx.passCtx.getModuleOp().getOperation(), symbol);
      if (calleeOp)
        callee = mlir::dyn_cast<mlir::FunctionOpInterface>(calleeOp);
    }

    // Build enriched call edge
    EnrichedCallEdge edge;
    edge.callSiteOp = op;
    edge.callee = callee;
    edge.argStores.resize(op->getNumOperands());

    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      if (!mlir::isa<mlir::MemRefType>(operand.getType()))
        continue;

      llvm::SmallVector<mlir::Value, 2> bases;
      collectBaseMemrefs(operand, allocRootFor, bases);
      for (mlir::Value base : bases) {
        auto it = allocRootFor.find(base);
        if (it == allocRootFor.end()) continue;
        auto rootIt = state.find(it->second);
        if (rootIt != state.end())
          for (const auto &entry : rootIt->second)
            edge.argStores[i].insert(entry.storeOp);
      }
    }

    // Record edge (keyed by callee op, or null for indirect/external)
    mlir::Operation *calleeOp = callee ? callee.getOperation() : nullptr;
    ctx.callGraph[calleeOp].push_back(std::move(edge));

    // Apply clobber with callee-aware refinements
    applyCall(op, state, allocRootFor, ctx.storeValueDeps,
              ctx.globalAllocOps, callee);
    return;
  }

  // --- Generic op with regions (catch-all) ---
  if (op->getNumRegions() > 0) {
    // Conservatively iterate all regions sequentially and join results.
    StoreMap joined = state;
    bool first = true;
    for (mlir::Region &region : op->getRegions()) {
      if (region.empty()) continue;
      StoreMap regionState = state;
      analyzeBlock(region.front(), regionState, ctx);
      if (first) {
        joined = regionState;
        first = false;
      } else {
        joined = joinStoreMaps(joined, regionState);
      }
    }
    if (!first)
      state = joinStoreMaps(state, joined);
  }
}


// ===== DOT Graph Emission Helpers =====

/// Produce a stable unique node ID from an operation pointer.
static std::string dotNodeId(mlir::Operation *op) {
  return llvm::formatv("op_{0}", (const void *)op);
}

/// Escape a string for use inside a DOT label.
static std::string dotEscape(llvm::StringRef s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '"':  result += "\\\""; break;
    case '\\': result += "\\\\"; break;
    case '<':  result += "\\<";  break;
    case '>':  result += "\\>";  break;
    case '{':  result += "\\{";  break;
    case '}':  result += "\\}";  break;
    case '|':  result += "\\|";  break;
    case '\n': result += "\\n";  break;
    default:   result += c;      break;
    }
  }
  return result;
}

/// Compact operation label: print without regions, truncate to 120 chars.
static std::string dotOpLabel(mlir::Operation *op) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  mlir::OpPrintingFlags flags;
  flags.skipRegions();
  op->print(os, flags);
  os.flush();
  if (buf.size() > 120)
    buf = buf.substr(0, 117) + "...";
  return dotEscape(buf);
}

/// Produce a stable unique node ID for a block argument.
static std::string dotBlockArgId(mlir::BlockArgument arg) {
  return llvm::formatv("barg_{0}_{1}",
      (const void *)arg.getOwner(), arg.getArgNumber());
}

/// Emit the full SSA operand tree rooted at `rootVal` into the DOT graph.
/// Each defining operation becomes a rounded-box node, block arguments become
/// hexagon nodes, and dashed edges show the dataflow.
static void emitOperandTree(
    mlir::Value rootVal,
    llvm::raw_ostream &out,
    llvm::DenseSet<mlir::Operation *> &emittedNodes,
    llvm::DenseSet<mlir::Value> &emittedBlockArgs,
    llvm::DenseSet<mlir::Operation *> &processedTreeOps) {
  llvm::SmallVector<mlir::Value, 8> worklist;
  worklist.push_back(rootVal);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      if (emittedBlockArgs.insert(current).second) {
        std::string id = dotBlockArgId(blockArg);
        std::string typeBuf;
        llvm::raw_string_ostream tos(typeBuf);
        blockArg.getType().print(tos);
        tos.flush();
        std::string label = llvm::formatv("arg#{0} : {1}",
            blockArg.getArgNumber(), typeBuf);
        out << "  \"" << id
            << "\" [shape=hexagon, style=filled, fillcolor=\"#f0e6ff\", label=\""
            << dotEscape(label) << "\"];\n";
      }
      continue;
    }

    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp) continue;
    if (!processedTreeOps.insert(defOp).second) continue;

    // Emit node only if not already present (as a store or load node).
    if (emittedNodes.insert(defOp).second) {
      out << "  \"" << dotNodeId(defOp)
          << "\" [shape=box, style=\"filled,rounded\", fillcolor=\"#f5f5f5\", label=\""
          << dotOpLabel(defOp) << "\"];\n";
    }

    for (mlir::Value operand : defOp->getOperands()) {
      std::string fromId;
      if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(operand))
        fromId = dotBlockArgId(barg);
      else if (auto *opDef = operand.getDefiningOp())
        fromId = dotNodeId(opDef);
      if (!fromId.empty()) {
        out << "  \"" << fromId << "\" -> \"" << dotNodeId(defOp)
            << "\" [style=dashed, color=\"#888888\"];\n";
      }
      worklist.push_back(operand);
    }
  }
}

/// Emit a GraphViz DOT file visualizing load-provenance analysis results.
static void emitProvenanceDot(
    llvm::StringRef filePath,
    llvm::SmallVectorImpl<std::pair<mlir::Operation *, StoreSet *>> &singleLoads,
    llvm::SmallVectorImpl<std::pair<mlir::Operation *, StoreSet *>> &multiLoads,
    llvm::SmallVectorImpl<std::pair<mlir::Operation *, StoreSet *>> &leakedLoads,
    llvm::SmallVectorImpl<std::pair<mlir::Operation *, StoreSet *>> &killedLoads) {
  std::error_code ec;
  llvm::raw_fd_ostream out(filePath, ec, llvm::sys::fs::OF_TextWithCRLF);
  if (ec) {
    llvm::errs() << "error: cannot open DOT file '" << filePath
                 << "': " << ec.message() << "\n";
    return;
  }

  out << "digraph load_provenance {\n";
  out << "  rankdir=LR;\n";
  out << "  node [fontname=\"Courier\", fontsize=10];\n";
  out << "  edge [fontname=\"Courier\", fontsize=9];\n\n";

  // Track emitted nodes to avoid duplicate/conflicting node definitions.
  llvm::DenseSet<mlir::Operation *> emittedNodes;
  llvm::SmallVector<mlir::Operation *> allStoreOps;
  bool needsExternalNode = false;

  // Helper: emit a store node if not already emitted.
  auto emitStoreNode = [&](mlir::Operation *storeOp) {
    if (!storeOp) {
      needsExternalNode = true;
      return;
    }
    if (!emittedNodes.insert(storeOp).second)
      return;
    allStoreOps.push_back(storeOp);
    out << "  \"" << dotNodeId(storeOp)
        << "\" [shape=ellipse, style=filled, fillcolor=\"#d4edda\", label=\""
        << dotOpLabel(storeOp) << "\"];\n";
  };

  // --- SINGLE loads: store --> load ---
  out << "  // === SINGLE loads ===\n";
  for (auto &[loadOp, stores] : singleLoads) {
    emittedNodes.insert(loadOp);
    out << "  \"" << dotNodeId(loadOp)
        << "\" [shape=box, style=filled, fillcolor=\"#cce5ff\", label=\""
        << dotOpLabel(loadOp) << "\"];\n";
    for (mlir::Operation *s : *stores) {
      emitStoreNode(s);
      if (s) {
        out << "  \"" << dotNodeId(s) << "\" -> \"" << dotNodeId(loadOp)
            << "\";\n";
      }
    }
  }

  // --- MULTI loads: stores --> diamond --> load ---
  out << "\n  // === MULTI loads ===\n";
  unsigned mergeId = 0;
  for (auto &[loadOp, stores] : multiLoads) {
    emittedNodes.insert(loadOp);
    out << "  \"" << dotNodeId(loadOp)
        << "\" [shape=box, style=filled, fillcolor=\"#cce5ff\", label=\""
        << dotOpLabel(loadOp) << "\"];\n";

    std::string diamondId = llvm::formatv("merge_{0}", mergeId++);
    out << "  \"" << diamondId
        << "\" [shape=diamond, style=filled, fillcolor=\"#fff3cd\", label=\"merge\"];\n";
    out << "  \"" << diamondId << "\" -> \"" << dotNodeId(loadOp) << "\";\n";

    for (mlir::Operation *s : *stores) {
      emitStoreNode(s);
      if (s) {
        out << "  \"" << dotNodeId(s) << "\" -> \"" << diamondId << "\";\n";
      }
    }
  }

  // --- LEAKED loads: stores --> diamond --> load (red) ---
  out << "\n  // === LEAKED loads ===\n";
  for (auto &[loadOp, stores] : leakedLoads) {
    emittedNodes.insert(loadOp);
    out << "  \"" << dotNodeId(loadOp)
        << "\" [shape=box, style=filled, fillcolor=\"#f8d7da\", "
           "color=red, penwidth=2, label=\""
        << dotOpLabel(loadOp) << "\"];\n";

    std::string diamondId = llvm::formatv("merge_{0}", mergeId++);
    out << "  \"" << diamondId
        << "\" [shape=diamond, style=filled, fillcolor=\"#fff3cd\", label=\"merge\"];\n";
    out << "  \"" << diamondId << "\" -> \"" << dotNodeId(loadOp) << "\";\n";

    for (mlir::Operation *s : *stores) {
      if (!s) {
        needsExternalNode = true;
        out << "  external_write -> \"" << diamondId << "\";\n";
      } else {
        emitStoreNode(s);
        out << "  \"" << dotNodeId(s) << "\" -> \"" << diamondId << "\";\n";
      }
    }
  }

  // --- KILLED loads: all reaching stores were killed ---
  out << "\n  // === KILLED loads ===\n";
  for (auto &[loadOp, stores] : killedLoads) {
    emittedNodes.insert(loadOp);
    out << "  \"" << dotNodeId(loadOp)
        << "\" [shape=box, style=filled, fillcolor=\"#e0e0e0\", "
           "color=\"#666666\", penwidth=2, label=\""
        << dotOpLabel(loadOp) << "\"];\n";
  }

  // Emit the shared external write node if needed.
  if (needsExternalNode) {
    out << "\n  external_write [shape=octagon, style=filled, "
           "fillcolor=\"#e2e3e5\", label=\"<external>\"];\n";
  }

  // --- Operand trees: full SSA computation chains for each store ---
  out << "\n  // === Operand Trees ===\n";
  {
    llvm::DenseSet<mlir::Operation *> processedTreeOps;
    llvm::DenseSet<mlir::Value> emittedBlockArgs;
    for (mlir::Operation *storeOp : allStoreOps) {
      auto memStoreOp = mlir::dyn_cast<mlir::memref::StoreOp>(storeOp);
      if (!memStoreOp) continue;
      mlir::Value val = memStoreOp.getValueToStore();
      std::string fromId;
      if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(val))
        fromId = dotBlockArgId(barg);
      else if (auto *opDef = val.getDefiningOp())
        fromId = dotNodeId(opDef);
      if (!fromId.empty()) {
        out << "  \"" << fromId << "\" -> \"" << dotNodeId(storeOp)
            << "\" [style=dashed, color=\"#888888\", label=\"val\"];\n";
      }
      emitOperandTree(val, out, emittedNodes, emittedBlockArgs,
                       processedTreeOps);
    }
  }

  // Legend subgraph.
  out << R"(
  subgraph cluster_legend {
    label="Legend";
    style=dashed;
    fontsize=12;
    legend_store [shape=ellipse, style=filled, fillcolor="#d4edda", label="store"];
    legend_load [shape=box, style=filled, fillcolor="#cce5ff", label="load"];
    legend_leaked [shape=box, style=filled, fillcolor="#f8d7da", color=red, penwidth=2, label="leaked load"];
    legend_killed [shape=box, style=filled, fillcolor="#e0e0e0", color="#666666", penwidth=2, label="killed load"];
    legend_merge [shape=diamond, style=filled, fillcolor="#fff3cd", label="merge"];
    legend_ext [shape=octagon, style=filled, fillcolor="#e2e3e5", label="<external>"];
    legend_compute [shape=box, style="filled,rounded", fillcolor="#f5f5f5", label="computation"];
    legend_arg [shape=hexagon, style=filled, fillcolor="#f0e6ff", label="block arg"];
  }
)";

  out << "}\n";
  DRDBG() << "Wrote load-provenance DOT graph to " << filePath << "\n";
}

using namespace mlir::detail;

class DataRecomputationPass final
    : public mlir::impl::DataRecomputationPassBase<DataRecomputationPass> {
public:
  using DataRecomputationPassBase<
      DataRecomputationPass>::DataRecomputationPassBase;

  void runOnOperation() override;

private:
  mlir::ModuleOp moduleOp;
};



// Simple Escape Analysis
llvm::DenseSet<mlir::FunctionOpInterface> getPotentialEscapingFunctions(DRPassContext &ctx)
{
  llvm::DenseSet<mlir::FunctionOpInterface> result{};

  mlir::ModuleOp moduleOp = ctx.getModuleOp();

  auto &symTabCollection = ctx.getSymTabCollection();

  // Find calls to functions
  moduleOp.walk([&](mlir::Operation *op) {
    if ( auto funcOpIface = mlir::dyn_cast<mlir::CallOpInterface>(op) ) {

      // Either Value or SymbolRefAttr
      auto callee = funcOpIface.getCallableForCallee();
      auto symbol = callee.dyn_cast<mlir::SymbolRefAttr>();

      if ( !symbol ) {
        return mlir::WalkResult::skip();
      }

      auto *calleeOp = symTabCollection.lookupSymbolIn(moduleOp.getOperation(), symbol);
      auto funcIface = mlir::dyn_cast<mlir::FunctionOpInterface>(calleeOp);

      if ( ! funcIface ) {
        return mlir::WalkResult::skip();
      }

      // If not private OR is external
      const bool external = funcIface.isExternal();
      const bool externallyVisible = funcIface.isPublic();

      const bool takesPtrLike = llvm::any_of(funcIface.getArgumentTypes(), [](mlir::Type t) {
        bool isPointerLike = mlir::isa<mlir::PtrLikeTypeInterface>(t);
        return isPointerLike;
      });

      if ( takesPtrLike && ( external || externallyVisible )  ) {
         result.insert(funcIface);
      }
    }

    return mlir::WalkResult::advance();
  });

  for ( auto &x : result ) {
    DRDBG() << "Leaky function: " << x.getName() << "\n";
  }

  return result;
}

/// Check if a value's SSA operand tree contains any function entry arguments.
static bool dependsOnFuncArg(mlir::Value val) {
  llvm::SmallVector<mlir::Value, 8> worklist;
  llvm::SmallDenseSet<mlir::Value> visited;
  worklist.push_back(val);
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      mlir::Block *block = blockArg.getOwner();
      if (block && block->isEntryBlock())
        if (mlir::isa_and_nonnull<mlir::FunctionOpInterface>(
                block->getParentOp()))
          return true;
      continue;
    }
    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;
    for (mlir::Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
  return false;
}

/// Recursively print the SSA operand tree of a value for debugging.
static void printOperandTree(mlir::Value val, unsigned indent = 0) {
  std::string prefix(indent * 2, ' ');
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    DRDBG() << prefix << "block-arg #" << blockArg.getArgNumber()
            << " : " << blockArg.getType() << "\n";
    return;
  }
  mlir::Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    DRDBG() << prefix << "<unknown value>\n";
    return;
  }
  DRDBG() << prefix << *defOp << "\n";
  for (mlir::Value operand : defOp->getOperands())
    printOperandTree(operand, indent + 1);
}

} // namespace

void DataRecomputationPass::runOnOperation() {
  moduleOp = this->getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::SymbolTableCollection symTabCollection{};

  DRPassContext passCtx{context, symTabCollection, moduleOp};

  auto globalAllocs = collectGlobalStaticAllocations(passCtx);

  // Collect allocation roots for the whole module so the analysis can resolve
  // memref bases that are allocated in one function and passed to another.
  auto allocRootFor = collectAllocationRoots(passCtx, moduleOp);

  // Log functions that may let memory escape to external callers.
  for (auto &f : getPotentialEscapingFunctions(passCtx))
    DRDBG() << "Potential escaper: " << *f.getOperation() << "\n";

  StoreValueDeps storeValueDeps = computeStoreValueDeps(moduleOp, allocRootFor);

  LoadProvenanceMap loadProv;
  EnrichedCallGraph callGraph;
  AnalysisContext ctx{passCtx, allocRootFor, loadProv, callGraph, {}, storeValueDeps, globalAllocs};

  // Analyze each top-level function
  moduleOp.walk([&](mlir::FunctionOpInterface funcOp) {
    mlir::Region *body = funcOp.getCallableRegion();
    if (!body || body->empty()) return;
    StoreMap state;
    analyzeBlock(body->front(), state, ctx);
  });

  // Classify loads into SINGLE, MULTI, LEAKED, and KILLED categories.
  // LEAKED  = provenance contains nullptr (external/unknown write).
  // KILLED  = all reaching stores were killed (empty provenance set).
  // SINGLE  = exactly one non-null provenance store (and not leaked).
  // MULTI   = multiple provenance stores (and not leaked).
  llvm::SmallVector<std::pair<mlir::Operation *, StoreSet *>> singleLoads;
  llvm::SmallVector<std::pair<mlir::Operation *, StoreSet *>> multiLoads;
  llvm::SmallVector<std::pair<mlir::Operation *, StoreSet *>> leakedLoads;
  llvm::SmallVector<std::pair<mlir::Operation *, StoreSet *>> killedLoads;

  for (auto &[loadOp, stores] : loadProv) {
    if (stores.contains(nullptr)) {
      leakedLoads.push_back({loadOp, &stores});
    } else if (stores.empty()) {
      killedLoads.push_back({loadOp, &stores});
    } else if (stores.size() == 1) {
      singleLoads.push_back({loadOp, &stores});
    } else {
      multiLoads.push_back({loadOp, &stores});
    }
  }

  // Emit MLIR remarks for test diagnostics (used with -verify-diagnostics).
  if (drTestDiagnostics) {
    for (auto &[loadOp, stores] : singleLoads)
      loadOp->emitRemark("load: SINGLE");
    for (auto &[loadOp, stores] : multiLoads)
      loadOp->emitRemark("load: MULTI");
    for (auto &[loadOp, stores] : leakedLoads)
      loadOp->emitRemark("load: LEAKED");
    for (auto &[loadOp, stores] : killedLoads)
      loadOp->emitRemark("load: KILLED");
  }

  // Helper to print a provenance store with ARGUMENT flag and operand tree.
  auto printProvenance = [](mlir::Operation *s) {
    if (!s) {
      DRDBG() << "  Provenance: <external write>\n";
      return;
    }
    if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(s)) {
      bool fromArg = dependsOnFuncArg(storeOp.getValueToStore());
      DRDBG() << "  Provenance store: " << *s
              << (fromArg ? " (ARGUMENT)" : "") << "\n";
      DRDBG() << "  Operand tree:\n";
      printOperandTree(storeOp.getValueToStore(), 2);
    } else {
      DRDBG() << "  Provenance store: " << *s << "\n";
    }
  };

  DRDBG() << "=== SINGLE (" << singleLoads.size() << ") ===\n";
  for (auto &[loadOp, stores] : singleLoads) {
    DRDBG() << "Load: " << *loadOp << "\n";
    printProvenance(*stores->begin());
  }

  DRDBG() << "=== MULTI (" << multiLoads.size() << ") ===\n";
  for (auto &[loadOp, stores] : multiLoads) {
    DRDBG() << "Load: " << *loadOp << "\n";
    for (mlir::Operation *s : *stores)
      printProvenance(s);
  }

  DRDBG() << "=== LEAKED (" << leakedLoads.size() << ") ===\n";
  for (auto &[loadOp, stores] : leakedLoads) {
    DRDBG() << "Load: " << *loadOp << "\n";
    for (mlir::Operation *s : *stores)
      printProvenance(s);
  }

  DRDBG() << "=== KILLED (" << killedLoads.size() << ") ===\n";
  for (auto &[loadOp, stores] : killedLoads) {
    DRDBG() << "Load: " << *loadOp << "\n";
  }

  if (!drDotFile.empty())
    emitProvenanceDot(drDotFile, singleLoads, multiLoads, leakedLoads,
                      killedLoads);

  // Log store value dependencies
  for (auto &[storeOp, deps] : storeValueDeps) {
    DRDBG() << "Store value deps: " << *storeOp << "\n";
    for (mlir::Operation *root : deps)
      DRDBG() << "  depends on alloc root: " << *root << "\n";
  }

  // Log enriched call graph edges
  for (auto &[calleeOp, edges] : callGraph) {
    if (calleeOp)
      DRDBG() << "Callee: " << calleeOp->getName() << "\n";
    else
      DRDBG() << "Callee: <indirect/external>\n";
    for (auto &edge : edges) {
      DRDBG() << "  Call site: " << *edge.callSiteOp << "\n";
      for (auto [i, stores] : llvm::enumerate(edge.argStores)) {
        if (stores.empty()) continue;
        DRDBG() << "    arg[" << i << "] reached by " << stores.size()
                << " store(s)\n";
      }
    }
  }
}

namespace mlir {
std::unique_ptr<Pass> createDataRecomputationPass() {
  return std::make_unique<DataRecomputationPass>();
}
} // namespace mlir
