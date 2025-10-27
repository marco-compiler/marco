//===- DataRecomputation.cpp - Handles recomputation candidates -----------===//
//
//===----------------------------------------------------------------------===//
//
// This file contains the DataRecomputation pass (pending a better name).
//===----------------------------------------------------------------------===//

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/DataRecomputation.h"

#include "marco/Dialect/BaseModelica/Transforms/DataRecomputation/IndexExpression.h"
#include "marco/Modeling/GraphDumper.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Vector/IR/VectorDialect.h.inc"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_DATARECOMPUTATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::memref;
using namespace ::mlir::func;

//=== Rationale:
// The idea behind this pass is to keep track of stores and do a full
// module-wide interprocedural analysis to find loads from globals or other
// scoped allocations that can provably be shown to originate from another
// store, and that there are no interjecting writes to the same location that
// would invalidate it.
//
// Requirements:
// - Simple control flow with respect to the data-path of the loaded value.
// - Single-source store origin for each load.
// - Provable thread-independence at the point of a load.
//  - If other threads can interject and invalidate the stored value, the
//      recomputation will be invalid.
//
// Strategy:
// Walk backwards from stores, and find where their memrefs originate. If
// we can show that the memref is uniquely referenced and the control-flow
// between two write-read pairs is sufficiently simple, we replace the load
// with the computation that produced the preceding store.
//
// On sufficiently simple control flow:
//
// Cache analysis:
// The cache analysis can depend on a simple fully associative LRU sim.
// The likelihood estimate will depend on a code-distance measure in terms
// likened to basic-block distance with some estimate of arithmetic and memory
// intensity for each basic block.
//
// To begin with, the estimate will be highly pessimistic. We aim to refine it
// by considering contiguous or small-stride access patterns.

//=== On parametric R-trees
// This applies to the decision problem that answers whether there are potential
// interposing writes / clobbers to candidate operands for reuse.
//
// An idea to figure out whether clobbers are resolvable or undeterminably
// overlapping is to use parametric R-trees. See bibtex below.
// Spatial queries for overlaps and minimal bounding rectangle (MBR) are
// efficient.
//
// The parametric range delimiters for each dimension can potentially be
// partitioned by thread ID. A question that remains is figuring out the
// worst-case scenario that would guarantee or (for now) reduce the risk
// of miscompiling.

/*
@inproceedings{cai2000parametric,
  title={Parametric R-tree: An index structure for moving objects},
  author={Cai, Mengchu and Revesz, Peter},
  booktitle={Proc. of the COMAD},
  year={2000}
}
  */

#define MARCO_DBG() llvm::dbgs() << R"(==== DataRecomputation: )"

namespace {
template <class... OpTys>
static bool isAnyOp(mlir::Operation *op) {
  return (mlir::isa<OpTys>(op) || ...);
}

} // namespace

namespace mlir::bmodelica::detail {

using AccessSet = llvm::DenseSet<mlir::Operation *>;

// TODO: Rename this
struct GlobalDef {
  mlir::memref::GlobalOp definingOperation;
  llvm::StringRef name;
  AccessSet accesses;

  GlobalDef() = default;

  GlobalDef(mlir::memref::GlobalOp op)
      : definingOperation{op}, name{op.getName()} {}

  GlobalDef(const GlobalDef &other) = default;
  GlobalDef &operator=(const GlobalDef &other) = default;

  GlobalDef(GlobalDef &&other) = default;
  GlobalDef &operator=(GlobalDef &&other) = default;
};

/// DataRecomputation Write
struct DRWrite {
  using StoreVariant =
      std::variant<::mlir::memref::StoreOp, ::mlir::affine::AffineStoreOp,
                   ::mlir::ptr::StoreOp>;

  DRWrite(mlir::memref::StoreOp storeOp) noexcept : storeOp{storeOp} {}
  DRWrite(mlir::affine::AffineStoreOp storeOp) noexcept : storeOp{storeOp} {}
  DRWrite(mlir::ptr::StoreOp storeOp) noexcept : storeOp{storeOp} {}

  DRWrite(const DRWrite &) = default;
  DRWrite(DRWrite &&) = default;
  DRWrite &operator=(const DRWrite &) = default;
  DRWrite &operator=(DRWrite &&) = default;
  // Discern between memref write, affine write, tensor write, etc.
  StoreVariant storeOp;
  std::optional<mlir::Operation *> allocatingOperation;

  bool isMemref() const {
    return std::holds_alternative<mlir::memref::StoreOp>(storeOp);
  }

  bool isAffine() const {
    return std::holds_alternative<mlir::affine::AffineStoreOp>(storeOp);
  }

  bool isPtr() const {
    return std::holds_alternative<mlir::ptr::StoreOp>(storeOp);
  }

  /// Utility function to check if an mlir::Operation * fits as a DRWrite
  static bool isStoreOp(mlir::Operation *op) {
    return isAnyOp<mlir::memref::StoreOp, mlir::affine::AffineStoreOp,
                   mlir::ptr::StoreOp>(op);
  }
};

struct DRLoad {
  using LoadVariant =
      std::variant<::mlir::memref::LoadOp, ::mlir::affine::AffineLoadOp,
                   ::mlir::ptr::LoadOp>;

  DRLoad(mlir::memref::LoadOp loadOp) noexcept : loadOp{loadOp} {}
  DRLoad(mlir::affine::AffineLoadOp loadOp) noexcept : loadOp{loadOp} {}
  DRLoad(mlir::ptr::LoadOp loadOp) noexcept : loadOp{loadOp} {}

  DRLoad(const DRLoad &) = default;
  DRLoad(DRLoad &&) = default;
  DRLoad &operator=(const DRLoad &) = default;
  DRLoad &operator=(DRLoad &&) = default;

  LoadVariant loadOp;

  // Casting to Operation *
  operator mlir::Operation *() {
    return std::visit(
        [](auto &op) -> mlir::Operation * { return op.getOperation(); },
        loadOp);
  }

  bool isMemref() const {
    return std::holds_alternative<mlir::memref::LoadOp>(loadOp);
  }

  bool isAffine() const {
    return std::holds_alternative<mlir::affine::AffineLoadOp>(loadOp);
  }

  bool isPtr() const {
    return std::holds_alternative<mlir::ptr::LoadOp>(loadOp);
  }

  /// Utility function to check if an mlir::Operation * fits as a DRLoad
  static bool isLoadOp(mlir::Operation *op) {
    return isAnyOp<mlir::memref::LoadOp, mlir::affine::AffineLoadOp,
                   mlir::ptr::LoadOp>(op);
  }
};

struct DataRecomputationMemrefTracer {

  using Chain = llvm::SmallVector<mlir::Operation *, 4>;

  static std::pair<llvm::SmallVector<mlir::Operation *, 4>, mlir::Operation *>
  traceStore(mlir::memref::StoreOp storeOp) {
    auto memref = storeOp.getMemref();
    return traceMemref(memref);
  }

  static std::pair<llvm::SmallVector<mlir::Operation *, 4>, mlir::Operation *>
  traceStore(mlir::affine::AffineStoreOp storeOp) {
    auto memref = storeOp.getMemref();
    return traceMemref(memref);
  }

  static std::pair<llvm::SmallVector<mlir::Operation *, 4>, mlir::Operation *>
  traceMemref(mlir::TypedValue<mlir::MemRefType> memrefValue) {
    // Walk it back to where the memref originated
    auto *definingOp = memrefValue.getDefiningOp();

    Chain memrefChain;
    memrefChain.emplace_back(definingOp);

    // If the value comes from an argument, we need to look at inbounds

    for (bool finishFlag = false; !finishFlag;) {
      auto [finish, op] = traceOnce(memrefChain.back());
      memrefChain.emplace_back(op);
      finishFlag = finish;
    }

    return std::make_pair(std::move(memrefChain), memrefChain.back());
  }

  using TraceReturnTy = std::pair<bool, mlir::Operation *>;
  static TraceReturnTy traceOnce(mlir::Operation *op) {

    // Check if the operation should terminate the chain
    if (isAnyOp<mlir::memref::GetGlobalOp, mlir::memref::AllocOp>(op)) {
      return std::make_pair(true, op);
    }

    return mlir::TypeSwitch<mlir::Operation *, TraceReturnTy>(op)
        .Case<mlir::memref::ReinterpretCastOp>(
            [](mlir::memref::ReinterpretCastOp reinterpretCastOp) {
              const auto source = reinterpretCastOp.getSource();
              auto *definingOp = source.getDefiningOp();
              return std::make_pair(false, definingOp);
            })
        .Default(
            [](mlir::Operation *op) { return std::make_pair(true, nullptr); });
  }
};

struct CallInfo {
  bool insideLoop = false;
};

using FunctionWritesVec = llvm::SmallVector<DRWrite, 4>;
using FunctionWritesMap = llvm::DenseMap<mlir::Operation *, FunctionWritesVec>;

struct CallGraph {

  CallGraph() = default;

  llvm::SmallVector<mlir::func::FuncOp> getCallers(mlir::func::FuncOp callee) {
    llvm::SmallVector<mlir::func::FuncOp, 4> result;
    for (auto &funcOps : nodes) {
      if (llvm::find(calleeMap[funcOps], callee) != calleeMap[funcOps].end()) {
        result.emplace_back(funcOps);
      }
    }

    return result;
  }
  bool connect(mlir::func::FuncOp caller, mlir::func::FuncOp callee) {
    if (llvm::find(nodes, callee) == nodes.end()) {
      nodes.insert(callee);
    }
    calleeMap[caller].emplace_back(callee);
    return true;
  }

  bool addNode(mlir::func::FuncOp funcOp) { return nodes.insert(funcOp); }

  auto begin() { return nodes.begin(); }

  auto end() { return nodes.end(); }

  mlir::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::func::FuncOp, 4>>
      calleeMap;
  llvm::SetVector<mlir::func::FuncOp> nodes;
};

}; // namespace mlir::bmodelica::detail

namespace {

using namespace mlir::bmodelica::detail;

class DataRecomputationPass final
    : public mlir::bmodelica::impl::DataRecomputationPassBase<
          DataRecomputationPass> {

public:
  // Use the base class' constructor
  using DataRecomputationPassBase<
      DataRecomputationPass>::DataRecomputationPassBase;

  void runOnOperation() override;

private:
  mlir::ModuleOp moduleOp;

  FunctionWritesVec getFunctionWrites(mlir::func::FuncOp funcOp);

  /// A map of functions in the module
  llvm::DenseMap<mlir::StringRef, FuncOp> functionMap;

  FunctionWritesMap functionWritesMap;

  mlir::func::FuncOp entrypointFuncOp;

  mlir::LogicalResult identifyOpportunitiesAux(
      mlir::ModuleOp moduleOp, mlir::func::FuncOp funcOp,
      mlir::SymbolTableCollection &symTabCollection,
      llvm::SmallVector<std::pair<DRWrite, DRLoad>, 4> &foundOpportunities,
      llvm::SmallSet<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 8> &visitSet);

  mlir::FailureOr<llvm::SmallVector<std::pair<DRWrite, DRLoad>, 4>>
  identifyOpportunities(mlir::ModuleOp moduleOp,
                        mlir::func::FuncOp entrypointFuncOp,
                        mlir::SymbolTableCollection &symTabCollection);

  mlir::FailureOr<mlir::func::FuncOp> selectEntrypoint(mlir::ModuleOp moduleOp);

  ::mlir::bmodelica::detail::CallGraph
  buildCallGraph(mlir::ModuleOp moduleOp, mlir::func::FuncOp entrypointFuncOp,
                 mlir::SymbolTableCollection &symTabCollection);
};
} // namespace

namespace {

using mlir::bmodelica::IndexExpression;
using mlir::bmodelica::IndexExpressionNode;

struct AccessorTagConstant {};
struct AccessorTagLinear {};
struct AccessorTagMultiparam {};

struct Accessor {
  IndexExpression tree;
};

template <class T, std::size_t LineSize = 64>
static constexpr auto smallVectorCacheFitCapacity =
    [](std::size_t numLines) -> std::size_t {
  std::size_t metadataSize = sizeof(llvm::SmallVector<T, 0>);
  std::size_t remaining =
      LineSize - metadataSize + (numLines - 1) * (LineSize / sizeof(T));
  std::size_t elems = remaining / sizeof(T);

  return elems;
};
struct AccessorInterpreter {

  // Single cache-line size
  static llvm::SmallVector<int64_t, smallVectorCacheFitCapacity<int64_t>(1)>
  accessedIndices(const Accessor &accessor) {

    llvm::DenseMap<const IndexExpressionNode *, int64_t> resolvedValues{};

    // Do a depth-first post-order walk to calculate
    // static constexpr std::size_t fitsInTwoCacheLines = vecSize<int64_t,
    // /*NumLines*/2>;
    llvm::SmallVector<const IndexExpressionNode *,
                      smallVectorCacheFitCapacity<int64_t>(2)>
        stack;
    const IndexExpressionNode *current = &accessor.tree.root;

    while (true) {
      while (current != nullptr) {
        if (current->right) {
          stack.push_back(current->right.get());
        }
        stack.push_back(current); // Should be met topmost

        current = current->left.get();
      }

      current = stack.back();
      stack.pop_back();

      if (current->right.get() == stack.back()) {
        auto *right = stack.back();
        stack.pop_back();
        stack.push_back(current);
        current = right;
      } else {
        // Process
        MARCO_DBG() << "Processing!" << "\n";
      }

      if (stack.empty())
        break;
    }
  }
};

struct ParametricClobber {
  std::size_t numDims;
  llvm::SmallVector<std::size_t> dimSizes;

  // Per-dimension accessor
};

struct ReinterpretChainAnalysis {};

} // namespace

void DataRecomputationPass::runOnOperation() {
  // Gather all global statics
  moduleOp = this->getOperation();

  mlir::SymbolTableCollection symTabCollection{};

  // IMPORTANT NOTE:
  // This selection is for an IPO-analysis. In the final pass,
  // the selection should not be informed by a _dynamic prefix, but rather
  // something more general.

  auto entrypointFuncOption = selectEntrypoint(moduleOp);
  if (mlir::failed(entrypointFuncOption)) {
    MARCO_DBG() << "Failed to select entrypoint function\n";
    signalPassFailure();
    return;
  }
  entrypointFuncOp = entrypointFuncOption.value();

  llvm::DenseMap<mlir::memref::StoreOp, mlir::Operation *>
      memrefStoreToOriginOp{};

  llvm::DenseMap<mlir::memref::StoreOp, llvm::SmallVector<mlir::Operation *, 4>>
      memrefStoreToOriginChain{};

  struct ReinterpretInfo {
    mlir::Operation *castingOp;

    llvm::SmallVector<int64_t, 4> stridesBefore;
    int64_t offsetBefore;

    // mlir::MemRefType before;
    // mlir::MemRefType after;

    llvm::SmallVector<int64_t, 4> stridesAfter;
    int64_t offsetAfter;
  };

  moduleOp.walk([&](mlir::memref::StoreOp storeOp) {
    auto [chain, originOp] = DataRecomputationMemrefTracer::traceStore(storeOp);
    memrefStoreToOriginOp.insert(std::make_pair(storeOp, originOp));
    memrefStoreToOriginChain.insert(std::make_pair(storeOp, std::move(chain)));

    // Walk the chain and use the operands involved to build the expression.
    // What happens when the stride is simply reset with reinterpret_cast?

    llvm::DenseMap<mlir::memref::ReinterpretCastOp, Accessor>
        reinterpretExpressions;

    for (auto &[storeOp, chain] : memrefStoreToOriginChain) {
      for (auto *op : chain) {
        if (mlir::memref::ReinterpretCastOp reinterpretCastOp =
                mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(op)) {
          Accessor accessorExpression;

          auto range = reinterpretCastOp.getOffsets();

          if (range.empty() || range.size() > 1) {
            continue; // Do not handle
            // TODO: Handle multiple offsets
          }

          auto val = range[0];
          auto *definingOp = val.getDefiningOp();

          mlir::FailureOr<IndexExpressionNode> result =
              mlir::TypeSwitch<mlir::Operation *,
                               mlir::FailureOr<IndexExpressionNode>>(definingOp)
                  .Case<mlir::arith::AddIOp>([](mlir::arith::AddIOp addIOp) {
                    return IndexExpressionNode{};
                  })
                  .Default([](mlir::Operation *unhandledOp) {
                    return mlir::failure();
                  });
        }
      }
    }

    auto indices = storeOp.getIndices();
  });

  llvm::DenseMap<mlir::affine::AffineStoreOp, mlir::Operation *>
      affineStoreToOriginOp{};

  llvm::DenseMap<mlir::affine::AffineStoreOp,
                 llvm::SmallVector<mlir::Operation *, 4>>
      affineStoreToOriginChain{};

  moduleOp.walk([&](mlir::affine::AffineStoreOp storeOp) {
    auto [chain, originOp] = DataRecomputationMemrefTracer::traceStore(storeOp);
    affineStoreToOriginOp.insert(std::make_pair(storeOp, originOp));
    affineStoreToOriginChain.insert(std::make_pair(storeOp, std::move(chain)));
  });

  moduleOp.walk([&](FuncOp funcOp) {
    // Insert the function into the function map
    functionMap.insert(std::make_pair(funcOp.getName(), funcOp));
    functionWritesMap.insert(
        std::make_pair(funcOp.getOperation(), getFunctionWrites(funcOp)));
  });

  if (entrypointFuncOp) {

    auto callGraph =
        buildCallGraph(moduleOp, entrypointFuncOp, symTabCollection);

    auto opportunities =
        identifyOpportunities(moduleOp, entrypointFuncOp, symTabCollection);
  }
}

mlir::FailureOr<mlir::func::FuncOp>
DataRecomputationPass::selectEntrypoint(mlir::ModuleOp moduleOp) {

  mlir::SmallVector<mlir::func::FuncOp> entrypointFuncs{};

  moduleOp.walk([&entrypointFuncs](mlir::func::FuncOp funcOp) {
    auto nameStr = funcOp.getName().str();

    auto strIter = nameStr.find("_dynamic");

    if (strIter != std::string::npos) {
      entrypointFuncs.emplace_back(funcOp);
    }
  });

  auto compareFunc = [](mlir::func::FuncOp first, mlir::func::FuncOp second) {
    auto firstOpsRange = first.getRegion().getOps();
    auto secondOpsRange = second.getRegion().getOps();

    auto countFirst = llvm::count_if(firstOpsRange, [](mlir::Operation &op) {
      return static_cast<bool>(mlir::dyn_cast<mlir::func::FuncOp>(op));
    });

    auto countSecond = llvm::count_if(secondOpsRange, [](mlir::Operation &op) {
      return static_cast<bool>(mlir::dyn_cast<mlir::func::FuncOp>(op));
    });

    return countFirst < countSecond;
  };

  if (entrypointFuncs.empty()) {
    return mlir::failure();
  }

  if (entrypointFuncs.size() > 1) {
    // TODO: Solve entrypoint selection.
    // Hacky non-robust entrypoint selection.
    // Select the one with the most function calls
    // auto mostCallsFunction = std::max_element(entrypointFuncs.begin(),
    // Not memoized -- shouldn't be more than one to two candidates in the
    // cases we're testing now
    return *llvm::max_element(entrypointFuncs, compareFunc);
  }

  return entrypointFuncs.front();
}

FunctionWritesVec
DataRecomputationPass::getFunctionWrites(mlir::func::FuncOp funcOp) {

  FunctionWritesVec writes{};

  funcOp.walk([&](mlir::Operation *op) {
    // If affine store
    if (mlir::affine::AffineStoreOp storeOp =
            mlir::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
      writes.emplace_back(storeOp);
    }

    // If memref store
    if (mlir::memref::StoreOp storeOp =
            mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
      writes.emplace_back(storeOp);
    }
  });

  return writes;
}

mlir::FailureOr<llvm::SmallVector<std::pair<DRWrite, DRLoad>, 4>>
DataRecomputationPass::identifyOpportunities(
    mlir::ModuleOp moduleOp, mlir::func::FuncOp entrypointFuncOp,
    mlir::SymbolTableCollection &symTabCollection) {
  // Get all function calls, stores, and loads in here. Keep track of last
  // write.

  llvm::SmallVector<std::pair<DRWrite, DRLoad>, 4> result{};
  llvm::SmallSet<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 8> visitSet{};
  return identifyOpportunitiesAux(moduleOp, entrypointFuncOp, symTabCollection,
                                  result, visitSet);

  return result;
}

mlir::LogicalResult DataRecomputationPass::identifyOpportunitiesAux(
    mlir::ModuleOp moduleOp, mlir::func::FuncOp funcOp,
    mlir::SymbolTableCollection &symTabCollection,
    llvm::SmallVector<std::pair<DRWrite, DRLoad>, 4> &foundOpportunities,
    llvm::SmallSet<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 8> &visitSet) {

  llvm::DenseMap<mlir::MemRefType, DRWrite> lastWrites;

  // Keep track of how deep we're going at the point of visit
  llvm::SmallVector<mlir::func::FuncOp, 4> callStack{};
  // Build a stack of visits
  llvm::SmallVector<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 4> visitStack{};
  visitStack.emplace_back(funcOp);

  // If we encounter a previously visited function, we should access some
  // cached data instead of walking it again.

  while (!visitStack.empty()) {

    auto [inboundFuncOp, current] = visitStack.back();
    visitStack.pop_back();

    if ( visitSet.contains(std::make_pair(inboundFuncOp, current)) ) {
      // Get cached info
      // TODO: Add some kind of marker value to indicate that the memref
      // needs further tracing.
    }

    llvm::SmallVector<mlir::Operation *, 8> nestedCallOps{};

    // Collect all nested calls
    current.walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::CallOpInterface>(op)) {
        nestedCallOps.emplace_back(op);
      }
    });

    for (auto *callOp : nestedCallOps) {
      mlir::Operation *parentOp = callOp->getParentOp();
      bool insideLoop = false;

      while (parentOp && !mlir::isa<mlir::func::FuncOp>(parentOp)) {
        if (mlir::isa<mlir::LoopLikeOpInterface>(parentOp)) {
          insideLoop = true;
          break;
        }

        parentOp = parentOp->getParentOp();
      }

      auto callOpInterface = mlir::dyn_cast<mlir::CallOpInterface>(callOp);
      auto callable = callOpInterface.getCallableForCallee()
                          .dyn_cast<mlir::SymbolRefAttr>();

      // debug output
      if (insideLoop) {
        MARCO_DBG() << "Nested function call is inside a loop!\n";
        llvm::dbgs() << "Loop-nested function calls are not handled yet"
                     << "\n";
        signalPassFailure();
        return mlir::failure();
        // llvm_unreachable("Loop-bound function calls are not handled");
      }

      MARCO_DBG() << "Nested function call to " << callable
                  << " is NOT inside a loop!\n";
    }
  }

  return mlir::success();
}

::mlir::bmodelica::detail::CallGraph DataRecomputationPass::buildCallGraph(
    mlir::ModuleOp moduleOp, mlir::func::FuncOp entrypointFuncOp,
    mlir::SymbolTableCollection &symTabCollection) {
  llvm::SmallSet<mlir::func::FuncOp, 8> visitedSet{};
  llvm::SmallVector<mlir::func::FuncOp> visitStack{};

  visitStack.emplace_back(entrypointFuncOp);

  ::mlir::bmodelica::detail::CallGraph callGraph;

  while (!visitStack.empty()) {
    llvm::SmallVector<mlir::func::FuncOp, 4> nestedCallees{};
    mlir::func::FuncOp currentFuncOp = visitStack.back();
    visitStack.pop_back();

    // Skip the already visited function
    if (visitedSet.contains(currentFuncOp))
      continue;

    // Mark visited
    visitedSet.insert(currentFuncOp);
    callGraph.addNode(currentFuncOp);

    currentFuncOp.walk([&](mlir::Operation *op) {
      if (auto callOpInterface = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
        auto callable = callOpInterface.getCallableForCallee();

        if (auto calleeSym = callable.dyn_cast<mlir::SymbolRefAttr>()) {
          mlir::Operation *calleeOp =
              symTabCollection.lookupSymbolIn(moduleOp, calleeSym);

          if (auto calleeFuncOp =
                  mlir::dyn_cast<mlir::func::FuncOp>(calleeOp)) {
            nestedCallees.emplace_back(calleeFuncOp);
          } else {
            llvm_unreachable(
                "Couldn't handle a callee when building call graph");
          }
        }
      }
    });

    llvm::for_each(nestedCallees, [&](mlir::func::FuncOp funcOp) {
      callGraph.connect(currentFuncOp, funcOp);
    });
  }

  llvm::for_each(callGraph, [&callGraph](mlir::func::FuncOp funcOp) {
    auto callees = callGraph.getCallers(funcOp);

    MARCO_DBG() << funcOp.getName() << " is called by " << callees.size()
                << " functions:\n";

    for (auto &callee : callees) {
      MARCO_DBG() << "-- " << callee.getName() << "\n";
    }
  });

  return callGraph;
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createDataRecomputationPass() {
  return std::make_unique<DataRecomputationPass>();
}
} // namespace mlir::bmodelica
//
//
//
// struct ReinterpretChain {
//   llvm::SmallVector<ReinterpretInfo, 4> chain;

//   void performAnalysis(llvm::SmallVector<mlir::Operation *> &chain) {

//     // Skip the end
//     auto iter = std::next(chain.rbegin());
//     auto end = chain.rend();

//     auto *originOp = chain.back();

//     // Get the base loaded / alloced memref
//     mlir::MemRefType baseMemref = [](mlir::Operation *op) {
//        return mlir::TypeSwitch<mlir::Operation *, mlir::MemRefType>(op)
//            .Case<mlir::memref::GetGlobalOp>([](mlir::memref::GetGlobalOp
//            getGlobalOp){
//              return getGlobalOp.getType();
//            }).Case<mlir::memref::AllocOp>([](mlir::memref::AllocOp allocOp)
//            {
//              return allocOp.getType();
//            }).Case<mlir::memref::AllocaOp>([](mlir::memref::AllocaOp
//            allocaOp) {
//              return allocaOp.getType();
//           }).Default([](auto *op) -> mlir::MemRefType {
//             MARCO_DBG() << "Encountered unhandled operation type: " <<
//             op->getName() << "\n"; llvm_unreachable("Should not fall into
//             default type switch case");
//       });
//     }(originOp);

//     ReinterpretInfo currentReinterpret{};

//     // Assumption: loaded memrefs are not strided, and not offset
//     currentReinterpret.offsetBefore = 0;
//     for ( auto i = 0; i < baseMemref.getRank(); i++ ) {
//       currentReinterpret.stridesBefore.emplace_back(0);
//     }
//     currentReinterpret.offsetBefore = 0;

//     while ( iter != end ) {
//       iter = std::next(iter);

//       mlir::Operation *nextOp = *iter;

//       mlir::TypeSwitch<mlir::Operation *, void>(nextOp)
//         .Case<mlir::memref::CastOp>([&currentReinterpret](mlir::memref::CastOp
//         castOp){
//           // castOp.getOffsets()
//           return;
//          })
//         .Case<mlir::memref::ReinterpretCastOp>([&currentReinterpret](mlir::memref::ReinterpretCastOp
//         reinterpretCastOp){

//           auto offsets = reinterpretCastOp.getOffsets();
//           auto strides = reinterpretCastOp.getStrides();

//           for ( auto x : offsets ) {
//             MARCO_DBG() << "Offset? ";  x.getType().dump();
//           }

//           return;

//         }).Case<mlir::memref::SubViewOp>([&currentReinterpret](mlir::memref::SubViewOp
//         subviewOp){

//           return;
//         })
//         .Default([](auto *){llvm_unreachable("A reinterpret chain op wasn't
//         supported.");});
//     }
//   }

//   ///////////
//   ReinterpretChain(const llvm::SmallVector<mlir::Operation *> &inChain) {
//     assert(!inChain.empty() && "Passed an empty chain to ReinterpretChain
//     move constructor");

//     // TODO: Avoid copy?
//     auto chain = inChain;

//     performAnalysis(chain);

//   }

//   ///////////
//   explicit ReinterpretChain(llvm::SmallVector<mlir::Operation *> &&inChain)
//   {
//     assert(!inChain.empty() && "Passed an empty chain to ReinterpretChain
//     move constructor");

//     performAnalysis(inChain);
//   }

//   mlir::memref::ReinterpretCastOp reinterpretCastOp;
//   mlir::TypedValue<mlir::MemRefType> original;
//   mlir::TypedValue<mlir::MemRefType> reinterpreted;

//   bool isOriginal() {
//     return original == reinterpreted;
//   }

//   mlir::TypedValue<mlir::MemRefType> getValue();
//   mlir::TypedValue<mlir::MemRefType> getOriginalValue();
// };
