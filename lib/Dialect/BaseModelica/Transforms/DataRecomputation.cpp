//===- DataRecomputation.cpp - Handles recomputation candidates -----------===//
//
//===----------------------------------------------------------------------===//
//
// This file contains the DataRecomputation pass (pending a better name).
//===----------------------------------------------------------------------===//

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/DataRecomputation.h"

#include "marco/Modeling/GraphDumper.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Vector/IR/VectorDialect.h.inc"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DirectedGraph.h"
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

namespace detail {

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

// Generic Instantiable Index Set Element

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
};

struct DataRecomputationMemrefTracer {
  static mlir::Operation *traceStore(mlir::memref::StoreOp storeOp) {
    auto value = storeOp.getMemref();

    // Walk it back to where the memref originated
    auto *definingOp = value.getDefiningOp();

    llvm::SmallVector<mlir::Operation *, 4> memrefChain;
    memrefChain.emplace_back(definingOp);

    for (bool finishFlag = false; !finishFlag;) {
      mlir::Operation *lastOp = memrefChain.back();

      mlir::LogicalResult result =
          mlir::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(lastOp)
              .Case<mlir::memref::ReinterpretCastOp>(
                  [&memrefChain](
                      mlir::memref::ReinterpretCastOp reinterpretCastOp) {
                    auto source = reinterpretCastOp.getSource();
                    auto *definingOp = source.getDefiningOp();
                    memrefChain.emplace_back(definingOp);
                    return mlir::LogicalResult::success();
                  })
              .Case<mlir::memref::GetGlobalOp>(
                  [&memrefChain,
                   &finishFlag](mlir::memref::GetGlobalOp getGlobalOp) {
                    memrefChain.emplace_back(getGlobalOp.getOperation());
                    finishFlag = true;
                    return mlir::success();
                  })
              .Default([&finishFlag](mlir::Operation *op) {
                finishFlag = true;
                return mlir::failure();
              });

      if (result.failed()) {
        return nullptr;
      }
    }

    return memrefChain.back();
  }
};

using GlobalAccessPair = std::pair<llvm::StringRef, GlobalDef>;
using GlobalAccessMap =
    llvm::DenseMap<decltype(std::declval<GlobalAccessPair>().first),
                   decltype(std::declval<GlobalAccessPair>().second)>;

using FunctionWritesVec = llvm::SmallVector<DRWrite, 4>;
using FunctionWritesMap = llvm::DenseMap<mlir::Operation *, FunctionWritesVec>;

}; // namespace detail

namespace {

class DataRecomputationPass final
    : public mlir::bmodelica::impl::DataRecomputationPassBase<
          DataRecomputationPass> {

public:
  // Use the base class' constructor
  using DataRecomputationPassBase<
      DataRecomputationPass>::DataRecomputationPassBase;

  void runOnOperation() override;

private:
  ::detail::GlobalAccessMap prepareGlobalAccessMap(mlir::ModuleOp moduleOp);

  // Returns a Map of Globals and a Vector of operations accessing the globals
  ::detail::GlobalAccessMap getGlobalReads(mlir::ModuleOp moduleOp);
  ::detail::GlobalAccessMap getGlobalWrites(mlir::ModuleOp moduleOp);

  ::detail::FunctionWritesVec getFunctionWrites(mlir::func::FuncOp funcOp);

  /// A map of functions in the module
  llvm::DenseMap<mlir::StringRef, FuncOp> functionMap;

  ::detail::FunctionWritesMap functionWritesMap;

  mlir::func::FuncOp entrypointFuncOp;
  std::optional<mlir::func::FuncOp> selectEntrypoint(mlir::ModuleOp moduleOp);
};
} // namespace

::detail::GlobalAccessMap
DataRecomputationPass::prepareGlobalAccessMap(mlir::ModuleOp moduleOp) {
  ::detail::GlobalAccessMap result;

  moduleOp.walk([&result](mlir::memref::GlobalOp globalOp) {
    ::detail::GlobalDef globalDefinition{globalOp};

    result.insert(
        std::make_pair(globalDefinition.name, std::move(globalDefinition)));
  });

  return result;
}

::detail::GlobalAccessMap
DataRecomputationPass::getGlobalReads(mlir::ModuleOp moduleOp) {
  auto result = prepareGlobalAccessMap(moduleOp);

  moduleOp.walk([&result](mlir::memref::GetGlobalOp getGlobalOp) {
    auto name = getGlobalOp.getName();
    auto &accesses = result[name].accesses;
    accesses.insert(getGlobalOp.getOperation());

    auto users = getGlobalOp.getResult().getUsers();

    llvm::for_each(users, [&](mlir::Operation *op) {
      if (mlir::memref::ReinterpretCastOp reinterpretCastOp =
              mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(op)) {

        auto sourceType = reinterpretCastOp.getSource().getType();

        llvm::dbgs() << "A loaded global's memref was reinterpreted!" << "\n";
      }
    });
  });

  return result;
}

::detail::GlobalAccessMap
DataRecomputationPass::getGlobalWrites(mlir::ModuleOp moduleOp) {
  auto result = prepareGlobalAccessMap(moduleOp);

  return result;
}

void DataRecomputationPass::runOnOperation() {
  // Gather all global statics
  mlir::ModuleOp moduleOp = this->getOperation();
  auto globalAccesses = getGlobalReads(moduleOp);

  // Assume entrypoint

  // IMPORTANT NOTE:
  // This selection is for an IPO-analysis. In the final pass,
  // the selection should not be informed by a _dynamic prefix, but rather
  // something more general.

  // Immediately invoked lambda expression (IILE)

  auto entrypointFuncOption = selectEntrypoint(moduleOp);
  if (!entrypointFuncOption) {
    llvm::dbgs() << "Failed to select entrypoint function\n";
    signalPassFailure();
    return;
  }

  llvm::dbgs() << "Selected " << entrypointFuncOp.getName()
               << " as entrypoint\n";

  llvm::for_each(globalAccesses, [](::detail::GlobalAccessPair &kvPair) {
    auto &[key, value] = kvPair;
    llvm::dbgs() << "Global Accesses: " << key.str() << "\n";
  });

  llvm::DenseMap<mlir::memref::StoreOp, mlir::Operation *>
      memrefStoreToOriginOp{};

  moduleOp.walk([&memrefStoreToOriginOp](mlir::memref::StoreOp storeOp) {
    auto *memrefOriginOp =
        ::detail::DataRecomputationMemrefTracer::traceStore(storeOp);
    memrefStoreToOriginOp.insert(std::make_pair(storeOp, memrefOriginOp));
  });

  llvm::DenseMap<mlir::ptr::StoreOp, mlir::Operation *> ptrStoreToOriginOp{};

  /*moduleOp.walk([&ptrStoreToOriginOp](mlir::ptr::StoreOp storeOp) {
    auto storePtr = storeOp.getPtr();
    auto *definingOp = storePtr.getDefiningOp();

    llvm::dbgs() << "Ptr store's ptr produced by: " << definingOp->getName() <<
  "\n";
  });*/

  llvm::DenseMap<mlir::affine::AffineStoreOp, mlir::Operation *>
      affineStoreToOriginOp{};

  moduleOp.walk([&affineStoreToOriginOp](mlir::affine::AffineStoreOp storeOp) {
    auto memref = storeOp.getMemref();
    auto *definingOp = memref.getDefiningOp();
  });

  for (auto &[k, v] : memrefStoreToOriginOp) {
    llvm::dbgs() << k->getName() << " had its memref loaded at " << v->getName()
                 << "\n";
  }

  moduleOp.walk([](mlir::memref::GetGlobalOp getGlobalOp) {
    ::mlir::Operation *parentOp = getGlobalOp->getParentOp();

    if (auto funcOp = mlir::dyn_cast<FuncOp>(parentOp)) {
      llvm::outs() << llvm::formatv("GetGlobalOp to {} has parent function {}",
                                    getGlobalOp.getName(), funcOp.getName())
                   << "\n";
    }
  });

  moduleOp.walk([&](FuncOp funcOp) {
    // Insert the function into the function map
    functionMap.insert(std::make_pair(funcOp.getName(), funcOp));
    functionWritesMap.insert(
        std::make_pair(funcOp.getOperation(), getFunctionWrites(funcOp)));

    // Get the basic blocks
    llvm::DirectedGraph<mlir::Block *, mlir::ValueRange> localCFG{};

    mlir::DenseMap<size_t, mlir::Block *> basicBlocks{};
    size_t basicBlockCount = 0;

    llvm::for_each(funcOp.getBlocks(), [&](mlir::Block &block) {
      basicBlocks.insert(std::make_pair(basicBlockCount, &block));
      localCFG.addNode(basicBlocks[basicBlockCount++]);
    });

    for (auto *op : functionWritesMap.keys()) {
      auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op);
    }
  });
}

std::optional<mlir::func::FuncOp>
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
    return std::nullopt;
  }

  if (entrypointFuncs.size() > 1) {
    // Select the one with the most function calls
    // auto mostCallsFunction = std::max_element(entrypointFuncs.begin(),
    // Not memoized -- shouldn't be more than one to two candidates in the
    // cases we're testing now
    return *llvm::max_element(entrypointFuncs, compareFunc);
  }

  return entrypointFuncs.front();
}

// template <class Variant, class... OpTys>
// void opTypeSwitch(mlir::Operation *op, std::function<void(mlir::Operation*)>
// callback) {
//     ([&](mlir::Operation *op) {
//         if ( auto castedOp = mlir::dyn_cast<OpTys>( op ) ) {
//             callback(castedOp);
//         }
//     });
// }

::detail::FunctionWritesVec
DataRecomputationPass::getFunctionWrites(mlir::func::FuncOp funcOp) {

  ::detail::FunctionWritesVec writes{};

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

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createDataRecomputationPass() {
  return std::make_unique<DataRecomputationPass>();
}
} // namespace mlir::bmodelica
