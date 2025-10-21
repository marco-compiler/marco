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

#define MARCO_DBG() llvm::dbgs() << R"(==== DataRecomputation: )"

namespace {
  template <class... OpTys>
  static bool isAnyOp(mlir::Operation *op)
  {
    return (mlir::isa<OpTys>(op) || ...);
  }
} // namespace

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
    if ( isAnyOp<mlir::memref::GetGlobalOp, mlir::memref::AllocOp>(op) ) {
      return std::make_pair(true, op);
    }

    return mlir::TypeSwitch<mlir::Operation *, TraceReturnTy>(op)
      .Case<mlir::memref::ReinterpretCastOp>(
        [](
          mlir::memref::ReinterpretCastOp reinterpretCastOp) {
          const auto source = reinterpretCastOp.getSource();
          auto *definingOp = source.getDefiningOp();
          return std::make_pair(false, definingOp);

        })
      .Default([](mlir::Operation *op) {
        return std::make_pair(true, nullptr);
      });
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

  mlir::ModuleOp moduleOp;

  ::detail::GlobalAccessMap prepareGlobalAccessMap(mlir::ModuleOp moduleOp);

  // Returns a Map of Globals and a Vector of operations accessing the globals
  ::detail::GlobalAccessMap getGlobalReads(mlir::ModuleOp moduleOp);
  ::detail::GlobalAccessMap getGlobalWrites(mlir::ModuleOp moduleOp);

  ::detail::FunctionWritesVec getFunctionWrites(mlir::func::FuncOp funcOp);

  /// A map of functions in the module
  llvm::DenseMap<mlir::StringRef, FuncOp> functionMap;

  ::detail::FunctionWritesMap functionWritesMap;

  mlir::func::FuncOp entrypointFuncOp;

  mlir::FailureOr<mlir::func::FuncOp>
   selectEntrypoint(mlir::ModuleOp moduleOp);
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
  moduleOp = this->getOperation();
  // auto globalAccesses = getGlobalReads(moduleOp);

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

  struct ReinterpretChain {
    llvm::SmallVector<ReinterpretInfo, 4> chain;

    explicit ReinterpretChain(llvm::SmallVector<mlir::Operation *> &&inChain)
    {

      assert(!inChain.empty() && "Passed an empty chain to ReinterpretChain move constructor");

      // Skip the end
      auto iter = std::next(inChain.rbegin());
      auto end = inChain.rend();

      auto *originOp = inChain.back();

      // Get the base loaded / alloced memref
      mlir::MemRefType baseMemref = [](mlir::Operation *op) {
         return mlir::TypeSwitch<mlir::Operation *, mlir::MemRefType>(op)
             .Case<mlir::memref::GetGlobalOp>([](mlir::memref::GetGlobalOp getGlobalOp){
               return getGlobalOp.getType();
             }).Case<mlir::memref::AllocOp>([](mlir::memref::AllocOp allocOp) {
               return allocOp.getType();
             }).Case<mlir::memref::AllocaOp>([](mlir::memref::AllocaOp allocaOp) {
               return allocaOp.getType();
            }).Default([](auto *) -> mlir::MemRefType { llvm_unreachable("Should not fall into default type switch case"); });
      }(originOp);

      ReinterpretInfo currentReinterpret{};

      // Assumption: loaded memrefs are not strided, and not offset
      currentReinterpret.offsetBefore = 0;
      for ( auto i = 0; i < baseMemref.getRank(); i++ ) {
        currentReinterpret.stridesBefore.emplace_back(0);
      }
      currentReinterpret.offsetBefore = 0;

      while ( iter != end ) {
        iter = std::next(iter);

        mlir::Operation *nextOp = *iter;

        mlir::TypeSwitch<mlir::Operation *, void>(nextOp)
          .Case<mlir::memref::CastOp>([&currentReinterpret](mlir::memref::CastOp castOp){
            // castOp.getOffsets()
            return;
           })
          .Case<mlir::memref::ReinterpretCastOp>([&currentReinterpret](mlir::memref::ReinterpretCastOp reinterpretCastOp){

            auto offsets = reinterpretCastOp.getOffsets();
            auto strides = reinterpretCastOp.getStrides();

            for ( auto x : offsets ) {
              MARCO_DBG() << "Offset? ";  x.getType().dump();

            }

            return;

          }).Case<mlir::memref::SubViewOp>([&currentReinterpret](mlir::memref::SubViewOp subviewOp){

            return;
          })
          .Default([](auto *){llvm_unreachable("A reinterpret chain op wasn't supported.");});


      }
    }

    mlir::memref::ReinterpretCastOp reinterpretCastOp;
    mlir::TypedValue<mlir::MemRefType> original;
    mlir::TypedValue<mlir::MemRefType> reinterpreted;

    bool isOriginal() {
      return original == reinterpreted;
    }

    mlir::TypedValue<mlir::MemRefType> getValue();
    mlir::TypedValue<mlir::MemRefType> getOriginalValue();
  };

  moduleOp.walk([&](mlir::memref::StoreOp storeOp) {
    auto [chain, originOp] =
        ::detail::DataRecomputationMemrefTracer::traceStore(storeOp);
    memrefStoreToOriginOp.insert(std::make_pair(storeOp, originOp));
    memrefStoreToOriginChain.insert(std::make_pair(storeOp, std::move(chain)));


    // Debug output
    for ( auto &[storeOp, chain] : memrefStoreToOriginChain ) {
      auto reinterpretingChain =
        llvm::filter_to_vector(chain, isAnyOp<mlir::memref::ReinterpretCastOp, mlir::memref::SubViewOp>);

      ReinterpretChain reinterpretedChain{std::move(reinterpretingChain)};


    }

    auto indices = storeOp.getIndices();
    MARCO_DBG() << "Memref store with " << indices.size() << " indices\n";
  });

  for (auto &[k, v] : memrefStoreToOriginOp) {
    mlir::StringRef parentName = "NONE";
    if ( auto parent = v->getParentOfType<mlir::func::FuncOp>() ) {
      parentName = parent.getName();
    }

  }

  llvm::DenseMap<mlir::ptr::StoreOp, mlir::Operation *> ptrStoreToOriginOp{};

  /*moduleOp.walk([&ptrStoreToOriginOp](mlir::ptr::StoreOp storeOp) {
    auto storePtr = storeOp.getPtr();
    auto *definingOp = storePtr.getDefiningOp();

    llvm::dbgs() << "Ptr store's ptr produced by: " << definingOp->getName() <<
  "\n";
  });*/

  llvm::DenseMap<mlir::affine::AffineStoreOp, mlir::Operation *>
      affineStoreToOriginOp{};

  llvm::DenseMap<mlir::affine::AffineStoreOp, llvm::SmallVector<mlir::Operation *, 4>>
      affineStoreToOriginChain{};

  moduleOp.walk([&](mlir::affine::AffineStoreOp storeOp) {
    auto [chain, originOp] =
        ::detail::DataRecomputationMemrefTracer::traceStore(storeOp);
    affineStoreToOriginOp.insert(std::make_pair(storeOp, originOp));
    affineStoreToOriginChain.insert(std::make_pair(storeOp, std::move(chain)));
  });

  moduleOp.walk([&](FuncOp funcOp) {
    // Insert the function into the function map
    functionMap.insert(std::make_pair(funcOp.getName(), funcOp));
    functionWritesMap.insert(
        std::make_pair(funcOp.getOperation(), getFunctionWrites(funcOp)));
  });
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
