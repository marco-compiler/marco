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
#include "mlir/Dialect/Vector/IR/VectorDialect.h.inc"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/ADT/DirectedGraph.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_DATARECOMPUTATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;
using namespace ::mlir::memref;
using namespace ::mlir::func;

//=== Rationale:
// The idea behind this pass is to keep track of stores and do a full
// module-wide interprocedural analysis to find loads from globals or other scoped
// allocations that can provably be shown to originate from another store, and
// that there are no interjecting writes to the same location that would
// invalidate it.
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
  DRWrite(mlir::memref::StoreOp storeOp) noexcept : storeOp{storeOp} { }
  DRWrite(mlir::affine::AffineStoreOp storeOp) noexcept : storeOp{storeOp} {}

  DRWrite(const DRWrite &) = default;
  DRWrite(DRWrite &&) = default;
  DRWrite &operator=(const DRWrite &) = default;
  DRWrite &operator=(DRWrite &&) = default;
  // Discern between memref write, affine write, tensor write, etc.
  std::variant<::mlir::memref::StoreOp, ::mlir::affine::AffineStoreOp> storeOp;
  std::optional<mlir::Operation *> allocatingOperation;

  bool isMemref() const {
    return std::holds_alternative<mlir::memref::StoreOp>(storeOp);
  }

  bool isAffine() const {
    return std::holds_alternative<mlir::affine::AffineStoreOp>(storeOp);
  }
};




using GlobalAccessPair = std::pair<llvm::StringRef, GlobalDef>;
using GlobalAccessMap = llvm::DenseMap<decltype(std::declval<GlobalAccessPair>().first), decltype(std::declval<GlobalAccessPair>().second)>;

using FunctionWritesVec = llvm::SmallVector<DRWrite, 4>;
using FunctionWritesMap = llvm::DenseMap<mlir::Operation *, FunctionWritesVec>;

} // namespace detail

namespace {

class DataRecomputationPass final
    : public impl::DataRecomputationPassBase<DataRecomputationPass> {

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
      if ( mlir::memref::ReinterpretCastOp reinterpretCastOp =
        mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(op) ) {

        auto sourceType = reinterpretCastOp.getSource().getType();
        auto shape = sourceType.getShape();

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

  llvm::for_each(globalAccesses, [](::detail::GlobalAccessPair &kvPair) {
    auto &[key, value] = kvPair;
    llvm::outs() << "Global Accesses: " << key.str() << "\n";
  });

  moduleOp.walk([](mlir::func::FuncOp funcOp) {
    auto &blocks = funcOp.getCallableRegion()->getBlocks();
    auto name = funcOp.getName();

    llvm::outs() << llvm::formatv("function {} has {} blocks", name,
                                  blocks.size())
                 << "\n";
  });

  moduleOp.walk([](GetGlobalOp getGlobalOp) {
    ::mlir::Block *block = getGlobalOp->getBlock();
    ::mlir::Operation *parentOp = block->getParentOp();

    if (auto funcOp = mlir::dyn_cast<FuncOp>(parentOp)) {
      llvm::outs() << llvm::formatv("block with {} ops has parent function {}",
                                    block->getOperations().size(),
                                    funcOp.getName())
                   << "\n";
    }
  });


  moduleOp.walk([&](FuncOp funcOp) {
    // Insert the function into the function map
    functionMap.insert(std::make_pair(funcOp.getName(), funcOp));
    functionWritesMap.insert(std::make_pair(funcOp.getOperation(), getFunctionWrites(funcOp)));

    llvm::dbgs() << "Walking " << funcOp.getName() << "\n";

    // Get the basic blocks
    llvm::DirectedGraph<mlir::Block *, mlir::ValueRange> localCFG{};

    mlir::DenseMap<size_t, mlir::Block *> basicBlocks{};
    size_t basicBlockCount = 0;

    llvm::for_each(funcOp.getBlocks(), [&](mlir::Block &block) {
      basicBlocks.insert(std::make_pair(basicBlockCount, &block));
      localCFG.addNode(basicBlocks[basicBlockCount++]);
    });

    // Find branching / jumping ops
    for ( auto *block : localCFG ) {
      auto *terminator = (*block)->getTerminator();
      auto numSuccessors = terminator->getSuccessors().size();
      auto bopi = mlir::cast<mlir::BranchOpInterface>(terminator);

      for ( size_t i = 0; i < numSuccessors; i++) {
        auto successorOperands = bopi.getSuccessorOperands(i);

        llvm::for_each(successorOperands.getForwardedOperands(), [](mlir::Value val) {
          if ( ! val.getDefiningOp() ) {
            return;
          }

          llvm::dbgs() << "Successor forwarding defining op: " << val.getDefiningOp()->getName() << "\n";
        });
      }
    }

    for ( auto *op : functionWritesMap.keys() ) {
      auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op);

      llvm::dbgs() << funcOp.getName() << " has " << functionWritesMap[op].size() << " writes\n";
    }

  });
}


::detail::FunctionWritesVec
DataRecomputationPass::getFunctionWrites(mlir::func::FuncOp funcOp) {

  ::detail::FunctionWritesVec writes{};

  funcOp.walk([&](mlir::Operation *op) {
    // If affine store
    if ( mlir::affine::AffineStoreOp storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op) ) {
      writes.emplace_back(storeOp);
    }

    // If memref store
    if ( mlir::memref::StoreOp storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op) ) {
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
