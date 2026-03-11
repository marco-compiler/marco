//===- AnalysisState.h - Data model for load-store provenance analysis ----===//
//
//===----------------------------------------------------------------------===//
//
// Vocabulary types for DataRecomputation.cpp.
//
// Every type alias, struct, and context class used by the reaching-stores
// analysis lives here.  Reading this header gives you a complete picture
// of the data that flows through the pass.
//
//===----------------------------------------------------------------------===//

#ifndef MARCO_TRANSFORMS_DATARECOMPUTATION_ANALYSISSTATE_H
#define MARCO_TRANSFORMS_DATARECOMPUTATION_ANALYSISSTATE_H

#include "marco/Transforms/DataRecomputationIndexing.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace dr {

/// Pass-level context threaded through every analysis function.
/// Bundles the MLIR context, symbol table collection, and module root.
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

/// Maps each value that stems from an allocation to the allocating operation.
using AllocationRoots = llvm::DenseMap<mlir::Value, mlir::Operation *>;

/// Maps each store op to the set of allocation roots its stored value
/// depends on (via loads in the SSA operand chain).
using StoreValueDeps =
    llvm::DenseMap<mlir::Operation *,
                   llvm::SmallDenseSet<mlir::Operation *, 4>>;

/// Set of memref.store operations (plus nullptr for external/unknown writes)
/// that may have been the most recent write to a given allocation root.
using StoreSet = llvm::SmallDenseSet<mlir::Operation *, 4>;

/// An entry in the reaching-stores analysis: a store op plus offset coverage.
struct IndexedStore {
  mlir::Operation *storeOp; // nullptr = external write
  /// Concrete indices this store must-write-to.
  /// nullopt = may-write-anywhere (dynamic indices, through views, rank-0).
  std::optional<PointSet> coverage;
};

/// Vector of indexed store entries for one allocation root.
using IndexedStoreVec = llvm::SmallVector<IndexedStore, 4>;

/// Maps each allocation root (the op that created the memory) to its
/// reaching indexed stores.  This is the state threaded through the analysis.
using StoreMap = llvm::DenseMap<mlir::Operation *, IndexedStoreVec>;

/// The pass's primary output: maps each load op to the set of stores that
/// may have been the last writer.
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

/// Result of analyzing how a defined callee uses a specific memref argument.
struct CalleeArgEffect {
  bool mayWrite = false;
  bool passedToCall = false;
  llvm::SmallVector<mlir::Operation *, 4> storeOps;
};

/// All analysis state bundled for threading through analyzeBlock/analyzeOp.
struct AnalysisContext {
  DRPassContext &passCtx;
  llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor;
  LoadProvenanceMap &loadProv;
  EnrichedCallGraph &callGraph;
  llvm::DenseSet<mlir::Operation *> inProgress;
  const StoreValueDeps &storeValueDeps;
  const llvm::DenseSet<mlir::Operation *> &globalAllocOps;
};

} // namespace dr

#endif // MARCO_TRANSFORMS_DATARECOMPUTATION_ANALYSISSTATE_H
