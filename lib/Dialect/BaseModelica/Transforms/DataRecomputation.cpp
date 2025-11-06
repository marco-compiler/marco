//===- DataRecomputation.cpp - Handles recomputation candidates -----------===//
//
//===----------------------------------------------------------------------===//
//
// This file contains the DataRecomputation pass (pending a better name).
//===----------------------------------------------------------------------===//

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/DataRecomputation.h"

#include "marco/Dialect/BaseModelica/Transforms/DataRecomputation/IndexExpression.h"
#include "marco/Dialect/Runtime/IR/Ops.h"
#include "marco/Modeling/GraphDumper.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Vector/IR/VectorDialect.h.inc"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
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
inproceedings{cai2000parametric,
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

template <class... OpTys>
struct OpTypeList {
  static bool isAnyOp(mlir::Operation *op) { return (op != nullptr) && ::isAnyOp<OpTys...>(op); }

  using VariantT = std::variant<OpTys...>;
};

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

template <class OpTypeListInstance>
struct VariantResolver {

  using VariantT = typename OpTypeListInstance::VariantT;

  template <class... OpTys>
  static VariantT resolveImpl(mlir::Operation *op, OpTypeList<OpTys...>) {
    VariantT result;

    bool success = ([&result](mlir::Operation *op) -> bool {
      if (OpTys resolvedOp = mlir::dyn_cast<OpTys>(op)) {
        result = resolvedOp;
        return true;
      }
      return false;
    }(op) || ...);

    if (!success) {
      llvm_unreachable("No support for op type");
    }

    return result;
  }

  static VariantT resolve(mlir::Operation *op) {
    return resolveImpl(op, OpTypeListInstance{});
  }
};

/// DataRecomputation Write
/// \see DenseMapInfo<DRWrite>
struct DRWrite {
  using SupportedOpTypes =
      OpTypeList<::mlir::memref::StoreOp, ::mlir::affine::AffineStoreOp,
                 ::mlir::ptr::StoreOp>;

  using StoreVariant = SupportedOpTypes::VariantT;

  StoreVariant resolveOp(mlir::Operation *op) {
    return VariantResolver<SupportedOpTypes>::resolve(op);
  }

  DRWrite(mlir::Operation *op) : storeOp{resolveOp(op)}, operation{op} {}
  DRWrite(mlir::memref::StoreOp storeOp) noexcept
      : storeOp{storeOp}, operation{storeOp.getOperation()} {}
  DRWrite(mlir::affine::AffineStoreOp storeOp) noexcept
      : storeOp{storeOp}, operation{storeOp.getOperation()} {}
  DRWrite(mlir::ptr::StoreOp storeOp) noexcept
      : storeOp{storeOp}, operation{storeOp.getOperation()} {}

  DRWrite(const DRWrite &) = default;
  DRWrite(DRWrite &&) = default;
  DRWrite &operator=(const DRWrite &) = default;
  DRWrite &operator=(DRWrite &&) = default;

  StoreVariant storeOp;
  mlir::Operation *operation;

  operator mlir::Operation *() const { return operation; }

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
    return SupportedOpTypes::isAnyOp(op);
  }
};

/// DataRecomputation Write
/// \see DenseMapInfo<DRWrite>
struct DRAlloc {
  using SupportedOpTypes =
      OpTypeList<::mlir::memref::GlobalOp, ::mlir::memref::AllocOp,
                 ::mlir::memref::AllocaOp>;

  using AllocVariant = SupportedOpTypes::VariantT;

  AllocVariant resolveOp(mlir::Operation *op) {
    return VariantResolver<SupportedOpTypes>::resolve(op);
  }

  DRAlloc(mlir::Operation *op) : allocOp{resolveOp(op)}, operation{op} {}
  DRAlloc(mlir::memref::AllocOp allocOp) noexcept
      : allocOp{allocOp}, operation{allocOp.getOperation()} {}

  DRAlloc(mlir::memref::AllocaOp allocaOp) noexcept
      : allocOp{allocaOp}, operation{allocaOp.getOperation()} {}

  DRAlloc(mlir::memref::GlobalOp globalOp) noexcept
      : allocOp{globalOp}, operation{globalOp.getOperation()} {}

  DRAlloc(const DRAlloc &) = default;
  DRAlloc(DRAlloc &&) = default;
  DRAlloc &operator=(const DRAlloc &) = default;
  DRAlloc &operator=(DRAlloc &&) = default;

  AllocVariant allocOp;
  mlir::Operation *operation;

  operator mlir::Operation *() const { return operation; }

  bool isGlobal() const {
    return std::holds_alternative<mlir::memref::GlobalOp>(allocOp);
  }

  bool isAlloc() const {
    return std::holds_alternative<mlir::memref::AllocOp>(allocOp);
  }

  bool isAlloca() const {
    return std::holds_alternative<mlir::memref::AllocaOp>(allocOp);
  }

  /// Utility function to check if an mlir::Operation * fits as a DRAlloc
  static bool isAllocOp(mlir::Operation *op) {
    return SupportedOpTypes::isAnyOp(op);
  }
};

//! A class representing a generic load with regards to the pass.
//! \see DenseMapInfo<DRLoad>
struct DRLoad {
  using SupportedOpTypes =
      OpTypeList<::mlir::memref::LoadOp, ::mlir::affine::AffineLoadOp,
                 ::mlir::ptr::LoadOp>;

  using LoadVariant = SupportedOpTypes::VariantT;

  LoadVariant resolveOp(mlir::Operation *op) {
    return VariantResolver<SupportedOpTypes>::resolve(op);
  }

  //! A general casting constructor. Will leave things in an empty state if
  //! failed.
  DRLoad(mlir::Operation *op) : loadOp{resolveOp(op)}, operation{op} {}
  DRLoad(mlir::memref::LoadOp loadOp) noexcept
      : loadOp{loadOp}, operation{loadOp.getOperation()} {}
  DRLoad(mlir::affine::AffineLoadOp loadOp) noexcept
      : loadOp{loadOp}, operation{loadOp.getOperation()} {}
  DRLoad(mlir::ptr::LoadOp loadOp) noexcept
      : loadOp{loadOp}, operation{loadOp.getOperation()} {}

  DRLoad(const DRLoad &) = default;
  DRLoad(DRLoad &&) = default;
  DRLoad &operator=(const DRLoad &) = default;
  DRLoad &operator=(DRLoad &&) = default;

  // Hold a pointer
  LoadVariant loadOp;
  mlir::Operation *operation = nullptr;
  ;

  // Casting to Operation *
  operator mlir::Operation *() const { return operation; }

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
    return SupportedOpTypes::isAnyOp(op);
  }
};

//===============================================//
// Memref Tracing Utility
//-----------------------------------------------//
// Allows you to trace back a memref to its
// originating op.
//===============================================//

struct DRMemrefTracerFailureMarker {};

struct DRMemrefTracerResult
    : public std::variant<mlir::Operation *, mlir::Value,
                          DRMemrefTracerFailureMarker> {
  using Base =
      std::variant<mlir::Operation *, mlir::Value, DRMemrefTracerFailureMarker>;
  using Base::variant;

  static DRMemrefTracerResult failure() {
    return DRMemrefTracerResult{DRMemrefTracerFailureMarker{}};
  }

  std::optional<mlir::Operation *> getOperation() {
    if (std::holds_alternative<mlir::Operation *>(*this)) {
      return std::get<operationIdx>(*this);
    }
    return std::nullopt;
  }

  std::optional<mlir::Value> getValue() {
    if (std::holds_alternative<mlir::Value>(*this)) {
      return std::get<valueIdx>(*this);
    }
    return std::nullopt;
  }

  bool isFailure() {
    return std::holds_alternative<DRMemrefTracerFailureMarker>(*this);
  }

private:
  static constexpr std::size_t operationIdx = 0;
  static constexpr std::size_t valueIdx = 1;
  static constexpr std::size_t failureMarkerIdx = 2;
};

struct DRMemrefTracer {

  using TraceResultTy = DRMemrefTracerResult;

  using Chain = llvm::SmallVector<TraceResultTy, 4>;
  using TracePairTy = std::pair<Chain, TraceResultTy>;

  static TracePairTy
  traceMemref(mlir::TypedValue<mlir::MemRefType> memrefValue,
              mlir::MLIRContext *context, mlir::ModuleOp moduleOp,
              mlir::SymbolTableCollection &symTabCollection) {

    Chain memrefChain;
    // Walk it back to where the memref originated
    auto *definingOp = memrefValue.getDefiningOp();

    if ( definingOp == nullptr ) {
      MARCO_DBG() << "Defining op is NULL\n";
    } else {
      definingOp->dump();
    }

    memrefChain.emplace_back(definingOp);

    for (bool finishFlag = false; !finishFlag;) {
      // Unwrap
      auto &back = memrefChain.back();

      if (auto operationOpt = back.getOperation()) {
        auto *backOp = operationOpt.value();
        auto [finish, singleTraceResult] =
            traceOnce(backOp, context, moduleOp, symTabCollection);
        memrefChain.emplace_back(singleTraceResult);
        finishFlag = finish;
      } else {
        // We may have a value here.
        break;
      }
    }

    return std::make_pair(std::move(memrefChain), memrefChain.back());
  }

  using TraceReturnTy = std::pair<bool, TraceResultTy>;
  static TraceReturnTy
  traceOnce(mlir::Operation *op, mlir::MLIRContext *context,
            mlir::ModuleOp moduleOp,
            mlir::SymbolTableCollection &symTabCollection) {

    // Check if the operation should terminate the chain
    if (DRAlloc::isAllocOp(op)) {
      return std::make_pair(true, DRMemrefTracerResult{op});
    }

    if (op == nullptr) {
      llvm_unreachable("Trace ran into a wall");
    } else {
      MARCO_DBG();
      op->dump();
      std::cerr << "\n";
    }

    /// Used to denote whether something illegal happened
    bool failureFlag;

    using SingleTraceResult = std::tuple<bool, mlir::Value, mlir::Operation * >;
    SingleTraceResult singleTraceResult =
        mlir::TypeSwitch<mlir::Operation *, SingleTraceResult>(op)
            .Case<mlir::memref::ReinterpretCastOp>(
                [](mlir::memref::ReinterpretCastOp reinterpretCastOp) {
                  const auto source = reinterpretCastOp.getSource();
                  auto *definingOp = source.getDefiningOp();
                  return std::make_tuple(false, source, definingOp);
                })
            .Case<mlir::memref::GetGlobalOp>(
                [&](mlir::memref::GetGlobalOp getGlobalOp) {
                  auto symbolRef = getGlobalOp.getName();
                  mlir::SymbolRefAttr symbolReference =
                      FlatSymbolRefAttr::get(context, symbolRef);

                  auto *symbolOp = symTabCollection.lookupSymbolIn(
                      moduleOp.getOperation(), symbolReference);

                  if (auto globalOp =
                          mlir::dyn_cast<mlir::memref::GlobalOp>(symbolOp)) {
                    return std::make_tuple(true, mlir::Value{}, symbolOp);
                  }

                  llvm_unreachable(
                      "A GetGlobalOp shouldn't resolve to anything "
                      "other than GlobalOp!");
                })
            .Default([&](mlir::Operation *op) {
              if ( DRAlloc::isAllocOp(op) ) {
                failureFlag = false;
                return std::make_tuple(true, mlir::Value{}, op);
              }

              failureFlag = true;
              return std::make_tuple(true, mlir::Value{}, static_cast<mlir::Operation *>(nullptr));
            });

    auto [terminateFlag, definingOp, value] = singleTraceResult;

    if ( failureFlag ) {
      return std::make_pair(true, TraceResultTy::failure());
    }

    if (definingOp == nullptr) {
      return std::make_pair(terminateFlag, TraceResultTy{value});
    }
    return std::make_pair(terminateFlag, TraceResultTy{definingOp});
  }
};

struct DRLoadTracer {

private:
  mlir::MLIRContext *context;
  mlir::ModuleOp moduleOp;
  mlir::SymbolTableCollection &symTabCollection;

public:
  using Self = DRLoadTracer;

  using SupportedLoadOpTypes =
      OpTypeList<::mlir::memref::LoadOp, ::mlir::affine::AffineLoadOp>;

  using TraceResultTy = DRMemrefTracer::TracePairTy;

  DRLoadTracer(mlir::MLIRContext *context, mlir::ModuleOp moduleOp,
               mlir::SymbolTableCollection &symTabCollection)
      : context{context}, moduleOp{moduleOp},
        symTabCollection{symTabCollection} {}

  /// Dispatch helper. The static member infrastructure
  /// takes care of efficient dispatch.
  template <class T>
  TraceResultTy trace(T op) {
    return Self::trace(op, context, moduleOp, symTabCollection);
  }

  ///===== STATIC MEMBERS ====== //
public:
  /// A variadic dispatch. Will try to cast each supported type in order, and
  /// dispatch to the correct instance.
  template <class... Tys>
  static TraceResultTy traceDispatcherImpl(
      mlir::Operation *op, mlir::MLIRContext *context, mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection &symTabCollection, OpTypeList<Tys...> opTys) {
    TraceResultTy result{};

    ([&](mlir::Operation *op) -> bool {
      if (auto typeQualifiedOp = mlir::dyn_cast<Tys>(op)) {
        result = std::move(
            trace(typeQualifiedOp, context, moduleOp, symTabCollection));
        return true;
      }
      return false;
    }(op) || ...);

    return result;
  }

  static TraceResultTy
  traceDispatcher(mlir::Operation *op, mlir::MLIRContext *context,
                  mlir::ModuleOp moduleOp,
                  mlir::SymbolTableCollection &symTabCollection,
                  SupportedLoadOpTypes opTys = {}) {
    return traceDispatcherImpl(op, context, moduleOp, symTabCollection, opTys);
  }

  static TraceResultTy trace(mlir::Operation *op, mlir::MLIRContext *context,
                             mlir::ModuleOp moduleOp,
                             mlir::SymbolTableCollection &symTabCollection) {
    return traceDispatcher(op, context, moduleOp, symTabCollection);
  }

  static TraceResultTy trace(mlir::memref::LoadOp loadOp,
                             mlir::MLIRContext *context,
                             mlir::ModuleOp moduleOp,
                             mlir::SymbolTableCollection &symTabCollection) {
    auto memref = loadOp.getMemref();
    return DRMemrefTracer::traceMemref(memref, context, moduleOp,
                                       symTabCollection);
  }

  static TraceResultTy trace(mlir::affine::AffineLoadOp loadOp,
                             mlir::MLIRContext *context,
                             mlir::ModuleOp moduleOp,
                             mlir::SymbolTableCollection &symTabCollection) {
    auto memref = loadOp.getMemref();
    return DRMemrefTracer::traceMemref(memref, context, moduleOp,
                                       symTabCollection);
  }
};

struct DRStoreTracer {

private:
  mlir::MLIRContext *context;
  mlir::ModuleOp moduleOp;
  mlir::SymbolTableCollection &symTabCollection;

public:
  using Self = DRStoreTracer;
  using SupportedStoreOpTypes =
      OpTypeList<mlir::memref::StoreOp, mlir::affine::AffineStoreOp>;

  using TraceResultTy = DRMemrefTracer::TracePairTy;

  DRStoreTracer(mlir::MLIRContext *context, mlir::ModuleOp moduleOp,
                mlir::SymbolTableCollection &symTabCollection)
      : context{context}, moduleOp{moduleOp},
        symTabCollection{symTabCollection} {}

  /// Dispatch helper. The static member infrastructure
  /// takes care of efficient dispatch.
  template <class T>
  TraceResultTy trace(T op) {
    return Self::trace(op, context, moduleOp, symTabCollection);
  }

  ///===== STATIC MEMBERS ====== //
public:
  /// A variadic dispatch. Will try to cast each supported type in order, and
  /// dispatch to the correct instance.
  template <class... Tys>
  static TraceResultTy traceDispatcherImpl(
      mlir::Operation *op, mlir::MLIRContext *context, mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection &symTabCollection, OpTypeList<Tys...> opTys) {
    TraceResultTy result{};

    ([&](mlir::Operation *op) -> bool {
      if (auto typeQualifiedOp = mlir::dyn_cast<Tys>(op)) {
        result = std::move(
            trace(typeQualifiedOp, context, moduleOp, symTabCollection));
        return true;
      }
      return false;
    }(op) || ...);

    return result;
  }

  static TraceResultTy
  traceDispatcher(mlir::Operation *op, mlir::MLIRContext *context,
                  mlir::ModuleOp moduleOp,
                  mlir::SymbolTableCollection &symTabCollection,
                  SupportedStoreOpTypes opTys = {}) {
    return traceDispatcherImpl(op, context, moduleOp, symTabCollection, opTys);
  }

  static TraceResultTy trace(mlir::Operation *op, mlir::MLIRContext *context,
                             mlir::ModuleOp moduleOp,
                             mlir::SymbolTableCollection &symTabCollection) {
    return traceDispatcher(op, context, moduleOp, symTabCollection);
  }

  static TraceResultTy trace(mlir::memref::StoreOp storeOp,
                             mlir::MLIRContext *context,
                             mlir::ModuleOp moduleOp,
                             mlir::SymbolTableCollection &symTabCollection) {
    auto memref = storeOp.getMemref();
    return DRMemrefTracer::traceMemref(memref, context, moduleOp,
                                       symTabCollection);
  }

  static TraceResultTy trace(mlir::affine::AffineStoreOp storeOp,
                             mlir::MLIRContext *context,
                             mlir::ModuleOp moduleOp,
                             mlir::SymbolTableCollection &symTabCollection) {
    auto memref = storeOp.getMemref();
    return DRMemrefTracer::traceMemref(memref, context, moduleOp,
                                       symTabCollection);
  }
};

///===============================================================
/// Call Graph
///===============================================================

/// A struct representing a forwarded / bound memref through a function call.
struct FunctionCallMemrefBinding {
  /// The index the argument appears in the function call
  std::size_t argumentIndex;

  /// The originating operation for the memref
  mlir::Operation *originOp;
};

struct DRFunctionArgumentBinder {

  using BindingListTy = llvm::SmallVector<FunctionCallMemrefBinding, 2>;

  static BindingListTy
  findBindings(mlir::CallOpInterface callOpInterface, mlir::ModuleOp moduleOp,
               mlir::MLIRContext *context,
               mlir::SymbolTableCollection &symTabCollection) {
    BindingListTy result{};

    auto operands = callOpInterface.getArgOperands();
    const std::size_t numArguments = operands.size();

    for (std::size_t idx = 0; idx < numArguments; idx++) {
      mlir::Value operand = operands[idx];

      if (auto typedMemrefValue =
              mlir::dyn_cast<mlir::TypedValue<mlir::MemRefType>>(operand)) {
        auto [chain, origin] = DRMemrefTracer::traceMemref(
            typedMemrefValue, context, moduleOp, symTabCollection);

        if ( auto definingOpOpt = origin.getOperation() ) {
          result.emplace_back(FunctionCallMemrefBinding{idx, definingOpOpt.value()});
        }
      }
    }
    return result;
  }
};

struct CallGraphNode;

// bool operator==(const CallGraphNode &, const CallGraphNode &);

// TODO(Tor): Add argument value / op set?
/// Represents a call from one function to another
struct CallGraphEdge {

  using BindingsContainerTy = llvm::SmallVector<FunctionCallMemrefBinding, 2>;

  CallGraphEdge(mlir::func::FuncOp caller, mlir::func::FuncOp callee)
      : CallGraphEdge{caller, callee, BindingsContainerTy{}} {}

  CallGraphEdge(mlir::func::FuncOp caller, mlir::func::FuncOp callee,
                BindingsContainerTy &&bindings)
      : caller{caller}, callee{callee}, bindings{std::move(bindings)} {}

  mlir::func::FuncOp caller;
  mlir::func::FuncOp callee;
  BindingsContainerTy bindings;

  /// Must be set explicitly after construction.
  /// TODO(Tor): If inspired, fix.
  CallGraphNode *targetNode;

  const CallGraphNode &getTargetNode() const { return *targetNode; }

  friend bool operator==(const CallGraphEdge &lhs, const CallGraphEdge &rhs) {
    CallGraphEdge *ncLHS = const_cast<CallGraphEdge *>(&lhs);
    CallGraphEdge *ncRHS = const_cast<CallGraphEdge *>(&rhs);

    return ncLHS->caller == ncRHS->caller && ncLHS->callee == ncRHS->callee;
  }

  bool isEqualTo(const CallGraphEdge &other) const { return *this == other; }
};

struct CallGraphNode {

  explicit CallGraphNode(mlir::func::FuncOp funcOp) : funcOp{funcOp}, edges{} {}

  friend bool operator==(const CallGraphNode &lhs, mlir::func::FuncOp rhs) {
    return lhs.funcOp == rhs;
  }

  friend bool operator==(const CallGraphNode &lhs, const CallGraphNode &rhs) {
    return lhs.funcOp == rhs.funcOp;
  }

  bool findEdgesTo(const CallGraphNode &otherNode,
                   ::mlir::SmallVectorImpl<CallGraphEdge *> &resEdges) {

    llvm::for_each(edges, [&](CallGraphEdge &edge) {
      if (edge.callee == otherNode.funcOp) {
        resEdges.emplace_back(&edge);
      }
    });

    return !resEdges.empty();
  }

  void clear() { edges.clear(); }

  /// The function this node represents
  mlir::func::FuncOp funcOp;

  bool isEqualTo(const CallGraphNode &other) const {
    return ::mlir::operator==(funcOp, other.funcOp);
  }

  bool addEdge(CallGraphEdge &edge) {
    edges.emplace_back(edge);
    return true;
  }

  void removeEdge(CallGraphEdge &edge) {
    // Use erase as it is stable.
    // Erase-remove if not necessary.
    llvm::erase_if(
        edges, [&edge](CallGraphEdge &other) { return other.isEqualTo(edge); });
  }

  llvm::SmallVector<CallGraphEdge> edges;
};

class CallGraph : public llvm::DirectedGraph<CallGraphNode, CallGraphEdge> {

  using StorageTy = llvm::SmallVector<CallGraphNode, 10>;
  using Base = llvm::DirectedGraph<CallGraphNode, CallGraphEdge>;

  using iterator = StorageTy::iterator;
  using const_iterator = StorageTy::const_iterator;

public:

  mlir::Operation *resolveBlockArgument(mlir::BlockArgument barg, mlir::func::FuncOp caller) {

    const auto argumentNumber = barg.getArgNumber();

    MARCO_DBG() << "Trying to resolve on argument " << argumentNumber << "\n";

    if ( const auto ownerFunc = mlir::dyn_cast<mlir::func::FuncOp>(barg.getOwner()->getParentOp()) ) {
      mlir::SmallVector<CallGraphEdge *> edges;
      findNodeByOp(caller)->findEdgesTo(*findNodeByOp(ownerFunc), edges);

      assert(edges.size() == 1 && "More than one edge to callee with potential bindings");

      auto &bindings = edges[0]->bindings;

      auto *bindingIter = llvm::find_if(bindings, [argumentNumber](FunctionCallMemrefBinding &binding) {
        return binding.argumentIndex == argumentNumber;
      });

      if ( bindingIter == bindings.end() ) {
        return nullptr;
      }

      MARCO_DBG() << "Resolved a binding on argument idx " << argumentNumber << "on function " << (*const_cast<mlir::func::FuncOp *>(&ownerFunc)).getName() << "\n";;

      return bindingIter->originOp;
    }

    return nullptr;
  }


public:
  iterator findNodeByOp(mlir::func::FuncOp funcOp) {
    return llvm::find(nodeStorage, funcOp);
  }

  bool addNodeByFuncOp(mlir::func::FuncOp funcOp) {
    if (findNodeByOp(funcOp) != nodeStorage.end()) {
      return false;
    }
    auto &node = nodeStorage.emplace_back(funcOp);
    Nodes.emplace_back(&node);

    return true;
  }

  void removeNode(mlir::func::FuncOp funcOp) {
    // Remove the pointer first
    iterator storedNode = llvm::find_if(
        nodeStorage, [&](CallGraphNode &n) { return n == funcOp; });
    if (storedNode == nodeStorage.end()) {
      return;
    }

    // Found it, now get its pointer
    this->Base::removeNode(*storedNode);
    llvm::erase(nodeStorage, funcOp);
  }

  bool
  connectEnriched(mlir::func::FuncOp caller, mlir::func::FuncOp callee,
                  mlir::SmallVector<FunctionCallMemrefBinding> &&bindings) {
    // Check if funcOp exists
    addNodeByFuncOp(caller);
    addNodeByFuncOp(callee);

    MARCO_DBG() << "Connecting enriched : " << bindings.size() << " bindings\n";

    auto *first = findNodeByOp(caller);
    auto *second = findNodeByOp(callee);

    CallGraphEdge edge{caller, callee, std::move(bindings)};
    edge.targetNode = second;
    return Base::connect(*first, *second, edge);
  }

  bool connect(mlir::func::FuncOp caller, mlir::func::FuncOp callee) {
    // Check if funcOp exists
    addNodeByFuncOp(caller);
    addNodeByFuncOp(callee);

    auto *first = findNodeByOp(caller);
    auto *second = findNodeByOp(callee);

    CallGraphEdge edge{caller, callee};
    edge.targetNode = second;
    return Base::connect(*first, *second, edge);
  }

  llvm::SmallVector<CallGraphEdge *> getCallers(mlir::func::FuncOp funcOp) {
    // Find the node
    auto *iterNode = llvm::find(nodeStorage, funcOp);
    if (iterNode == nodeStorage.end()) {
      return {};
    }

    llvm::SmallVector<CallGraphEdge *> edges;
    findIncomingEdgesToNode(*iterNode, edges);

    return edges;
  }

  iterator begin() { return nodeStorage.begin(); }
  iterator end() { return nodeStorage.end(); }
  const_iterator begin() const { return nodeStorage.begin(); }
  const_iterator end() const { return nodeStorage.end(); }

  StorageTy nodeStorage;
};

struct CallInfo {
  bool insideLoop = false;
};

using FunctionWritesVec = llvm::SmallVector<DRWrite, 4>;
using FunctionWritesMap = llvm::DenseMap<mlir::Operation *, FunctionWritesVec>;

struct CallGraphDiagnostics {
  static void outputDiagnostic(DiagnosticEngine &engine, CallGraph &callGraph) {
    for (auto &funcNode : callGraph) {
      for (auto &callSite : funcNode.edges) {
        auto diag = callSite.caller->emitRemark();
        diag << callSite.caller.getName() << "," << callSite.callee.getName()
             << "," << callSite.bindings.size();
      }
    }
  }
};

}; // namespace mlir::bmodelica::detail

namespace llvm {
template <>
struct DenseMapInfo<::mlir::bmodelica::detail::DRLoad> {
  using KeyT = ::mlir::bmodelica::detail::DRLoad;
  static constexpr mlir::Operation *getEmptyKey() { return nullptr; }
  static constexpr mlir::Operation *getTombstoneKey() {
    return std::numeric_limits<mlir::Operation *>::max();
  }
  static unsigned getHashValue(const KeyT &val) {
    return DenseMapInfo<mlir::Operation *>::getHashValue(
        static_cast<mlir::Operation *>(val));
  }
  static bool isEqual(const KeyT &lhs, const KeyT &rhs) {
    return static_cast<mlir::Operation *>(lhs) ==
           static_cast<mlir::Operation *>(rhs);
  }
};

template <>
struct DenseMapInfo<::mlir::bmodelica::detail::DRWrite> {
  using KeyT = ::mlir::bmodelica::detail::DRWrite;
  static constexpr mlir::Operation *getEmptyKey() { return nullptr; }
  static constexpr mlir::Operation *getTombstoneKey() {
    return std::numeric_limits<mlir::Operation *>::max();
  }
  static unsigned getHashValue(const KeyT &val) {
    return DenseMapInfo<mlir::Operation *>::getHashValue(
        static_cast<mlir::Operation *>(val));
  }
  static bool isEqual(const KeyT &lhs, const KeyT &rhs) {
    return static_cast<mlir::Operation *>(lhs) ==
           static_cast<mlir::Operation *>(rhs);
  }
};

template <>
struct DenseMapInfo<::mlir::bmodelica::detail::DRAlloc> {
  using KeyT = ::mlir::bmodelica::detail::DRAlloc;
  static constexpr mlir::Operation *getEmptyKey() { return nullptr; }
  static constexpr mlir::Operation *getTombstoneKey() {
    return std::numeric_limits<mlir::Operation *>::max();
  }
  static unsigned getHashValue(const KeyT &val) {
    return DenseMapInfo<mlir::Operation *>::getHashValue(
        static_cast<mlir::Operation *>(val));
  }
  static bool isEqual(const KeyT &lhs, const KeyT &rhs) {
    return static_cast<mlir::Operation *>(lhs) ==
           static_cast<mlir::Operation *>(rhs);
  }
};
} // namespace llvm

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

  mlir::LogicalResult findOpportunitiesAux(
      mlir::ModuleOp moduleOp, mlir::MLIRContext *context,
      mlir::func::FuncOp funcOp, mlir::SymbolTableCollection &symTabCollection,
      CallGraph &callGraph, llvm::SmallVector<std::pair<DRWrite, DRLoad>, 4> &foundOpportunities,
      llvm::SmallSet<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 8>
          &visitSet);

  mlir::FailureOr<llvm::SmallVector<std::pair<DRWrite, DRLoad>, 4>>
  findOpportunities(mlir::ModuleOp moduleOp, mlir::MLIRContext *context,
                        mlir::func::FuncOp entrypointFuncOp,
                        mlir::SymbolTableCollection &symTabCollection, CallGraph &callGraph);

  mlir::FailureOr<mlir::func::FuncOp> selectEntrypoint(mlir::ModuleOp moduleOp);

  ::mlir::bmodelica::detail::CallGraph
  buildCallGraph(mlir::MLIRContext *context, mlir::ModuleOp moduleOp,
                 mlir::func::FuncOp entrypointFuncOp,
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
    // TODO(Tor): This is incomplete.
    return {};
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

  /// Whether to output diagnostics for tests
  const bool outputDiagnostics = drTestDiagnostics.getValue();

  // Gather all global statics
  moduleOp = this->getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::SymbolTableCollection symTabCollection{};

  auto &diagnostics = context->getDiagEngine();

  DRLoadTracer loadTracer{context, moduleOp, symTabCollection};
  DRStoreTracer storeTracer{context, moduleOp, symTabCollection};

  // IMPORTANT NOTE:
  // This selection is for an IPO-analysis. In the final pass,
  // the selection should not be informed by a _dynamic prefix, but rather
  // something more general.
  //
  // Later idea: Disjoint set analysis. Refine and split clusters based on
  // feasibility. Can be costly, so amortized and cached information necessary.

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

    llvm::SmallVector<int64_t, 4> stridesAfter;
    int64_t offsetAfter;
  };

  if ( ! entrypointFuncOp ) {
    signalPassFailure();
    return;
  }

  auto callGraph =
      buildCallGraph(context, moduleOp, entrypointFuncOp, symTabCollection);

  auto opportunities = findOpportunities(
      moduleOp, context, entrypointFuncOp, symTabCollection, callGraph);

  if ( mlir::failed(opportunities) ) {
    MARCO_DBG() << "findOpportunities failed\n";
  }
  MARCO_DBG() << "Found " << opportunities->size() << " opportunities\n";

  if (outputDiagnostics) {
    CallGraphDiagnostics::outputDiagnostic(diagnostics, callGraph);
  }

  // moduleOp.walk([&](mlir::memref::StoreOp storeOp) {
  //   auto [chain, originOp] = storeTracer.trace(storeOp);
  //   memrefStoreToOriginOp.insert(std::make_pair(storeOp, originOp));
  //   memrefStoreToOriginChain.insert(std::make_pair(storeOp, std::move(chain)));

  //   // Walk the chain and use the operands involved to build the expression.
  //   // What happens when the stride is simply reset with reinterpret_cast?

  //   llvm::DenseMap<mlir::memref::ReinterpretCastOp, Accessor>
  //       reinterpretExpressions;

  //   // NOTE(TOR): DEBUG OUTPUT
  //   // if (auto globalOp = mlir::dyn_cast<mlir::memref::GlobalOp>(
  //   //         memrefStoreToOriginOp[storeOp])) {
  //   //   MARCO_DBG() << "Traced a memref back to " << globalOp.getName() <<
  //   //   "\n";
  //   // }

  //   for (auto &[storeOp, chain] : memrefStoreToOriginChain) {
  //     for (auto *op : chain) {
  //       if (mlir::memref::ReinterpretCastOp reinterpretCastOp =
  //               mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(op)) {
  //         Accessor accessorExpression;

  //         auto range = reinterpretCastOp.getOffsets();

  //         if (range.empty() || range.size() > 1) {
  //           continue; // Do not handle
  //           // TODO: Handle multiple offsets
  //         }

  //         auto val = range[0];
  //         auto *definingOp = val.getDefiningOp();

  //         /// Possibly a BlockArgument
  //         if (definingOp == nullptr) {
  //           // Try to cast it
  //         }

  //         mlir::FailureOr<IndexExpressionNode> result =
  //             mlir::TypeSwitch<mlir::Operation *,
  //                              mlir::FailureOr<IndexExpressionNode>>(definingOp)
  //                 .Case<mlir::arith::AddIOp>([](mlir::arith::AddIOp addIOp) {
  //                   return IndexExpressionNode{};
  //                 })
  //                 .Default([](mlir::Operation *unhandledOp) {
  //                   return mlir::failure();
  //                 });
  //       }
  //     }
  //   }
  // });

  llvm::DenseMap<mlir::affine::AffineStoreOp, mlir::Operation *>
      affineStoreToOriginOp{};

  llvm::DenseMap<mlir::affine::AffineStoreOp,
                 llvm::SmallVector<mlir::Operation *, 4>>
      affineStoreToOriginChain{};

  // moduleOp.walk([&](mlir::affine::AffineStoreOp storeOp) {
  //   auto [chain, originOp] = storeTracer.trace(storeOp);
  //   affineStoreToOriginOp.insert(std::make_pair(storeOp, originOp));
  //   affineStoreToOriginChain.insert(std::make_pair(storeOp, std::move(chain)));
  // });

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

    // If simply main
    if (nameStr.find("main") == 0) {
      entrypointFuncs.emplace_back(funcOp);
    }

    if (nameStr.find("updateNonStateVariables") == 0) {
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
DataRecomputationPass::findOpportunities(
    mlir::ModuleOp moduleOp, mlir::MLIRContext *context,
    mlir::func::FuncOp entrypointFuncOp,
    mlir::SymbolTableCollection &symTabCollection, CallGraph &callGraph) {
  // Get all function calls, stores, and loads in here. Keep track of last
  // write.

  llvm::SmallVector<std::pair<DRWrite, DRLoad>, 4> result{};
  llvm::SmallSet<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 8>
      visitSet{};
  auto report = findOpportunitiesAux(moduleOp, context, entrypointFuncOp,
                                         symTabCollection, callGraph, result, visitSet);

  if (report.failed())
    return mlir::failure();
  return result;
}

mlir::LogicalResult DataRecomputationPass::findOpportunitiesAux(
    mlir::ModuleOp moduleOp, mlir::MLIRContext *context,
    mlir::func::FuncOp funcOp, mlir::SymbolTableCollection &symTabCollection,
    CallGraph &callGraph, llvm::SmallVector<std::pair<DRWrite, DRLoad>, 4> &foundOpportunities,
    llvm::SmallSet<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 8>
        &visitSet) {

  llvm::DenseMap</*OriginOp*/ mlir::Operation *, DRWrite> lastWrites{};
  llvm::DenseMap<DRWrite, llvm::SmallVector<DRLoad, 4>> candidateLoads{};

  // Keep track of how deep we're going at the point of visit
  llvm::SmallVector<mlir::func::FuncOp, 4> callStack{};
  // Build a stack of visits
  llvm::SmallVector<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 4>
      visitStack{};
  // Mark it as a self-visit.
  visitStack.emplace_back(std::make_pair(funcOp, funcOp));

  while (!visitStack.empty()) {

    auto poppedElement = visitStack.back();
    visitStack.pop_back();

    auto inboundFuncOp = poppedElement.first;
    auto currentFuncOp = poppedElement.second;

    if (visitSet.contains(std::make_pair(inboundFuncOp, currentFuncOp))) {
      continue; // Exactly this avenue has already been explored
    }

    llvm::SmallVector<mlir::Operation *, 8> nestedCallOps{};

    currentFuncOp.walk([&](mlir::Operation *op) {
      // Collect all nested calls
      if (mlir::isa<mlir::CallOpInterface>(op)) {
        op->dump();
        nestedCallOps.emplace_back(op);
      }

      // Collect writes
      if (DRWrite::isStoreOp(op)) {

        mlir::Operation *resolvedOriginOp = nullptr;

        auto [resChain, traceResult] =
            DRStoreTracer::trace(op, context, moduleOp, symTabCollection);

        if ( auto valueOpt = traceResult.getValue() ) {
          auto value = valueOpt.value();
          if ( auto arg = mlir::dyn_cast<mlir::BlockArgument>(value) ) {
            MARCO_DBG() << "RESOLVING STORE ARGUMENT MEMREF\n";
            resolvedOriginOp = callGraph.resolveBlockArgument(arg, inboundFuncOp);
          }

        } else if ( auto opOpt = traceResult.getOperation() ){
          resolvedOriginOp = opOpt.value();
        } else if (traceResult.isFailure()) {
          MARCO_DBG() << "Tracing returned a Failure Marker!";
        }

        assert(resolvedOriginOp != nullptr && "Unable to resolve bound memref");

        lastWrites.insert(std::make_pair(resolvedOriginOp, DRWrite{op}));
      }

      if (DRLoad::isLoadOp(op)) {
        // Trace it back
        auto [resChain, originOp] = DRLoadTracer::trace(op, context,
        moduleOp, symTabCollection);

        assert(!originOp.isFailure() && "Failed to trace back load");

        mlir::Operation *resolvedOriginOp = nullptr;

        if ( auto valueOpt = originOp.getValue() ) {
          auto value = valueOpt.value();
          if ( auto arg = mlir::dyn_cast<mlir::BlockArgument>(value) ) {
            MARCO_DBG() << "RESOLVING LOAD ARGUMENT MEMREF\n";
            resolvedOriginOp = callGraph.resolveBlockArgument(arg, inboundFuncOp);
          }
        } else if ( auto opOpt = originOp.getOperation() ) {
          resolvedOriginOp = opOpt.value();
        }
        // Try to find
        if ( ! lastWrites.contains(resolvedOriginOp) ) {
          // Skip this one. Not a candidate
          return;
        }

        foundOpportunities.emplace_back(lastWrites.at(resolvedOriginOp), op);
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
      auto callableSymbol = callOpInterface.getCallableForCallee()
                          .dyn_cast<mlir::SymbolRefAttr>();


      auto *callableOp = symTabCollection.lookupSymbolIn(moduleOp, callableSymbol);

      if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(callableOp) ) {
        visitStack.push_back( std::make_pair(currentFuncOp, funcOp) );
      }



      // debug output
      if (insideLoop) {
        MARCO_DBG() << "Nested function call is inside a loop!\n";
        llvm::dbgs() << "Loop-nested function calls are not handled yet"
                     << "\n";
        signalPassFailure();
        return mlir::failure();
        // llvm_unreachable("Loop-bound function calls are not handled");
      }

      MARCO_DBG() << "Nested function call to " << callableSymbol
                  << " is NOT inside a loop!\n";
    }
  }

  return mlir::success();
}

::mlir::bmodelica::detail::CallGraph DataRecomputationPass::buildCallGraph(
    mlir::MLIRContext *context, mlir::ModuleOp moduleOp,
    mlir::func::FuncOp entrypointFuncOp,
    mlir::SymbolTableCollection &symTabCollection) {
  llvm::SmallSet<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 8>
      visitedSet{};
  llvm::SmallVector<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>>
      visitStack{};

  visitStack.emplace_back(std::make_pair(entrypointFuncOp, entrypointFuncOp));

  ::mlir::bmodelica::detail::CallGraph callGraph;

  while (!visitStack.empty()) {
    auto currentCallSite = visitStack.back();
    auto &callerFuncOp = currentCallSite.first;
    auto &calleeFuncOp = currentCallSite.second;
    visitStack.pop_back();

    MARCO_DBG() << "Handling arc from " << callerFuncOp.getName() << " to "
                << calleeFuncOp.getName() << "\n";
    // Skip the already visited function
    if (visitedSet.contains(std::make_pair(callerFuncOp, calleeFuncOp))) {
      MARCO_DBG() << "Arc already seen! Skipping iteration\n";
      continue;
    }

    // Mark visited
    visitedSet.insert(std::make_pair(callerFuncOp, calleeFuncOp));
    callGraph.addNodeByFuncOp(callerFuncOp);

    calleeFuncOp.walk([&](mlir::CallOpInterface callOpInterface) {
      auto callable = callOpInterface.getCallableForCallee();

      // TODO(Tor): Investigate parameter list
      if (auto calleeSym = callable.dyn_cast<mlir::SymbolRefAttr>()) {
        mlir::Operation *calleeOp =
            symTabCollection.lookupSymbolIn(moduleOp, calleeSym);

        if (auto nestedCalleeFuncOp =
                mlir::dyn_cast<mlir::func::FuncOp>(calleeOp)) {

          auto bindings = DRFunctionArgumentBinder::findBindings(
              callOpInterface, moduleOp, context, symTabCollection);

          MARCO_DBG() << "Adding call to CallGraph from"
                      << callerFuncOp.getName() << " to "
                      << nestedCalleeFuncOp.getName() << " with "
                      << bindings.size() << " bindings\n";
          callGraph.connectEnriched(calleeFuncOp, nestedCalleeFuncOp,
                                    std::move(bindings));

          // TODO(Tor): Debug why the same arcs are investigated multiple times.
          MARCO_DBG() << "Pushing (" << calleeFuncOp.getName() << ", "
                      << nestedCalleeFuncOp.getName() << ") onto the stack\n";
          visitStack.push_back(
              std::make_pair(calleeFuncOp, nestedCalleeFuncOp));
        } else if (calleeOp->getName().getStringRef() ==
                   llvm::StringRef{"runtime.function"}) {
          // TODO(Tor): This is incredibly hacky. For the time being, add the
          // runtime dialect as a dependency dialect and try to cast to the
          // runtime.function op MARCO_DBG() << "Skipping runtime.func" << "\n";
        } else {
          llvm_unreachable("Couldn't handle a callee when building call graph");
        }
      }
    });
  }

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
//
// template <class Pass>
// class PassContextBundle {
// public:
//   PassContextBundle(Pass &pass)
//     : context{pass.getContext()},
//       symTabCollection{}
//   {
//
//   }
//
//   mlir::MLIRContext *getContext() { return context; }
//   mlir::SymbolTableCollection &getSymTabCollection() { return
//   symTabCollection; }
//
// private:
//   mlir::MLIRContext *context = nullptr;
//   mlir::SymbolTableCollection symTabCollection;
// };
// //
