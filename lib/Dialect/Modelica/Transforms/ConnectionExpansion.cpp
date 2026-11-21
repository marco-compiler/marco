#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion.h"
#include "marco/Dialect/Modelica/IR/ConstantFolding.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/AccessFunctionUtils.h"
#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionEquations.h"
#include "marco/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionGraph.h"
#include "marco/Dialect/Modelica/Transforms/Evaluation.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/PatternMatch.h"
#include <map>

namespace mlir::modelica {
#define GEN_PASS_DEF_CONNECTIONEXPANSIONPASS
#include "marco/Dialect/Modelica/Transforms/Passes.h.inc"
} // namespace mlir::modelica

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::connection;

namespace {
/// Primitive variable of the instance tree.
struct Variable {
  mutable VariableOp op;

  /// The operations from the model to the variable.
  llvm::SmallVector<mlir::Operation *> path;

  /// Number of dimensions of each operation of the path.
  llvm::SmallVector<unsigned> ranks;

  /// Dimensions of the enclosing instances followed by the ones of the
  /// variable.
  llvm::SmallVector<int64_t> shape;
};

/// Connector referenced by a connect-equation: the instance or variable, and
/// the indices of the dimensions of the operations from the model to it, as
/// affine functions of the equation indices. The equation indices are the
/// iterators of the enclosing for-equations, followed by one dimension for
/// each dimension of the connector that is not selected by a scalar index.
struct ConnectorAccess {
  mlir::Operation *op;
  llvm::SmallVector<mlir::AffineExpr> indices;

  /// Sizes of the dimensions not selected by a scalar index.
  llvm::SmallVector<int64_t> sliceSizes;

  /// Whether the sizes of some slices depend on the iterators, and are known
  /// only once the iterators have a value.
  bool hasDependentSizes = false;
};

/// Primitive variable within a connector, identified by its path relative
/// to the connector.
struct Leaf {
  std::string relativePath;
  unsigned variable;

  /// Number of the dimensions belonging to the connector.
  unsigned numPrefixDims;
};

/// Indices along a dimension of the equations: a list of disjoint ranges.
// FIX: Use MultidimensionalRange
using DimensionRanges = llvm::SmallVector<marco::modeling::Range, 1>;

/// Indices of the iterators of nested for-equations, possibly followed by the
/// sliced dimensions of the connectors: the combinations of the ranges of each
/// dimension.
using LoopBox = llvm::SmallVector<DimensionRanges, 3>;

class ConnectionExpander {
public:
  explicit ConnectionExpander(ModelOp modelOp)
      : modelOp(modelOp), graph(modelOp.getContext()) {}

  mlir::LogicalResult run();

private:
  // FIX: Add a comment describing what the method does
  void collectVariables(mlir::Operation *op,
                        llvm::SmallVectorImpl<mlir::Operation *> &path);

  // FIX: Add a comment describing what the method does
  mlir::LogicalResult processConnect(ConnectOp connectOp);

  // FIX: Add a comment describing what the method does
  mlir::LogicalResult getLoopIndices(llvm::ArrayRef<ForEquationOp> loops,
                                     llvm::SmallVectorImpl<LoopBox> &boxes);

  /// Resolve a connector. The sizes of the slices are computed by the
  /// evaluator knowing the values of the iterators, if given.
  mlir::LogicalResult resolveConnector(ConnectOp connectOp, mlir::Value value,
                                       llvm::ArrayRef<mlir::Value> iterators,
                                       Evaluator &evaluator,
                                       Evaluator *iteratorsEvaluator,
                                       ConnectorAccess &access);

  // FIX: Add a comment describing what the method does
  llvm::ArrayRef<Leaf> getLeaves(mlir::Operation *op);

  /// Get the vertex of a primitive variable, creating it if needed. Return
  /// std::nullopt if the variable has no elements.
  std::optional<VertexId> getVertex(unsigned variable, bool outside);

  // FIX: Add a comment describing what the method does
  mlir::LogicalResult addEdge(ConnectOp connectOp,
                              llvm::ArrayRef<LoopBox> connectBoxes,
                              llvm::ArrayRef<DimensionRanges> dimensions,
                              unsigned lhsVariable, bool lhsOutside,
                              llvm::ArrayRef<mlir::AffineExpr> lhsIndices,
                              unsigned rhsVariable, bool rhsOutside,
                              llvm::ArrayRef<mlir::AffineExpr> rhsIndices);

  // FIX: Add a comment describing what the method does
  void emitEquations(const ConnectionEquations &equations);

  // FIX: Add a comment describing what the method does
  mlir::Value emitReference(mlir::OpBuilder &builder, mlir::Location loc,
                            unsigned variable,
                            llvm::ArrayRef<mlir::Value> indices);

  /// Emit the indices of a vertex given by an access function, whose
  /// dimensions are replaced by the given affine expressions of the
  /// iterators.
  llvm::SmallVector<mlir::Value>
  emitIndices(mlir::OpBuilder &builder, mlir::Location loc, VertexId vertex,
              const marco::modeling::AccessFunction &accessFunction,
              llvm::ArrayRef<mlir::AffineExpr> replacements,
              llvm::ArrayRef<mlir::Value> iterators);

private:
  ModelOp modelOp;
  llvm::SmallVector<Variable> variables;
  llvm::DenseMap<mlir::Operation *, unsigned> variableIds;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Leaf>> leaves;

  ConnectionGraph graph;
  std::map<std::pair<unsigned, bool>, VertexId> vertexIds;

  /// Elements of the outside vertices reached by the connect-equations. The
  /// other ones don't belong to any connection set.
  llvm::DenseMap<VertexId, marco::modeling::IndexSet> reachedIndices;
};

/// Get the dimensions of an instance or variable.
llvm::ArrayRef<int64_t> getOpShape(mlir::Operation *op) {
  if (auto instanceOp = mlir::dyn_cast<InstanceOp>(op)) {
    return instanceOp.getShape();
  }

  auto variableOp = mlir::cast<VariableOp>(op);

  if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(variableOp.getType())) {
    return shapedType.getShape();
  }

  return {};
}

mlir::Type getOpType(mlir::Operation *op) {
  if (auto instanceOp = mlir::dyn_cast<InstanceOp>(op)) {
    return instanceOp.getType();
  }

  return mlir::cast<VariableOp>(op).getType();
}

llvm::StringRef getOpName(mlir::Operation *op) {
  return mlir::SymbolTable::getSymbolName(op).getValue();
}

mlir::Operation *lookupChild(mlir::Operation *parent, llvm::StringRef name) {
  return mlir::SymbolTable::lookupSymbolIn(parent, name);
}

/// Get the index computed by an expression as an affine function of the
/// iterators, if it is one. The expressions not depending on the iterators
/// are evaluated.
std::optional<mlir::AffineExpr>
getAffineIndex(mlir::Value index, llvm::ArrayRef<mlir::Value> iterators,
               Evaluator &evaluator) {
  mlir::MLIRContext *context = index.getContext();

  if (const auto *it = llvm::find(iterators, index); it != iterators.end()) {
    return mlir::getAffineDimExpr(it - iterators.begin(), context);
  }

  if (mlir::Attribute value = evaluator.evaluate(index)) {
    if (std::optional<int64_t> position = getIndexValue(value)) {
      return mlir::getAffineConstantExpr(*position, context);
    }

    return std::nullopt;
  }

  mlir::Operation *op = index.getDefiningOp();

  if (!op || !mlir::isa<AddOp, SubOp, NegateOp, MulOp>(op)) {
    return std::nullopt;
  }

  llvm::SmallVector<mlir::AffineExpr, 2> operands;

  for (mlir::Value operand : op->getOperands()) {
    std::optional<mlir::AffineExpr> expr =
        getAffineIndex(operand, iterators, evaluator);

    if (!expr) {
      return std::nullopt;
    }

    operands.push_back(*expr);
  }

  if (mlir::isa<AddOp>(op)) {
    return operands[0] + operands[1];
  }

  if (mlir::isa<SubOp>(op)) {
    return operands[0] - operands[1];
  }

  if (mlir::isa<NegateOp>(op)) {
    return -operands[0];
  }

  if (!mlir::isa<mlir::AffineConstantExpr>(operands[0]) &&
      !mlir::isa<mlir::AffineConstantExpr>(operands[1])) {
    return std::nullopt;
  }

  return operands[0] * operands[1];
}

/// Check whether a value depends on the given ones.
bool dependsOn(mlir::Value value, llvm::ArrayRef<mlir::Value> values) {
  if (llvm::is_contained(values, value)) {
    return true;
  }

  mlir::Operation *op = value.getDefiningOp();

  return op && llvm::any_of(op->getOperands(), [&](mlir::Value operand) {
           return dependsOn(operand, values);
         });
}

/// Append a sequence of positions to the ranges of a dimension, merging the
/// consecutive ones.
void appendPosition(DimensionRanges &ranges, int64_t position) {
  if (!ranges.empty() && ranges.back().getEnd() == position) {
    ranges.back() =
        marco::modeling::Range(ranges.back().getBegin(), position + 1);
  } else {
    ranges.emplace_back(position, position + 1);
  }
}

void ConnectionExpander::collectVariables(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::Operation *> &path) {
  for (mlir::Operation &nestedOp : op->getRegion(0).getOps()) {
    if (auto instanceOp = mlir::dyn_cast<InstanceOp>(nestedOp)) {
      path.push_back(instanceOp);
      collectVariables(instanceOp, path);
      path.pop_back();
      continue;
    }

    if (auto variableOp = mlir::dyn_cast<VariableOp>(nestedOp)) {
      Variable &variable = variables.emplace_back();
      variable.op = variableOp;
      variable.path.assign(path.begin(), path.end());
      variable.path.push_back(variableOp);

      for (mlir::Operation *pathOp : variable.path) {
        llvm::ArrayRef<int64_t> shape = getOpShape(pathOp);
        variable.ranks.push_back(shape.size());
        variable.shape.append(shape.begin(), shape.end());
      }

      variableIds[variableOp] = variables.size() - 1;
    }
  }
}

llvm::ArrayRef<Leaf> ConnectionExpander::getLeaves(mlir::Operation *op) {
  if (auto it = leaves.find(op); it != leaves.end()) {
    return it->second;
  }

  llvm::SmallVector<Leaf> result;

  if (auto variableOp = mlir::dyn_cast<VariableOp>(op)) {
    // The indices of the connector cover all the dimensions of the variable.
    unsigned id = variableIds.lookup(variableOp);
    result.push_back(
        Leaf{"", id, static_cast<unsigned>(variables[id].shape.size())});
  } else {
    for (mlir::Operation &nested : op->getRegion(0).front()) {
      if (!mlir::isa<InstanceOp, VariableOp>(nested)) {
        continue;
      }

      for (const Leaf &leaf : getLeaves(&nested)) {
        Leaf &newLeaf = result.emplace_back(leaf);
        newLeaf.relativePath =
            getOpName(&nested).str() +
            (leaf.relativePath.empty() ? "" : "." + leaf.relativePath);
      }
    }

    // The dimensions of the connector are the ones of the path up to it.
    unsigned prefix = 0;

    if (!result.empty()) {
      const Variable &variable = variables[result.front().variable];

      for (auto [pathOp, rank] :
           llvm::zip_equal(variable.path, variable.ranks)) {
        prefix += rank;

        if (pathOp == op) {
          break;
        }
      }
    }

    for (Leaf &leaf : result) {
      leaf.numPrefixDims = prefix;
    }
  }

  return leaves[op] = std::move(result);
}

std::optional<VertexId> ConnectionExpander::getVertex(unsigned variable,
                                                      bool outside) {
  auto key = std::make_pair(variable, outside);

  if (auto it = vertexIds.find(key); it != vertexIds.end()) {
    return it->second;
  }

  // Scalar variables have a single element, with index 0.
  llvm::SmallVector<marco::modeling::Range, 3> ranges;

  for (int64_t size : variables[variable].shape) {
    if (size <= 0) {
      return std::nullopt;
    }

    ranges.emplace_back(0, size);
  }

  if (ranges.empty()) {
    ranges.emplace_back(0, 1);
  }

  Vertex vertex;
  vertex.variable = variable;
  vertex.outside = outside;

  vertex.isFlow =
      variables[variable].op.getConnectorKind() == ConnectorKind::flow;

  vertex.indices =
      marco::modeling::IndexSet(marco::modeling::MultidimensionalRange(ranges));

  VertexId id = graph.addVertex(std::move(vertex));
  vertexIds[key] = id;
  return id;
}

/// Get the range yielded by a for-equation, if it is explicit.
static mlir::Value getLoopRange(ForEquationOp loop) {
  if (loop.hasImplicitRange()) {
    return nullptr;
  }

  return mlir::cast<YieldOp>(loop.getRangeRegion().front().getTerminator())
      .getValues()[0];
}

/// Evaluate the range of a for-equation. Return false if it can't be
/// evaluated.
// FIX: Can this maybe be part of the Interpreter class?
static bool evaluateLoopRange(ForEquationOp loop, Evaluator &evaluator,
                              DimensionRanges &ranges) {
  mlir::Value range = getLoopRange(loop);

  if (!range) {
    return false;
  }

  // The ranges are evaluated without enumerating their values, when
  // possible.
  if (auto rangeOp = range.getDefiningOp<RangeOp>()) {
    mlir::Attribute startValue = evaluator.evaluate(rangeOp.getStart());
    mlir::Attribute stopValue = evaluator.evaluate(rangeOp.getStop());

    std::optional<int64_t> start =
        startValue ? getIndexValue(startValue) : std::nullopt;

    std::optional<int64_t> stop =
        stopValue ? getIndexValue(stopValue) : std::nullopt;

    std::optional<int64_t> step = 1;

    if (rangeOp.getStep()) {
      mlir::Attribute stepValue = evaluator.evaluate(rangeOp.getStep());
      step = stepValue ? getIntegerValue(stepValue) : std::nullopt;
    }

    if (!start || !stop || !step || *step == 0) {
      return false;
    }

    if (*step == 1) {
      if (*start <= *stop) {
        ranges.emplace_back(*start, *stop + 1);
      }

      return true;
    }

    // TODO: keep compact once ranges with a step are supported.
    for (int64_t value = *start; *step > 0 ? value <= *stop : value >= *stop;
         value += *step) {
      appendPosition(ranges, value);
    }

    return true;
  }

  mlir::Attribute values = evaluator.evaluate(range);

  if (!values) {
    return false;
  }

  // TODO: keep compact once ranges with a step are supported.
  for (mlir::Attribute value : getConstantElements(values)) {
    std::optional<int64_t> position = getIndexValue(value);

    if (!position) {
      return false;
    }

    appendPosition(ranges, *position);
  }

  return true;
}

/// Enumerate the combinations of values of the iterators of the first
/// 'numOfEnumerated' for-equations. The callback gets an evaluator knowing
/// their values, and the box made of the single point they form.
static mlir::LogicalResult enumerateIterators(
    llvm::ArrayRef<ForEquationOp> loops, size_t numOfEnumerated,
    llvm::function_ref<mlir::LogicalResult(Evaluator &, const LoopBox &)>
        callback) {
  llvm::SmallVector<mlir::Attribute> values;
  LoopBox outerBox;

  std::function<mlir::LogicalResult(size_t)> enumerate =
      [&](size_t depth) -> mlir::LogicalResult {
    // A new evaluator is used for each combination of values, as the
    // evaluated values are cached.
    Evaluator evaluator;

    for (auto [i, value] : llvm::enumerate(values)) {
      evaluator.setValue(ForEquationOp(loops[i]).getInductionVariable(), value);
    }

    if (depth == numOfEnumerated) {
      return callback(evaluator, outerBox);
    }

    ForEquationOp loop = loops[depth];

    auto emitRangeError = [&]() {
      return loop.emitError("can't evaluate the range of the for-equation "
                            "containing connect-equations");
    };

    mlir::Value range = getLoopRange(loop);
    mlir::Attribute rangeValue = range ? evaluator.evaluate(range) : nullptr;

    if (!rangeValue) {
      return emitRangeError();
    }

    for (mlir::Attribute value : getConstantElements(rangeValue)) {
      std::optional<int64_t> position = getIndexValue(value);

      if (!position) {
        return emitRangeError();
      }

      values.push_back(value);
      outerBox.push_back({marco::modeling::Range(*position, *position + 1)});

      if (mlir::failed(enumerate(depth + 1))) {
        return mlir::failure();
      }

      values.pop_back();
      outerBox.pop_back();
    }

    return mlir::success();
  };

  return enumerate(0);
}

mlir::LogicalResult
ConnectionExpander::getLoopIndices(llvm::ArrayRef<ForEquationOp> loops,
                                   llvm::SmallVectorImpl<LoopBox> &boxes) {
  // Ranges not depending on the iterators of the enclosing for-equations
  // form a single box.
  {
    Evaluator evaluator;
    LoopBox box;

    if (llvm::all_of(loops, [&](ForEquationOp loop) {
          return evaluateLoopRange(loop, evaluator, box.emplace_back());
        })) {
      boxes.push_back(std::move(box));
      return mlir::success();
    }
  }

  // Otherwise, the ranges are evaluated for each combination of values of
  // the iterators of the enclosing for-equations, which are enumerated.
  // TODO: keep compact, e.g. by representing the ranges symbolically in
  // terms of the iterators.
  ForEquationOp innermost = loops.back();

  return enumerateIterators(
      loops, loops.size() - 1,
      [&](Evaluator &evaluator,
          const LoopBox &outerBox) -> mlir::LogicalResult {
        LoopBox box(outerBox);

        if (!evaluateLoopRange(innermost, evaluator, box.emplace_back())) {
          return innermost.emitError("can't evaluate the range of the "
                                     "for-equation containing "
                                     "connect-equations");
        }

        if (!box.back().empty()) {
          boxes.push_back(std::move(box));
        }

        return mlir::success();
      });
}

mlir::LogicalResult ConnectionExpander::resolveConnector(
    ConnectOp connectOp, mlir::Value value,
    llvm::ArrayRef<mlir::Value> iterators, Evaluator &evaluator,
    Evaluator *iteratorsEvaluator, ConnectorAccess &access) {
  llvm::SmallVector<mlir::Operation *> chain;
  mlir::Value current = value;

  while (mlir::Operation *definingOp = current.getDefiningOp()) {
    chain.push_back(definingOp);

    if (auto memberOp = mlir::dyn_cast<MemberOp>(definingOp)) {
      current = memberOp.getObject();
    } else if (auto subscriptOp = mlir::dyn_cast<SubscriptOp>(definingOp)) {
      current = subscriptOp.getArray();
    } else if (mlir::isa<RefOp>(definingOp)) {
      break;
    } else {
      return mlir::emitError(value.getLoc()) << "invalid connector reference";
    }
  }

  std::reverse(chain.begin(), chain.end());

  if (chain.empty() || !mlir::isa<RefOp>(chain.front()) ||
      !mlir::cast<RefOp>(chain.front()).hasSimpleName()) {
    return mlir::emitError(value.getLoc()) << "invalid connector reference";
  }

  mlir::MLIRContext *context = value.getContext();

  auto emitNonAffineError = [&]() {
    // TODO: enumerate the indices once Modelica expressions can be
    // evaluated.
    return connectOp.emitError("non-affine indices in connect-equations are "
                               "not supported yet");
  };

  // Indices selected along each dimension of the last operation of the
  // path: either a scalar index, or a slice made of 'size' indices starting
  // from 'start' with the given step. A missing selection denotes the whole
  // dimension.
  struct Selection {
    mlir::AffineExpr start;
    int64_t step = 0;
    std::optional<int64_t> size;
  };

  mlir::Operation *lastOp = nullptr;
  llvm::SmallVector<std::optional<Selection>> selections;

  // Convert the selections of the last operation of the path to indices.
  auto flushLevel = [&]() {
    if (!lastOp) {
      return;
    }

    for (auto [size, selection] :
         llvm::zip_equal(getOpShape(lastOp), selections)) {
      if (selection && !selection->size) {
        access.indices.push_back(selection->start);
        continue;
      }

      mlir::AffineExpr dimension = mlir::getAffineDimExpr(
          iterators.size() + access.sliceSizes.size(), context);

      if (!selection) {
        access.indices.push_back(dimension);
        access.sliceSizes.push_back(size);
        continue;
      }

      access.indices.push_back(dimension * selection->step + selection->start);
      access.sliceSizes.push_back(*selection->size);
    }
  };

  auto addLevel = [&](mlir::Operation *op) {
    flushLevel();
    lastOp = op;
    selections.assign(getOpShape(op).size(), std::nullopt);
  };

  bool subscripted = false;

  for (mlir::Operation *op : chain) {
    if (auto refOp = mlir::dyn_cast<RefOp>(op)) {
      mlir::Operation *child = lookupChild(modelOp, refOp.getSimpleName());

      if (!child) {
        return refOp.emitError("unknown component '") << refOp.getName() << "'";
      }

      addLevel(child);
      subscripted = false;
      continue;
    }

    if (auto memberOp = mlir::dyn_cast<MemberOp>(op)) {
      mlir::Operation *child = mlir::isa<InstanceOp>(lastOp)
                                   ? lookupChild(lastOp, memberOp.getMember())
                                   : nullptr;

      if (!child) {
        return memberOp.emitError("unknown component '")
               << memberOp.getMember() << "'";
      }

      addLevel(child);
      subscripted = false;
      continue;
    }

    auto subscriptOp = mlir::cast<SubscriptOp>(op);

    if (subscripted) {
      return mlir::emitError(value.getLoc()) << "invalid connector reference";
    }

    subscripted = true;
    auto index = subscriptOp.getIndices().begin();

    for (auto [dim, isColon] : llvm::enumerate(subscriptOp.getColons())) {
      if (isColon) {
        continue;
      }

      mlir::Value indexValue = *index++;

      if (!mlir::isa<mlir::ShapedType>(indexValue.getType())) {
        std::optional<mlir::AffineExpr> expr =
            getAffineIndex(indexValue, iterators, evaluator);

        if (!expr) {
          return emitNonAffineError();
        }

        selections[dim] = Selection{*expr, 0, std::nullopt};
        continue;
      }

      // Slices given by ranges.
      if (auto rangeOp = indexValue.getDefiningOp<RangeOp>()) {
        std::optional<mlir::AffineExpr> start =
            getAffineIndex(rangeOp.getStart(), iterators, evaluator);

        std::optional<int64_t> step = 1;

        if (rangeOp.getStep()) {
          mlir::Attribute stepValue = evaluator.evaluate(rangeOp.getStep());
          step = stepValue ? getIntegerValue(stepValue) : std::nullopt;
        }

        auto rangeType = mlir::cast<mlir::ShapedType>(rangeOp.getType());
        std::optional<int64_t> size;

        if (rangeType.hasStaticShape()) {
          size = rangeType.getDimSize(0);
        } else if (step && *step != 0) {
          // The size is computed from the bounds, without enumerating the
          // indices.
          Evaluator &sizeEvaluator =
              iteratorsEvaluator ? *iteratorsEvaluator : evaluator;

          mlir::Attribute startValue =
              sizeEvaluator.evaluate(rangeOp.getStart());

          mlir::Attribute stopValue = sizeEvaluator.evaluate(rangeOp.getStop());

          std::optional<int64_t> first =
              startValue ? getIndexValue(startValue) : std::nullopt;

          std::optional<int64_t> last =
              stopValue ? getIndexValue(stopValue) : std::nullopt;

          if (first && last) {
            // The bounds pointing against the step give an empty slice, even
            // when their distance is less than the step.
            if (*step > 0 ? *last < *first : *last > *first) {
              size = 0;
            } else {
              size = (*last - *first) / *step + 1;
            }
          } else if (!iteratorsEvaluator &&
                     (dependsOn(rangeOp.getStart(), iterators) ||
                      dependsOn(rangeOp.getStop(), iterators))) {
            // The size is computed for each value of the iterators.
            access.hasDependentSizes = true;
            size = 0;
          }
        }

        if (!start || !step || !size) {
          return emitNonAffineError();
        }

        selections[dim] = Selection{*start, *step, *size};
        continue;
      }

      // Slices given by constant vectors, which must be made of equally
      // spaced indices. Their elements are written in the code, so they are
      // few.
      mlir::Attribute vector = evaluator.evaluate(indexValue);

      if (!vector) {
        return emitNonAffineError();
      }

      llvm::SmallVector<int64_t> positions;

      for (mlir::Attribute element : getConstantElements(vector)) {
        std::optional<int64_t> position = getIndexValue(element);

        if (!position) {
          return subscriptOp.emitError("invalid index of the connector");
        }

        positions.push_back(*position);
      }

      int64_t step = positions.size() > 1 ? positions[1] - positions[0] : 1;

      for (size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] - positions[i - 1] != step) {
          return emitNonAffineError();
        }
      }

      selections[dim] =
          Selection{mlir::getAffineConstantExpr(
                        positions.empty() ? 0 : positions[0], context),
                    step, static_cast<int64_t>(positions.size())};
    }
  }

  flushLevel();
  access.op = lastOp;
  return mlir::success();
}

mlir::LogicalResult ConnectionExpander::addEdge(
    ConnectOp connectOp, llvm::ArrayRef<LoopBox> connectBoxes,
    llvm::ArrayRef<DimensionRanges> dimensions, unsigned lhsVariable,
    bool lhsOutside, llvm::ArrayRef<mlir::AffineExpr> lhsIndices,
    unsigned rhsVariable, bool rhsOutside,
    llvm::ArrayRef<mlir::AffineExpr> rhsIndices) {
  // The equation indices are the combinations of the ranges of the
  // dimensions, the ones of the iterators and of the sliced dimensions of
  // the connectors being given by the boxes. Equations without dimensions
  // have the single index 0.
  marco::modeling::IndexSet equationIndices;
  size_t numOfBoxDims = connectBoxes.empty() ? 0 : connectBoxes.front().size();

  for (const LoopBox &connectBox : connectBoxes) {
    llvm::SmallVector<DimensionRanges> boxDimensions(connectBox.begin(),
                                                     connectBox.end());

    boxDimensions.append(dimensions.begin(), dimensions.end());

    // Nothing is connected if any dimension is empty.
    if (llvm::any_of(boxDimensions, [](const DimensionRanges &ranges) {
          return ranges.empty();
        })) {
      continue;
    }

    llvm::SmallVector<size_t> counters(boxDimensions.size(), 0);

    while (true) {
      llvm::SmallVector<marco::modeling::Range, 3> ranges;

      for (auto [dimension, counter] :
           llvm::zip_equal(boxDimensions, counters)) {
        ranges.push_back(dimension[counter]);
      }

      if (ranges.empty()) {
        ranges.emplace_back(0, 1);
      }

      equationIndices += marco::modeling::MultidimensionalRange(ranges);

      size_t pos = boxDimensions.size();

      while (pos > 0 && ++counters[pos - 1] == boxDimensions[pos - 1].size()) {
        counters[--pos] = 0;
      }

      if (pos == 0) {
        break;
      }
    }
  }

  if (equationIndices.empty()) {
    return mlir::success();
  }

  mlir::MLIRContext *context = connectOp.getContext();
  unsigned numOfDims = std::max<size_t>(numOfBoxDims + dimensions.size(), 1);

  auto buildAccess = [&](llvm::ArrayRef<mlir::AffineExpr> indices) {
    llvm::SmallVector<mlir::AffineExpr> results(indices);

    if (results.empty()) {
      results.push_back(mlir::getAffineConstantExpr(0, context));
    }

    return marco::modeling::AccessFunction::build(
        mlir::AffineMap::get(numOfDims, 0, results, context));
  };

  std::optional<VertexId> lhs = getVertex(lhsVariable, lhsOutside);
  std::optional<VertexId> rhs = getVertex(rhsVariable, rhsOutside);

  Edge edge;
  edge.equationIndices = std::move(equationIndices);
  edge.lhsAccess = buildAccess(lhsIndices);
  edge.rhsAccess = buildAccess(rhsIndices);

  // Check that the connected elements exist.
  for (auto [vertex, accessFunction] :
       {std::make_pair(lhs, edge.lhsAccess.get()),
        std::make_pair(rhs, edge.rhsAccess.get())}) {
    if (!vertex || !isAccessWithin(*accessFunction, edge.equationIndices,
                                   graph.getVertex(*vertex).indices)) {
      return connectOp.emitError("the indices of the connectors are out of "
                                 "bounds");
    }
  }

  for (auto [vertex, accessFunction] :
       {std::make_pair(*lhs, edge.lhsAccess.get()),
        std::make_pair(*rhs, edge.rhsAccess.get())}) {
    if (graph.getVertex(vertex).outside) {
      reachedIndices[vertex] += getImage(*accessFunction, edge.equationIndices);
    }
  }

  edge.lhs = *lhs;
  edge.rhs = *rhs;
  graph.addEdge(std::move(edge));
  return mlir::success();
}

mlir::LogicalResult ConnectionExpander::processConnect(ConnectOp connectOp) {
  // The enclosing for-equations, outermost first.
  llvm::SmallVector<ForEquationOp> loops;

  for (mlir::Operation *parentOp = connectOp->getParentOp();
       !mlir::isa<EquationSectionOp>(parentOp);
       parentOp = parentOp->getParentOp()) {
    auto forEquationOp = mlir::dyn_cast<ForEquationOp>(parentOp);

    if (!forEquationOp) {
      return connectOp.emitError("connect-equations can only be nested in "
                                 "for-equations after the instantiation");
    }

    loops.push_back(forEquationOp);
  }

  std::reverse(loops.begin(), loops.end());

  // The iterators of the for-equations are the first dimensions of the
  // equation indices.
  Evaluator evaluator;
  llvm::SmallVector<mlir::Value> iterators;
  llvm::SmallVector<LoopBox> loopBoxes;

  if (mlir::failed(getLoopIndices(loops, loopBoxes))) {
    return mlir::failure();
  }

  for (ForEquationOp loop : loops) {
    iterators.push_back(loop.getInductionVariable());
  }

  auto yieldOp =
      mlir::cast<YieldOp>(connectOp.getBodyRegion().front().getTerminator());

  bool outside[2] = {connectOp.getLhsOutside(), connectOp.getRhsOutside()};
  ConnectorAccess sides[2];

  for (size_t side = 0; side < 2; ++side) {
    if (mlir::failed(resolveConnector(connectOp, yieldOp.getValues()[side],
                                      iterators, evaluator, nullptr,
                                      sides[side]))) {
      return mlir::failure();
    }
  }

  // The boxes of the iterators, each followed by the sliced dimensions of
  // the connectors.
  llvm::SmallVector<LoopBox> connectBoxes;

  auto addConnectBoxes =
      [&](llvm::ArrayRef<LoopBox> boxes, const ConnectorAccess &lhsAccess,
          const ConnectorAccess &rhsAccess) -> mlir::LogicalResult {
    // The dimensions not selected by scalar indices are matched in order.
    if (lhsAccess.sliceSizes != rhsAccess.sliceSizes) {
      return connectOp.emitError("the connectors have different sizes");
    }

    for (const LoopBox &box : boxes) {
      LoopBox &connectBox = connectBoxes.emplace_back(box);

      for (int64_t size : lhsAccess.sliceSizes) {
        DimensionRanges &ranges = connectBox.emplace_back();

        if (size > 0) {
          ranges.emplace_back(0, size);
        }
      }
    }

    return mlir::success();
  };

  if (!sides[0].hasDependentSizes && !sides[1].hasDependentSizes) {
    if (mlir::failed(addConnectBoxes(loopBoxes, sides[0], sides[1]))) {
      return mlir::failure();
    }
  } else {
    // The sizes of the slices are computed for each combination of values
    // of the iterators, which are enumerated.
    // TODO: keep compact, e.g. by representing the sizes symbolically in
    // terms of the iterators.
    auto result = enumerateIterators(
        loops, loops.size(),
        [&](Evaluator &iteratorsEvaluator,
            const LoopBox &box) -> mlir::LogicalResult {
          ConnectorAccess pointSides[2];

          for (size_t side = 0; side < 2; ++side) {
            if (mlir::failed(resolveConnector(
                    connectOp, yieldOp.getValues()[side], iterators, evaluator,
                    &iteratorsEvaluator, pointSides[side]))) {
              return mlir::failure();
            }
          }

          return addConnectBoxes(box, pointSides[0], pointSides[1]);
        });

    if (mlir::failed(result)) {
      return mlir::failure();
    }
  }

  size_t numOfSlicedDims = sides[0].sliceSizes.size();
  const ConnectorAccess &lhs = sides[0];
  const ConnectorAccess &rhs = sides[1];

  llvm::ArrayRef<Leaf> lhsLeaves = getLeaves(lhs.op);
  llvm::ArrayRef<Leaf> rhsLeaves = getLeaves(rhs.op);

  if (lhsLeaves.size() != rhsLeaves.size()) {
    return connectOp.emitError("the connectors have different "
                               "structures");
  }

  for (const Leaf &lhsLeaf : lhsLeaves) {
    const Leaf *rhsLeaf = llvm::find_if(rhsLeaves, [&](const Leaf &leaf) {
      return leaf.relativePath == lhsLeaf.relativePath;
    });

    if (rhsLeaf == rhsLeaves.end()) {
      return connectOp.emitError("the connector element '")
             << lhsLeaf.relativePath << "' is missing in the other connector";
    }

    const Variable &lhsVariable = variables[lhsLeaf.variable];
    const Variable &rhsVariable = variables[rhsLeaf->variable];

    auto lhsKind = lhsVariable.op.getConnectorKind();
    auto rhsKind = rhsVariable.op.getConnectorKind();

    if (lhsKind != rhsKind) {
      return connectOp.emitError("the connector element '")
             << lhsLeaf.relativePath
             << "' has different prefixes in the two connectors";
    }

    llvm::ArrayRef<int64_t> lhsRelativeShape =
        llvm::ArrayRef(lhsVariable.shape).drop_front(lhsLeaf.numPrefixDims);

    llvm::ArrayRef<int64_t> rhsRelativeShape =
        llvm::ArrayRef(rhsVariable.shape).drop_front(rhsLeaf->numPrefixDims);

    if (lhsRelativeShape != rhsRelativeShape) {
      return connectOp.emitError("the connector element '")
             << lhsLeaf.relativePath
             << "' has different dimensions in the two connectors";
    }

    // Constants and parameters are not part of the connection sets.
    if (auto variability = lhsVariable.op.getVariability();
        variability && *variability != Variability::discrete) {
      continue;
    }

    // Stream variables are handled by the stream operators.
    if (lhsKind == ConnectorKind::stream) {
      continue;
    }

    // Each dimension of the primitive variables within the connectors is a
    // further dimension of the equations.
    llvm::SmallVector<DimensionRanges> dimensions;
    llvm::SmallVector<mlir::AffineExpr> lhsIndices(lhs.indices);
    llvm::SmallVector<mlir::AffineExpr> rhsIndices(rhs.indices);

    for (int64_t size : lhsRelativeShape) {
      mlir::AffineExpr dimension = mlir::getAffineDimExpr(
          loops.size() + numOfSlicedDims + dimensions.size(),
          connectOp.getContext());

      lhsIndices.push_back(dimension);
      rhsIndices.push_back(dimension);

      DimensionRanges &ranges = dimensions.emplace_back();

      if (size > 0) {
        ranges.emplace_back(0, size);
      }
    }

    if (mlir::failed(addEdge(connectOp, connectBoxes, dimensions,
                             lhsLeaf.variable, outside[0], lhsIndices,
                             rhsLeaf->variable, outside[1], rhsIndices))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::Value
ConnectionExpander::emitReference(mlir::OpBuilder &builder, mlir::Location loc,
                                  unsigned variableId,
                                  llvm::ArrayRef<mlir::Value> indices) {
  const Variable &variable = variables[variableId];
  mlir::Value value;

  for (auto [op, rank] : llvm::zip_equal(variable.path, variable.ranks)) {
    mlir::Type type = getOpType(op);

    if (!value) {
      value = builder.create<RefOp>(loc, type, getOpName(op));
    } else {
      value = builder.create<MemberOp>(loc, type, value, getOpName(op));
    }

    if (rank != 0) {
      value = builder.create<SubscriptOp>(loc, mlir::getElementTypeOrSelf(type),
                                          value, indices.take_front(rank));
      indices = indices.drop_front(rank);
    }
  }

  return value;
}

/// Emit an affine expression of the iterators.
static mlir::Value emitAffineExpr(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::AffineExpr expr,
                                  llvm::ArrayRef<mlir::Value> iterators) {
  mlir::MLIRContext *context = builder.getContext();

  if (auto constantExpr = mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    return builder.create<ConstantOp>(
        loc, IntegerAttr::get(context, constantExpr.getValue()));
  }

  if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
    return iterators[dimExpr.getPosition()];
  }

  auto binaryExpr = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
  mlir::Value lhs =
      emitAffineExpr(builder, loc, binaryExpr.getLHS(), iterators);
  mlir::Value rhs =
      emitAffineExpr(builder, loc, binaryExpr.getRHS(), iterators);

  if (expr.getKind() == mlir::AffineExprKind::Add) {
    return builder.create<AddOp>(loc, lhs, rhs);
  }

  if (expr.getKind() == mlir::AffineExprKind::Mul) {
    return builder.create<MulOp>(loc, lhs, rhs);
  }

  llvm_unreachable("unexpected affine expression");
}

llvm::SmallVector<mlir::Value> ConnectionExpander::emitIndices(
    mlir::OpBuilder &builder, mlir::Location loc, VertexId vertex,
    const marco::modeling::AccessFunction &accessFunction,
    llvm::ArrayRef<mlir::AffineExpr> replacements,
    llvm::ArrayRef<mlir::Value> iterators) {
  llvm::SmallVector<mlir::Value> indices;

  // Scalar variables have the single index 0, which is not emitted.
  if (variables[graph.getVertex(vertex).variable].shape.empty()) {
    return indices;
  }

  mlir::AffineMap map = mlir::simplifyAffineMap(
      accessFunction.getAffineMap().replaceDimsAndSymbols(replacements, {},
                                                          iterators.size(), 0));

  for (mlir::AffineExpr result : map.getResults()) {
    indices.push_back(emitAffineExpr(builder, loc, result, iterators));
  }

  return indices;
}

/// Emit the equations of a family for each range of its indices, within
/// for-equations iterating on the dimensions spanning multiple indices. The
/// callback emits the sides of an equation, given the replacements of the
/// dimensions of the indices (constants for the dimensions of a single
/// index, and the iterators for the other ones) and the iterators.
static void emitFamily(mlir::OpBuilder &builder, mlir::Location loc,
                       const marco::modeling::IndexSet &indices,
                       llvm::function_ref<void(llvm::ArrayRef<mlir::AffineExpr>,
                                               llvm::ArrayRef<mlir::Value>)>
                           emitSides) {
  mlir::MLIRContext *context = builder.getContext();
  llvm::SmallVector<marco::modeling::MultidimensionalRange> ranges;

  // Compacting the ranges takes a time quadratic in their number. The sets
  // with many ranges come from the elements processed point by point, and
  // are emitted as they are.
  // TODO: compact them once ranges with a step are supported.
  if (std::distance(indices.rangesBegin(), indices.rangesEnd()) <= 64) {
    indices.getCompactRanges(ranges);
  } else {
    ranges.append(indices.rangesBegin(), indices.rangesEnd());
  }

  for (const marco::modeling::MultidimensionalRange &range : ranges) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    llvm::SmallVector<mlir::AffineExpr> replacements;
    llvm::SmallVector<mlir::Value> iterators;

    for (unsigned i = 0, e = range.rank(); i < e; ++i) {
      const marco::modeling::Range &dimension = range[i];

      if (dimension.size() == 1) {
        replacements.push_back(
            mlir::getAffineConstantExpr(dimension.getBegin(), context));

        continue;
      }

      auto forEquationOp = builder.create<ForEquationOp>(loc);
      builder.createBlock(&forEquationOp.getRangeRegion());

      auto from = builder.create<ConstantOp>(
          loc, IntegerAttr::get(context, dimension.getBegin()));

      auto to = builder.create<ConstantOp>(
          loc, IntegerAttr::get(context, dimension.getEnd() - 1));

      auto rangeOp = builder.create<RangeOp>(
          loc,
          mlir::RankedTensorType::get({static_cast<int64_t>(dimension.size())},
                                      IntegerType::get(context)),
          from, to, mlir::Value());

      builder.create<YieldOp>(loc, rangeOp.getResult());

      mlir::Block *body = builder.createBlock(
          &forEquationOp.getBodyRegion(), {}, IntegerType::get(context), loc);

      replacements.push_back(mlir::getAffineDimExpr(iterators.size(), context));

      iterators.push_back(body->getArgument(0));
    }

    auto equationOp = builder.create<EquationOp>(loc);
    builder.createBlock(&equationOp.getBodyRegion());
    emitSides(replacements, iterators);
  }
}

void ConnectionExpander::emitEquations(const ConnectionEquations &equations) {
  mlir::MLIRContext *context = modelOp.getContext();
  mlir::Location loc = modelOp.getLoc();

  // Flow families whose roots are outside elements not reached by any
  // connect-equation are not emitted: such elements don't belong to any
  // connection set.
  llvm::SmallVector<marco::modeling::IndexSet> flowIndices;
  bool hasEquations = !equations.potentials.empty();

  for (const FlowFamily &family : equations.flows) {
    marco::modeling::IndexSet &indices =
        flowIndices.emplace_back(family.rootIndices);

    if (graph.getVertex(family.root).outside) {
      indices = indices.intersect(reachedIndices.lookup(family.root));
    }

    hasEquations |= !indices.empty();
  }

  if (!hasEquations) {
    return;
  }

  mlir::OpBuilder builder(context);
  builder.setInsertionPointToEnd(modelOp.getBody());
  auto sectionOp = builder.create<EquationSectionOp>(loc, false);
  builder.createBlock(&sectionOp.getBodyRegion());

  // Equalities of the potential variables.
  for (const PotentialFamily &family : equations.potentials) {
    auto identity = getIdentity(context, family.indices.rank());

    emitFamily(builder, loc, family.indices,
               [&](llvm::ArrayRef<mlir::AffineExpr> replacements,
                   llvm::ArrayRef<mlir::Value> iterators) {
                 mlir::Value lhs = emitReference(
                     builder, loc, graph.getVertex(family.vertex).variable,
                     emitIndices(builder, loc, family.vertex, *identity,
                                 replacements, iterators));

                 mlir::Value rhs = emitReference(
                     builder, loc, graph.getVertex(family.root).variable,
                     emitIndices(builder, loc, family.root, *family.toRoot,
                                 replacements, iterators));

                 builder.create<EquationSidesOp>(loc, mlir::ValueRange(lhs),
                                                 mlir::ValueRange(rhs));
               });
  }

  // Zero-sum equations of the flow variables, with inside connectors taken
  // positively and outside connectors negatively.
  for (size_t i = 0, e = equations.flows.size(); i < e; ++i) {
    const FlowFamily &family = equations.flows[i];
    const marco::modeling::IndexSet &indices = flowIndices[i];

    if (indices.empty()) {
      continue;
    }

    emitFamily(
        builder, loc, indices,
        [&](llvm::ArrayRef<mlir::AffineExpr> replacements,
            llvm::ArrayRef<mlir::Value> iterators) {
          mlir::Value sum;

          auto addTerm = [&](mlir::Value term, VertexId vertex) {
            bool isOutside = graph.getVertex(vertex).outside;

            if (!sum) {
              sum = isOutside ? builder.create<NegateOp>(loc, term).getResult()
                              : term;
            } else if (isOutside) {
              sum = builder.create<SubOp>(loc, sum, term);
            } else {
              sum = builder.create<AddOp>(loc, sum, term);
            }
          };

          for (const FlowTerm &term : family.terms) {
            addTerm(emitReference(
                        builder, loc, graph.getVertex(term.vertex).variable,
                        emitIndices(builder, loc, term.vertex, *term.fromRoot,
                                    replacements, iterators)),
                    term.vertex);
          }

          for (const FlowReduction &reduction : family.reductions) {
            // The iterators of the reduction follow the root indices.
            llvm::SmallVector<mlir::AffineExpr> reductionReplacements(
                replacements);

            llvm::SmallVector<mlir::Value> reductionIterators(iterators);
            llvm::SmallVector<mlir::Value> rangeValues;
            llvm::SmallVector<mlir::Type> argTypes;
            llvm::SmallVector<mlir::Location> argLocs;

            for (unsigned i = 0, e = reduction.iterators.rank(); i < e; ++i) {
              const marco::modeling::Range &dimension = reduction.iterators[i];

              auto from = builder.create<ConstantOp>(
                  loc, IntegerAttr::get(context, dimension.getBegin()));

              auto to = builder.create<ConstantOp>(
                  loc, IntegerAttr::get(context, dimension.getEnd() - 1));

              rangeValues.push_back(builder.create<RangeOp>(
                  loc,
                  mlir::RankedTensorType::get(
                      {static_cast<int64_t>(dimension.size())},
                      IntegerType::get(context)),
                  from, to, mlir::Value()));

              argTypes.push_back(IntegerType::get(context));
              argLocs.push_back(loc);
            }

            const Variable &variable =
                variables[graph.getVertex(reduction.vertex).variable];

            auto reductionOp = builder.create<ReductionOp>(
                loc, mlir::getElementTypeOrSelf(variable.op.getType()),
                ReductionKind::sum, rangeValues);

            mlir::OpBuilder::InsertionGuard guard(builder);

            mlir::Block *body = builder.createBlock(&reductionOp.getBody(), {},
                                                    argTypes, argLocs);

            for (mlir::BlockArgument arg : body->getArguments()) {
              reductionReplacements.push_back(
                  mlir::getAffineDimExpr(reductionIterators.size(), context));

              reductionIterators.push_back(arg);
            }

            mlir::Value element = emitReference(
                builder, loc, graph.getVertex(reduction.vertex).variable,
                emitIndices(builder, loc, reduction.vertex, *reduction.access,
                            reductionReplacements, reductionIterators));

            builder.create<YieldOp>(loc, element);
            builder.setInsertionPointAfter(reductionOp);
            addTerm(reductionOp.getResult(), reduction.vertex);
          }

          mlir::Type type = sum.getType();
          mlir::Value zero = builder.create<ConstantOp>(
              loc, mlir::cast<mlir::TypedAttr>(getNumericAttr(type, 0)));

          builder.create<EquationSidesOp>(loc, mlir::ValueRange(sum),
                                          mlir::ValueRange(zero));
        });
  }
}

mlir::LogicalResult ConnectionExpander::run() {
  mlir::IRRewriter rewriter(modelOp.getContext());
  llvm::SmallVector<mlir::Operation *> path;
  collectVariables(modelOp, path);

  // Each primitive flow variable is part of a connection set as inside
  // element, so that it is set to zero if not connected.
  for (auto [id, variable] : llvm::enumerate(variables)) {
    if (variable.op.getConnectorKind() == ConnectorKind::flow) {
      getVertex(id, false);
    }
  }

  llvm::SmallVector<ConnectOp> connectOps;
  modelOp.walk([&](ConnectOp connectOp) { connectOps.push_back(connectOp); });

  for (ConnectOp connectOp : connectOps) {
    if (mlir::failed(processConnect(connectOp))) {
      return mlir::failure();
    }
  }

  graph.computeRepresentatives();
  emitEquations(computeEquations(graph));

  for (ConnectOp connectOp : connectOps) {
    rewriter.eraseOp(connectOp);
  }

  // Remove the for-equations left empty.
  bool changed = true;

  while (changed) {
    changed = false;

    modelOp.walk([&](ForEquationOp forEquationOp) {
      if (forEquationOp.getBodyRegion().front().empty()) {
        rewriter.eraseOp(forEquationOp);
        changed = true;
      }
    });
  }

  return mlir::success();
}

class ConnectionExpansionPass final
    : public mlir::modelica::impl::ConnectionExpansionPassBase<
          ConnectionExpansionPass> {
public:
  using ConnectionExpansionPassBase<
      ConnectionExpansionPass>::ConnectionExpansionPassBase;

  void runOnOperation() override {
    for (ModelOp modelOp : getOperation().getOps<ModelOp>()) {
      ConnectionExpander expander(modelOp);

      if (mlir::failed(expander.run())) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace
