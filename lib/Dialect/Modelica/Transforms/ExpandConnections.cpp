#include "marco/Dialect/Modelica/Transforms/ExpandConnections.h"
#include "marco/Dialect/Modelica/IR/ConstantFolding.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "marco/Dialect/Modelica/Transforms/Evaluation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include <map>

namespace mlir::modelica {
#define GEN_PASS_DEF_EXPANDCONNECTIONSPASS
#include "marco/Dialect/Modelica/Transforms/Passes.h.inc"
} // namespace mlir::modelica

using namespace ::mlir::modelica;

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

/// Scalar element of a primitive variable, as part of a connection set.
struct SetElement {
  unsigned variable;
  int64_t offset;
  bool outside;

  bool operator<(const SetElement &other) const {
    return std::tie(variable, offset, outside) <
           std::tie(other.variable, other.offset, other.outside);
  }

  bool operator==(const SetElement &other) const {
    return variable == other.variable && offset == other.offset &&
           outside == other.outside;
  }
};

/// Element of a connector referenced by a connect-equation: the operations
/// from the model to the referenced instance or variable, with the indices
/// of all their dimensions.
struct ConnectorElement {
  mlir::Operation *op;
  llvm::SmallVector<mlir::Operation *> path;
  llvm::SmallVector<int64_t> indices;
};

/// Primitive variable within a connector, identified by its path relative
/// to the connector.
struct Leaf {
  std::string relativePath;
  unsigned variable;

  /// Number of the dimensions belonging to the connector.
  unsigned numPrefixDims;
};

/// Family of equations described by an affine function of the iteration
/// indices: the indices of the i-th equation are
/// start + sum_d (stride_d * j_d), with 0 <= j_d < count_d.
struct EquationRange {
  llvm::SmallVector<int64_t> start;
  llvm::SmallVector<std::pair<llvm::SmallVector<int64_t>, int64_t>> dims;
};

/// Signature of the equations generated for connection sets: the kind of
/// the equation, and the variables involved with their signs.
struct EquationSignature {
  bool isFlow;
  llvm::SmallVector<std::pair<unsigned, bool>> terms;

  bool operator<(const EquationSignature &other) const {
    return std::tie(isFlow, terms) < std::tie(other.isFlow, other.terms);
  }
};

class ConnectionExpander {
public:
  explicit ConnectionExpander(ModelOp modelOp) : modelOp(modelOp) {}

  mlir::LogicalResult run();

private:
  void collectVariables(mlir::Operation *op,
                        llvm::SmallVectorImpl<mlir::Operation *> &path);

  mlir::LogicalResult processConnect(ConnectOp connectOp);

  mlir::LogicalResult
  resolveConnector(mlir::Value value, Evaluator &evaluator,
                   llvm::SmallVectorImpl<ConnectorElement> &elements);

  llvm::ArrayRef<Leaf> getLeaves(mlir::Operation *op);

  int64_t getOffset(const Variable &variable,
                    llvm::ArrayRef<int64_t> indices) const;

  llvm::SmallVector<int64_t> getIndices(const Variable &variable,
                                        int64_t offset) const;

  unsigned getNode(const SetElement &element);

  mlir::LogicalResult emitEquations();

  mlir::Value emitReference(mlir::OpBuilder &builder, mlir::Location loc,
                            unsigned variable,
                            llvm::ArrayRef<mlir::Value> indices);

private:
  ModelOp modelOp;
  llvm::SmallVector<Variable> variables;
  llvm::DenseMap<mlir::Operation *, unsigned> variableIds;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Leaf>> leaves;

  std::map<SetElement, unsigned> nodeIds;
  llvm::SmallVector<SetElement> nodes;
  llvm::EquivalenceClasses<unsigned> sets;
};

/// Get the dimensions of an instance or variable.
llvm::ArrayRef<int64_t> getOpShape(mlir::Operation *op) {
  if (auto instanceOp = mlir::dyn_cast<InstanceOp>(op)) {
    return instanceOp.getShape();
  }

  return mlir::modelica::getShape(mlir::cast<VariableOp>(op).getType());
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

void ConnectionExpander::collectVariables(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::Operation *> &path) {
  for (mlir::Operation &nested : op->getRegion(0).front()) {
    if (auto instanceOp = mlir::dyn_cast<InstanceOp>(nested)) {
      path.push_back(instanceOp);
      collectVariables(instanceOp, path);
      path.pop_back();
      continue;
    }

    if (auto variableOp = mlir::dyn_cast<VariableOp>(nested)) {
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

int64_t ConnectionExpander::getOffset(const Variable &variable,
                                      llvm::ArrayRef<int64_t> indices) const {
  int64_t offset = 0;

  for (auto [size, index] : llvm::zip_equal(variable.shape, indices)) {
    offset = offset * size + index;
  }

  return offset;
}

llvm::SmallVector<int64_t>
ConnectionExpander::getIndices(const Variable &variable, int64_t offset) const {
  llvm::SmallVector<int64_t> indices(variable.shape.size());

  for (size_t i = variable.shape.size(); i > 0; --i) {
    indices[i - 1] = offset % variable.shape[i - 1];
    offset /= variable.shape[i - 1];
  }

  return indices;
}

unsigned ConnectionExpander::getNode(const SetElement &element) {
  auto [it, inserted] = nodeIds.try_emplace(element, nodes.size());

  if (inserted) {
    nodes.push_back(element);
    sets.insert(it->second);
  }

  return it->second;
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

      for (auto [pathOp, rank] : llvm::zip_equal(variable.path, variable.ranks)) {
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

mlir::LogicalResult ConnectionExpander::resolveConnector(
    mlir::Value value, Evaluator &evaluator,
    llvm::SmallVectorImpl<ConnectorElement> &elements) {
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

  // Positions selected along each dimension of each operation of the path.
  llvm::SmallVector<mlir::Operation *> path;
  llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<int64_t>>> selections;

  auto addLevel = [&](mlir::Operation *op) {
    path.push_back(op);
    auto &selection = selections.emplace_back();

    for (int64_t size : getOpShape(op)) {
      auto &positions = selection.emplace_back();

      for (int64_t i = 0; i < size; ++i) {
        positions.push_back(i);
      }
    }
  };

  for (mlir::Operation *op : chain) {
    if (auto refOp = mlir::dyn_cast<RefOp>(op)) {
      mlir::Operation *child = lookupChild(modelOp, refOp.getSimpleName());

      if (!child) {
        return refOp.emitError("unknown component '")
               << refOp.getName() << "'";
      }

      addLevel(child);
      continue;
    }

    if (auto memberOp = mlir::dyn_cast<MemberOp>(op)) {
      mlir::Operation *child =
          mlir::isa<InstanceOp>(path.back())
              ? lookupChild(path.back(), memberOp.getMember())
              : nullptr;

      if (!child) {
        return memberOp.emitError("unknown component '")
               << memberOp.getMember() << "'";
      }

      addLevel(child);
      continue;
    }

    auto subscriptOp = mlir::cast<SubscriptOp>(op);
    auto &selection = selections.back();
    auto index = subscriptOp.getIndices().begin();

    for (auto [dim, isColon] : llvm::enumerate(subscriptOp.getColons())) {
      if (isColon) {
        continue;
      }

      mlir::Attribute indexValue = evaluator.evaluate(*index++);

      if (!indexValue) {
        return subscriptOp.emitError("can't evaluate the index of the "
                                     "connector");
      }

      llvm::SmallVector<int64_t> positions;

      for (mlir::Attribute element : getConstantElements(indexValue)) {
        std::optional<int64_t> position = getIndexValue(element);

        if (!position) {
          return subscriptOp.emitError("invalid index of the connector");
        }

        positions.push_back(*position);
      }

      selection[dim] = std::move(positions);
    }
  }

  // Enumerate the selected elements, in row-major order.
  llvm::SmallVector<llvm::ArrayRef<int64_t>> dims;

  for (auto &selection : selections) {
    for (auto &positions : selection) {
      dims.push_back(positions);
    }
  }

  llvm::SmallVector<size_t> counters(dims.size(), 0);

  if (llvm::any_of(dims, [](llvm::ArrayRef<int64_t> d) { return d.empty(); })) {
    return mlir::success();
  }

  while (true) {
    ConnectorElement &element = elements.emplace_back();
    element.op = path.back();
    element.path = path;

    for (size_t i = 0; i < dims.size(); ++i) {
      element.indices.push_back(dims[i][counters[i]]);
    }

    size_t pos = dims.size();

    while (pos > 0 && ++counters[pos - 1] == dims[pos - 1].size()) {
      counters[--pos] = 0;
    }

    if (pos == 0) {
      break;
    }
  }

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
      return connectOp.emitError("connect-equations can only be nested "
                                 "within for-equations");
    }

    loops.push_back(forEquationOp);
  }

  std::reverse(loops.begin(), loops.end());

  auto yieldOp =
      mlir::cast<YieldOp>(connectOp.getBodyRegion().front().getTerminator());

  bool outside[2] = {connectOp.getLhsOutside(), connectOp.getRhsOutside()};
  llvm::SmallVector<mlir::Attribute> iterationValues;

  std::function<mlir::LogicalResult(size_t)> iterate =
      [&](size_t depth) -> mlir::LogicalResult {
    Evaluator evaluator;

    for (auto [loop, value] : llvm::zip(loops, iterationValues)) {
      evaluator.setValue(loop.getInductionVariable(), value);
    }

    if (depth < loops.size()) {
      ForEquationOp loop = loops[depth];

      mlir::Attribute range = evaluator.evaluate(
          mlir::cast<YieldOp>(loop.getRangeRegion().front().getTerminator())
              .getValues()[0]);

      if (!range) {
        return loop.emitError("can't evaluate the range of the for-equation "
                              "containing connect-equations");
      }

      for (mlir::Attribute value : getConstantElements(range)) {
        iterationValues.push_back(value);

        if (mlir::failed(iterate(depth + 1))) {
          return mlir::failure();
        }

        iterationValues.pop_back();
      }

      return mlir::success();
    }

    llvm::SmallVector<ConnectorElement> sides[2];

    for (size_t side = 0; side < 2; ++side) {
      if (mlir::failed(resolveConnector(yieldOp.getValues()[side], evaluator,
                                        sides[side]))) {
        return mlir::failure();
      }
    }

    if (sides[0].size() != sides[1].size()) {
      return connectOp.emitError("the connectors have different sizes");
    }

    for (auto [lhs, rhs] : llvm::zip_equal(sides[0], sides[1])) {
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
                 << lhsLeaf.relativePath
                 << "' is missing in the other connector";
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

        // Each scalar element of the primitive variables is connected.
        int64_t numElements = 1;

        for (int64_t size : lhsRelativeShape) {
          numElements *= size;
        }

        for (int64_t i = 0; i < numElements; ++i) {
          llvm::SmallVector<int64_t> relative;
          int64_t remaining = i;

          for (int64_t size : llvm::reverse(lhsRelativeShape)) {
            relative.push_back(remaining % size + 1);
            remaining /= size;
          }

          std::reverse(relative.begin(), relative.end());

          llvm::SmallVector<int64_t> lhsIndices(lhs.indices);
          lhsIndices.append(relative.begin(), relative.end());

          llvm::SmallVector<int64_t> rhsIndices(rhs.indices);
          rhsIndices.append(relative.begin(), relative.end());

          unsigned lhsNode =
              getNode({lhsLeaf.variable, getOffset(lhsVariable, lhsIndices),
                       outside[0]});

          unsigned rhsNode =
              getNode({rhsLeaf->variable, getOffset(rhsVariable, rhsIndices),
                       outside[1]});

          sets.unionSets(lhsNode, rhsNode);
        }
      }
    }

    return mlir::success();
  };

  return iterate(0);
}

mlir::Value ConnectionExpander::emitReference(
    mlir::OpBuilder &builder, mlir::Location loc, unsigned variableId,
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

mlir::LogicalResult ConnectionExpander::emitEquations() {
  mlir::MLIRContext *context = modelOp.getContext();
  mlir::Location loc = modelOp.getLoc();

  // Generate the equations of each set, grouping the ones having the same
  // structure.
  std::map<EquationSignature, std::vector<llvm::SmallVector<int64_t>>> groups;

  for (const auto *set : sets) {
    if (!set->isLeader()) {
      continue;
    }

    llvm::SmallVector<SetElement> members;

    for (unsigned member : sets.members(*set)) {
      members.push_back(nodes[member]);
    }

    llvm::sort(members);

    const Variable &first = variables[members.front().variable];
    auto connectorKind = first.op.getConnectorKind();

    if (connectorKind == ConnectorKind::stream) {
      // Stream variables are handled by the stream operators.
      continue;
    }

    if (connectorKind == ConnectorKind::flow) {
      // Zero-sum equation, with inside connectors taken positively and
      // outside connectors negatively.
      EquationSignature signature{true, {}};
      llvm::SmallVector<int64_t> point;

      for (const SetElement &member : members) {
        signature.terms.emplace_back(member.variable, member.outside);
        auto indices = getIndices(variables[member.variable], member.offset);
        point.append(indices.begin(), indices.end());
      }

      groups[signature].push_back(std::move(point));
      continue;
    }

    // Equalities of the potential variables. The same variable can belong
    // to the set both as inside and as outside element.
    llvm::SmallVector<std::pair<unsigned, int64_t>> unique;

    for (const SetElement &member : members) {
      std::pair<unsigned, int64_t> key(member.variable, member.offset);

      if (unique.empty() || unique.back() != key) {
        unique.push_back(key);
      }
    }

    for (size_t i = 1; i < unique.size(); ++i) {
      EquationSignature signature{false,
                                  {{unique[0].first, false},
                                   {unique[i].first, false}}};

      auto lhsIndices = getIndices(variables[unique[0].first], unique[0].second);
      auto rhsIndices = getIndices(variables[unique[i].first], unique[i].second);

      llvm::SmallVector<int64_t> point(lhsIndices);
      point.append(rhsIndices.begin(), rhsIndices.end());
      groups[signature].push_back(std::move(point));
    }
  }

  if (groups.empty()) {
    return mlir::success();
  }

  mlir::OpBuilder builder(context);
  builder.setInsertionPointToEnd(modelOp.getBody());
  auto sectionOp = builder.create<EquationSectionOp>(loc, false);
  builder.createBlock(&sectionOp.getBodyRegion());

  for (auto &[signature, points] : groups) {
    // Group the equations into affine families, by repeatedly merging the
    // families having the same shape and whose starting points are
    // equally spaced.
    std::vector<EquationRange> ranges;

    for (auto &point : points) {
      ranges.push_back(EquationRange{point, {}});
    }

    bool changed = true;

    while (changed && ranges.size() > 1) {
      changed = false;

      llvm::sort(ranges, [](const EquationRange &lhs, const EquationRange &rhs) {
        return std::tie(lhs.dims, lhs.start) < std::tie(rhs.dims, rhs.start);
      });

      std::vector<EquationRange> merged;

      for (size_t i = 0; i < ranges.size();) {
        size_t j = i + 1;
        llvm::SmallVector<int64_t> stride;

        if (j < ranges.size() && ranges[j].dims == ranges[i].dims) {
          for (auto [lhs, rhs] :
               llvm::zip_equal(ranges[i].start, ranges[j].start)) {
            stride.push_back(rhs - lhs);
          }

          while (j < ranges.size() && ranges[j].dims == ranges[i].dims) {
            bool isNext = true;

            for (size_t k = 0; k < stride.size(); ++k) {
              if (ranges[j].start[k] - ranges[j - 1].start[k] != stride[k]) {
                isNext = false;
                break;
              }
            }

            if (!isNext) {
              break;
            }

            ++j;
          }
        }

        // A family of two equations is not formed if the second equation
        // starts a longer family.
        if (j - i == 2) {
          auto getRunLength = [&](size_t first) {
            size_t last = first + 1;

            if (last >= ranges.size() || ranges[last].dims != ranges[first].dims) {
              return static_cast<size_t>(1);
            }

            while (last + 1 < ranges.size() &&
                   ranges[last + 1].dims == ranges[first].dims) {
              bool isNext = true;

              for (size_t k = 0; k < ranges[first].start.size(); ++k) {
                if (ranges[last + 1].start[k] - ranges[last].start[k] !=
                    ranges[first + 1].start[k] - ranges[first].start[k]) {
                  isNext = false;
                  break;
                }
              }

              if (!isNext) {
                break;
              }

              ++last;
            }

            return last - first + 1;
          };

          if (getRunLength(i + 1) > 2) {
            j = i + 1;
          }
        }

        if (j - i == 1) {
          merged.push_back(std::move(ranges[i]));
        } else {
          EquationRange range;
          range.start = ranges[i].start;
          range.dims.emplace_back(stride, static_cast<int64_t>(j - i));
          range.dims.append(ranges[i].dims.begin(), ranges[i].dims.end());
          merged.push_back(std::move(range));
          changed = true;
        }

        i = j;
      }

      ranges = std::move(merged);
    }

    for (const EquationRange &range : ranges) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      llvm::SmallVector<mlir::Value> iterators;

      for (const auto &dim : range.dims) {
        auto forEquationOp = builder.create<ForEquationOp>(loc);
        builder.createBlock(&forEquationOp.getRangeRegion());

        auto from = builder.create<ConstantOp>(loc, IntegerAttr::get(context, 0));
        auto to = builder.create<ConstantOp>(
            loc, IntegerAttr::get(context, dim.second - 1));

        auto rangeOp = builder.create<RangeOp>(
            loc,
            mlir::RankedTensorType::get({dim.second},
                                        IntegerType::get(context)),
            from, to, mlir::Value());

        builder.create<YieldOp>(loc, rangeOp.getResult());

        mlir::Block *body = builder.createBlock(
            &forEquationOp.getBodyRegion(), {}, IntegerType::get(context), loc);

        iterators.push_back(body->getArgument(0));
      }

      auto equationOp = builder.create<EquationOp>(loc);
      builder.createBlock(&equationOp.getBodyRegion());

      // Compute the indices as affine functions of the iterators.
      llvm::SmallVector<mlir::Value> indices;

      for (size_t k = 0; k < range.start.size(); ++k) {
        mlir::Value index = builder.create<ConstantOp>(
            loc, IntegerAttr::get(context, range.start[k]));

        for (auto [dim, iterator] : llvm::zip_equal(range.dims, iterators)) {
          int64_t stride = dim.first[k];

          if (stride == 0) {
            continue;
          }

          mlir::Value term = iterator;

          if (stride != 1) {
            auto strideValue = builder.create<ConstantOp>(
                loc, IntegerAttr::get(context, stride));

            term = builder.create<MulOp>(loc, strideValue, iterator);
          }

          index = builder.create<AddOp>(loc, index, term);
        }

        indices.push_back(index);
      }

      llvm::SmallVector<mlir::Value> terms;
      llvm::ArrayRef<mlir::Value> remainingIndices = indices;

      for (auto [variable, outside] : signature.terms) {
        size_t rank = variables[variable].shape.size();

        terms.push_back(emitReference(builder, loc, variable,
                                      remainingIndices.take_front(rank)));

        remainingIndices = remainingIndices.drop_front(rank);
      }

      if (signature.isFlow) {
        mlir::Value sum;

        for (auto [term, element] : llvm::zip_equal(terms, signature.terms)) {
          bool isOutside = element.second;

          if (!sum) {
            sum = isOutside ? builder.create<NegateOp>(loc, term).getResult()
                            : term;
          } else if (isOutside) {
            sum = builder.create<SubOp>(loc, sum, term);
          } else {
            sum = builder.create<AddOp>(loc, sum, term);
          }
        }

        mlir::Type type = sum.getType();
        mlir::Value zero = builder.create<ConstantOp>(
            loc, mlir::cast<mlir::TypedAttr>(getNumericAttr(type, 0)));

        builder.create<EquationSidesOp>(loc, mlir::ValueRange(sum),
                                        mlir::ValueRange(zero));
      } else {
        builder.create<EquationSidesOp>(loc, mlir::ValueRange(terms[0]),
                                        mlir::ValueRange(terms[1]));
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult ConnectionExpander::run() {
  llvm::SmallVector<mlir::Operation *> path;
  collectVariables(modelOp, path);

  llvm::SmallVector<ConnectOp> connectOps;
  modelOp.walk([&](ConnectOp connectOp) { connectOps.push_back(connectOp); });

  for (ConnectOp connectOp : connectOps) {
    if (mlir::failed(processConnect(connectOp))) {
      return mlir::failure();
    }
  }

  // Each primitive flow variable is part of a connection set as inside
  // element, so that it is set to zero if not connected.
  for (auto [id, variable] : llvm::enumerate(variables)) {
    if (variable.op.getConnectorKind() != ConnectorKind::flow) {
      continue;
    }

    int64_t numElements = 1;

    for (int64_t size : variable.shape) {
      numElements *= size;
    }

    for (int64_t offset = 0; offset < numElements; ++offset) {
      getNode({static_cast<unsigned>(id), offset, false});
    }
  }

  if (mlir::failed(emitEquations())) {
    return mlir::failure();
  }

  for (ConnectOp connectOp : connectOps) {
    connectOp.erase();
  }

  // Remove the for-equations left empty.
  bool changed = true;

  while (changed) {
    changed = false;

    modelOp.walk([&](ForEquationOp forEquationOp) {
      if (forEquationOp.getBodyRegion().front().empty()) {
        forEquationOp.erase();
        changed = true;
      }
    });
  }

  return mlir::success();
}

class ExpandConnectionsPass final
    : public mlir::modelica::impl::ExpandConnectionsPassBase<
          ExpandConnectionsPass> {
public:
  using ExpandConnectionsPassBase<
      ExpandConnectionsPass>::ExpandConnectionsPassBase;

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
