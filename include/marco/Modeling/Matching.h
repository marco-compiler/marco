#ifndef MARCO_MODELING_MATCHING_H
#define MARCO_MODELING_MATCHING_H

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "matching"
#endif

#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/AccessFunctionAffineConstant.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/Graph.h"
#include "marco/Modeling/LocalMatchingSolutions.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <list>
#include <memory>
#include <mutex>
#include <variant>

namespace marco::modeling {
namespace matching {
// This class must be specialized for the variable type that is used during
// the matching process.
template <typename VariableType>
struct VariableTraits {
  // Elements to provide:
  //
  // typedef Id : the ID type of the variable.
  //
  // static Id getId(const VariableType*)
  //    return the ID of the variable.
  //
  // static size_t getRank(const VariableType*)
  //    return the number of dimensions.
  //
  // static IndexSet getIndices(const VariableType*)
  //    return the indices of a variable.
  //
  // static llvm::raw_ostream& dump(const VariableType*, llvm::raw_ostream&)
  //    print debug information.

  using Id = typename VariableType::UnknownVariableTypeError;
};

// This class must be specialized for the equation type that is used during
// the matching process.
template <typename EquationType>
struct EquationTraits {
  // Elements to provide:
  //
  // typedef Id : the ID type of the equation.
  //
  // static Id getId(const EquationType*)
  //    return the ID of the equation.
  //
  // static size_t getNumOfIterationVars(const EquationType*)
  //    return the number of induction variables.
  //
  // static IndexSet getIndices(const EquationType*)
  //    return the indices of the equation.
  //
  // typedef VariableType : the type of the accessed variable
  //
  // static std::vector<Access<VariableTraits<VariableType>::Id>>
  // getAccesses(const EquationType*)
  //    return the accesses done by the equation.
  //
  // static llvm::raw_ostream& dump(const EquationType*, llvm::raw_ostream&)
  //    print debug information.

  using Id = typename EquationType::UnknownEquationTypeError;
};

struct MatchingOptions {
  bool enableLeafNodesSimplification{true};
  bool enableScalarization{true};
  double scalarAccessThreshold{0.5};
};
} // namespace matching

namespace internal {
namespace matching {
/// Represent a generic vectorized entity whose scalar elements
/// can be matched with the scalar elements of other arrays.
/// The relationship is tracked by means of an incidence matrix.
class Matchable {
public:
  explicit Matchable(std::shared_ptr<const IndexSet> matchableIndices);

private:
  const IndexSet &getMatchableIndices() const;

public:
  const IndexSet &getMatched() const;

  const IndexSet &getUnmatched() const;

  /// Check whether all the scalar elements of this array have been matched.
  bool allComponentsMatched() const;

  void setMatch(IndexSet indices);

  void addMatch(const IndexSet &newMatch);

  void removeMatch(const IndexSet &removedMatch);

private:
  std::shared_ptr<const IndexSet> matchableIndices;
  IndexSet matched;
  IndexSet unmatched;
};

/// Identifier of the variable coupled with an optional mask.
template <typename VariableProperty>
class VariableId {
  using Traits = ::marco::modeling::matching::VariableTraits<VariableProperty>;

public:
  using BaseId = typename Traits::Id;

private:
  BaseId baseId;
  std::optional<Point> mask;

public:
  VariableId(BaseId baseId, std::optional<Point> mask)
      : baseId(std::move(baseId)), mask(std::move(mask)) {}

  bool operator==(const VariableId &other) const {
    return baseId == other.baseId && mask == other.mask;
  }

  bool operator!=(const VariableId &other) const { return !(*this == other); }

  bool operator<(const VariableId &other) const {
    if (baseId != other.baseId) {
      return baseId < other.baseId;
    }

    if (mask && !other.mask) {
      return false;
    }

    if (!mask && other.mask) {
      return true;
    }

    if (mask && other.mask) {
      return *mask < *other.mask;
    }

    return false;
  }

  friend llvm::hash_code hash_value(const VariableId &val) {
    return llvm::hash_combine(val.baseId, val.mask);
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const VariableId &obj) {
    os << obj.baseId;

    if (obj.mask) {
      os << " @ " << *obj.mask;
    }

    return os;
  }

  BaseId getBaseId() const { return baseId; }
};

/// Graph node representing a variable.
template <typename VariableProperty>
class VariableVertex : public Matchable, public Dumpable {
public:
  using Property = VariableProperty;
  using Traits = ::marco::modeling::matching::VariableTraits<VariableProperty>;
  using Id = VariableId<VariableProperty>;

private:
  /// The identifier of the variable.
  Id id;

  /// Custom variable property.
  std::shared_ptr<VariableProperty> property;

  /// The indices of the variable.
  std::shared_ptr<const IndexSet> indices;

  /// Whether the node is visible or has been erased.
  bool visible{true};

public:
  explicit VariableVertex(std::shared_ptr<VariableProperty> property)
      : VariableVertex(
            {Traits::getId(property.get()), std::nullopt}, property,
            std::make_shared<const IndexSet>(getIndices(*property))) {}

  explicit VariableVertex(std::shared_ptr<VariableProperty> property,
                          Point mask)
      : VariableVertex({Traits::getId(property.get()), mask}, property,
                       std::make_shared<const IndexSet>(mask)) {}

  explicit VariableVertex(std::shared_ptr<VariableProperty> property,
                          IndexSet mask)
      : VariableVertex({Traits::getId(property.get()), std::nullopt}, property,
                       std::make_shared<const IndexSet>(std::move(mask))) {}

private:
  VariableVertex(Id id, std::shared_ptr<VariableProperty> property,
                 std::shared_ptr<const IndexSet> indices)
      : Matchable(indices), id(std::move(id)), property(std::move(property)),
        indices(std::move(indices)) {
    assert(getRank() > 0 && "Scalar variables must be represented as array "
                            "variables made of one element");

    assert(getUnmatched() == getIndices());
  }

public:
  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    Traits::dump(property.get(), os);
    os << " @ " << *indices;
  }

  VariableProperty &getProperty() {
    assert(property && "Property not set");
    return *property;
  }

  const VariableProperty &getProperty() const {
    assert(property && "Property not set");
    return *property;
  }

  const Id &getId() const { return id; }

  size_t getRank() const { return getIndices().rank(); }

  const IndexSet &getIndices() const {
    assert(indices && "Indices not set");
    assert(!indices->empty() && "Empty indices");
    return *indices;
  }

  std::shared_ptr<const IndexSet> getIndicesPtr() const {
    assert(indices && "Indices not set");
    return indices;
  }

  size_t flatSize() const { return getIndices().flatSize(); }

  VariableVertex withMask(Point mask) const {
    VariableVertex result(property, mask);

    if (auto matched = getMatched().intersect(mask); !matched.empty()) {
      result.addMatch(matched);
    }

    result.visible = visible;
    return result;
  }

  VariableVertex withMask(IndexSet mask) const {
    VariableVertex result(property, mask);
    result.id = id;

    if (auto matched = getMatched().intersect(mask); !matched.empty()) {
      result.addMatch(matched);
    }

    result.visible = visible;
    return result;
  }

  bool isVisible() const { return visible; }

  void setVisibility(bool visibility) { visible = visibility; }

private:
  static size_t getRank(const VariableProperty &p) {
    return Traits::getRank(&p);
  }

  static IndexSet getIndices(const VariableProperty &p) {
    return Traits::getIndices(&p);
  }
};
} // namespace matching
} // namespace internal

namespace matching {
template <typename VariableId>
class Access {
public:
  Access(VariableId variable, std::unique_ptr<AccessFunction> accessFunction)
      : variable(std::move(variable)),
        accessFunction(std::move(accessFunction)) {}

  Access(const Access &other)
      : variable(other.variable),
        accessFunction(other.accessFunction->clone()) {}

  ~Access() = default;

  const VariableId &getVariable() const { return variable; }

  const AccessFunction &getAccessFunction() const {
    assert(accessFunction && "Access function not set");
    assert(accessFunction->getNumOfDims() > 0 &&
           "Access function has no dimension");
    assert(accessFunction->getNumOfResults() > 0 &&
           "Access function has no result");
    return *accessFunction;
  }

private:
  VariableId variable;
  std::unique_ptr<AccessFunction> accessFunction;
};
} // namespace matching

namespace internal::matching {
/// Identifier of the equation coupled with an optional mask.
template <typename EquationProperty>
class EquationId {
  using Traits = ::marco::modeling::matching::EquationTraits<EquationProperty>;

public:
  using BaseId = typename Traits::Id;

private:
  BaseId baseId;
  std::optional<Point> mask;

public:
  EquationId(BaseId baseId, std::optional<Point> mask)
      : baseId(std::move(baseId)), mask(std::move(mask)) {}

  bool operator==(const EquationId &other) const {
    return baseId == other.baseId && mask == other.mask;
  }

  bool operator!=(const EquationId &other) const { return !(*this == other); }

  bool operator<(const EquationId &other) const {
    if (baseId != other.baseId) {
      return baseId < other.baseId;
    }

    if (mask && !other.mask) {
      return false;
    }

    if (!mask && other.mask) {
      return true;
    }

    if (mask && other.mask) {
      return *mask < *other.mask;
    }

    return false;
  }

  friend llvm::hash_code hash_value(const EquationId &val) {
    return llvm::hash_combine(val.baseId, val.mask);
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const EquationId &obj) {
    os << obj.baseId;

    if (obj.mask) {
      os << " @ " << *obj.mask;
    }

    return os;
  }

  BaseId getBaseId() const { return baseId; }
};

/// Graph node representing an equation.
template <typename EquationProperty>
class EquationVertex : public Matchable, public Dumpable {
  using EquationTraits =
      ::marco::modeling::matching::EquationTraits<EquationProperty>;

  using VariableProperty = typename EquationTraits::VariableType;

  using VariableTraits =
      ::marco::modeling::matching::VariableTraits<VariableProperty>;

public:
  using ScalarVariablesMap = llvm::DenseMap<
      typename VariableTraits::Id,
      llvm::DenseMap<Point, typename VariableVertex<VariableProperty>::Id>>;

  using Property = std::shared_ptr<EquationProperty>;
  using Id = EquationId<EquationProperty>;

private:
  /// MLIR context.
  mlir::MLIRContext *context;

  /// The identifier of the equation.
  Id id;

  /// Custom equation property.
  std::shared_ptr<EquationProperty> property;

  /// The indices of the equation.
  std::shared_ptr<const IndexSet> indices;

  /// Whether the node is visible or has been erased.
  bool visible{true};

  /// Scalarized variables.
  std::shared_ptr<ScalarVariablesMap> scalarVariablesMap;

public:
  using OriginalAccess =
      ::marco::modeling::matching::Access<typename VariableTraits::Id>;

  using Access = ::marco::modeling::matching::Access<
      typename VariableVertex<VariableProperty>::Id>;

public:
  explicit EquationVertex(mlir::MLIRContext *context,
                          std::shared_ptr<EquationProperty> property)
      : EquationVertex(
            context, {EquationTraits::getId(property.get()), std::nullopt},
            property, std::make_shared<const IndexSet>(getIndices(*property))) {
  }

  explicit EquationVertex(mlir::MLIRContext *context,
                          std::shared_ptr<EquationProperty> property,
                          Point mask)
      : EquationVertex(context, {EquationTraits::getId(property.get()), mask},
                       property, std::make_shared<const IndexSet>(mask)) {}

  explicit EquationVertex(mlir::MLIRContext *context,
                          std::shared_ptr<EquationProperty> property,
                          IndexSet mask)
      : EquationVertex(
            context, {EquationTraits::getId(property.get()), std::nullopt},
            property, std::make_shared<const IndexSet>(std::move(mask))) {}

private:
  EquationVertex(mlir::MLIRContext *context, Id id,
                 std::shared_ptr<EquationProperty> property,
                 std::shared_ptr<const IndexSet> indices)
      : Matchable(indices), context(context), id(std::move(id)),
        property(std::move(property)), indices(std::move(indices)),
        scalarVariablesMap(std::make_shared<ScalarVariablesMap>()) {
    assert(getNumOfIterationVars() > 0 &&
           "Scalar equations must be represented as array equations made of "
           "one element");

    assert(getUnmatched() == getIndices());
  }

public:
  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    EquationTraits::dump(property.get(), os);
    os << " @ " << *indices;
  }

  EquationProperty &getProperty() {
    assert(property && "Property not set");
    return property;
  }

  const EquationProperty &getProperty() const {
    assert(property && "Property not set");
    return *property;
  }

  std::shared_ptr<EquationProperty> getPropertyPtr() const { return property; }

  void setScalarVariablesMap(
      std::shared_ptr<ScalarVariablesMap> scalarVariablesMap) {
    this->scalarVariablesMap = std::move(scalarVariablesMap);
  }

  ScalarVariablesMap &getScalarVariablesMap() {
    assert(scalarVariablesMap && "Scalar variables map not set");
    return *scalarVariablesMap;
  }

  const ScalarVariablesMap &getScalarVariablesMap() const {
    assert(scalarVariablesMap && "Scalar variables map not set");
    return *scalarVariablesMap;
  }

  const Id &getId() const { return id; }

  size_t getNumOfIterationVars() const { return getIndices().rank(); }

  const IndexSet &getIndices() const {
    assert(indices && "Indices not set");
    assert(!indices->empty() && "Empty indices");
    return *indices;
  }

  std::shared_ptr<const IndexSet> getIndicesPtr() const {
    assert(indices && "Indices not set");
    return indices;
  }

  unsigned int flatSize() const { return getIndices().flatSize(); }

  EquationVertex withMask(Point mask) const {
    EquationVertex result(context, property, mask);

    if (auto matched = getMatched().intersect(mask); !matched.empty()) {
      result.addMatch(matched);
    }

    result.visible = visible;
    result.scalarVariablesMap = scalarVariablesMap;
    return result;
  }

  EquationVertex withMask(IndexSet mask) const {
    EquationVertex result(context, property, mask);
    result.id = id;

    if (auto matched = getMatched().intersect(mask); !matched.empty()) {
      result.addMatch(matched);
    }

    result.visible = visible;
    result.scalarVariablesMap = scalarVariablesMap;
    return result;
  }

  std::vector<Access> getVariableAccesses() const {
    std::vector<OriginalAccess> originalAccesses =
        EquationTraits::getAccesses(property.get());

    std::vector<Access> result;
    IndexSet equationIndices = getIndices();

    for (const OriginalAccess &originalAccess : originalAccesses) {
      auto scalarVariablesIt =
          getScalarVariablesMap().find(originalAccess.getVariable());

      if (scalarVariablesIt != getScalarVariablesMap().end()) {
        IndexSet accessedIndices =
            originalAccess.getAccessFunction().map(equationIndices);

        // If an equation accesses to a scalarized variable, then either the
        // access is constant (e.g., x[0]), or the equation had to be scalarized
        // too. In both cases, the accessed variable indices consist of only one
        // point by construction.
        assert(accessedIndices.flatSize() == 1);

        for (Point point : accessedIndices) {
          const auto &pointsMap = scalarVariablesIt->second;
          auto pointVariableIt = pointsMap.find(point);

          if (pointVariableIt != pointsMap.end()) {
            // Change the linked variable. The access is kept untouched, as the
            // variable indices are restricted but not remapped.
            result.push_back(
                Access(pointVariableIt->second,
                       originalAccess.getAccessFunction().clone()));
          }
        }
      } else {
        result.push_back(Access({originalAccess.getVariable(), std::nullopt},
                                originalAccess.getAccessFunction().clone()));
      }
    }

    return result;
  }

  bool isVisible() const { return visible; }

  void setVisibility(bool visibility) { visible = visibility; }

private:
  static size_t getNumOfIterationVars(const EquationProperty &p) {
    return EquationTraits::getNumOfIterationVars(&p);
  }

  static IndexSet getIndices(const EquationProperty &p) {
    return EquationTraits::getIndices(&p);
  }
};

template <typename Variable, typename Equation>
class Edge : public Dumpable {
public:
  Edge(typename Equation::Id equation, typename Variable::Id variable,
       std::shared_ptr<const IndexSet> equationIndices,
       std::shared_ptr<const IndexSet> variableIndices)
      : equation(std::move(equation)), variable(std::move(variable)),
        incidenceMatrix(equationIndices, variableIndices),
        matchMatrix(equationIndices, variableIndices),
        unmatchMatrix(equationIndices, variableIndices), visible(true) {
    unmatchMatrix = incidenceMatrix;
  }

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Edge\n";
    os << "  - Equation: " << equation << "\n";
    os << "  - Variable: " << variable << "\n";
    os << "  - Incidence matrix:\n" << incidenceMatrix << "\n";
    os << "  - Matched equations: " << getMatched().flattenColumns() << "\n";
    os << "  - Matched variables: " << getMatched().flattenRows() << "\n";
    os << "  - Matched matrix:\n" << getMatched() << "\n";
    os << "  - Unmatched matrix:\n" << getUnmatched() << "\n";
  }

  const MCIM &getIncidenceMatrix() const { return incidenceMatrix; }

  void addMatch(const MCIM &match) {
    matchMatrix += match;
    unmatchMatrix -= match;
  }

  void removeMatch(const MCIM &match) {
    matchMatrix -= match;
    unmatchMatrix += match;
  }

  const MCIM &getMatched() const { return matchMatrix; }

  const MCIM &getUnmatched() const { return unmatchMatrix; }

  bool isVisible() const { return visible; }

  void setVisibility(bool visibility) { visible = visibility; }

  void addAccessFunction(std::unique_ptr<AccessFunction> accessFunction) {
    assert(accessFunction && "Null access function");
    incidenceMatrix.apply(*accessFunction);
    accessFunctions.push_back(std::move(accessFunction));
    unmatchMatrix = incidenceMatrix - matchMatrix;
  }

  auto getAccessFunctions() const {
    return llvm::make_pointee_range(accessFunctions);
  }

private:
  // Equation's ID. Just for debugging purpose
  typename Equation::Id equation;

  // Variable's ID. Just for debugging purpose
  typename Variable::Id variable;

  MCIM incidenceMatrix;
  MCIM matchMatrix;
  MCIM unmatchMatrix;

  bool visible;

  llvm::SmallVector<std::unique_ptr<AccessFunction>> accessFunctions;
};

template <typename Graph, typename Variable, typename Equation>
class BFSStep : public Dumpable {
public:
  using VertexDescriptor = typename Graph::VertexDescriptor;
  using EdgeDescriptor = typename Graph::EdgeDescriptor;

  using VertexProperty = typename Graph::VertexProperty;

  BFSStep(const Graph &graph, VertexDescriptor node, IndexSet candidates)
      : graph(&graph), previous(nullptr), node(std::move(node)),
        candidates(std::move(candidates)), edge(std::nullopt),
        mappedFlow(std::nullopt) {}

  BFSStep(const Graph &graph, std::shared_ptr<BFSStep> previous,
          EdgeDescriptor edge, VertexDescriptor node, IndexSet candidates,
          MCIM mappedFlow)
      : graph(&graph), previous(previous), node(std::move(node)),
        candidates(std::move(candidates)), edge(std::move(edge)),
        mappedFlow(std::move(mappedFlow)) {}

  BFSStep(const BFSStep &other) = default;

  BFSStep(BFSStep &&other) = default;

  ~BFSStep() = default;

  BFSStep &operator=(const BFSStep &other) = default;

  BFSStep &operator=(BFSStep &&other) = default;

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    std::vector<const BFSStep *> path;
    path.push_back(this);

    while (path.back()->hasPrevious()) {
      path.push_back(path.back()->getPrevious());
    }

    for (size_t i = 0, e = path.size(); i < e; ++i) {
      if (i != 0) {
        os << " -> ";
      }

      const BFSStep *step = path[e - i - 1];
      dumpId(os, step->getNode());
      os << " " << step->getCandidates();
    }
  }

  bool hasPrevious() const { return previous != nullptr; }

  const BFSStep *getPrevious() const { return previous.get(); }

  const VertexDescriptor &getNode() const { return node; }

  const IndexSet &getCandidates() const { return candidates; }

  const EdgeDescriptor &getEdge() const {
    assert(edge.has_value());
    return *edge;
  }

  const MCIM &getMappedFlow() const {
    assert(mappedFlow.has_value());
    return *mappedFlow;
  }

private:
  void dumpId(llvm::raw_ostream &os, VertexDescriptor descriptor) const {
    const VertexProperty &nodeProperty = (*graph)[descriptor];

    if (std::holds_alternative<Variable>(nodeProperty)) {
      os << std::get<Variable>(nodeProperty).getId();
    } else {
      os << std::get<Equation>(nodeProperty).getId();
    }
  }

private:
  // Stored for debugging purpose
  const Graph *graph;

  std::shared_ptr<BFSStep> previous;
  VertexDescriptor node;
  IndexSet candidates;
  std::optional<EdgeDescriptor> edge;
  std::optional<MCIM> mappedFlow;
};

template <typename BFSStep>
class Frontier : public Dumpable {
private:
  template <typename T>
  using Container = std::vector<std::shared_ptr<T>>;

public:
  using iterator = typename Container<BFSStep>::iterator;
  using const_iterator = typename Container<BFSStep>::const_iterator;

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Frontier\n";

    for (const auto &step : steps) {
      step->dump(os);
      os << "\n";
    }
  }

  friend void swap(Frontier &first, Frontier &second) {
    using std::swap;

    swap(first.steps, second.steps);
  }

  BFSStep &operator[](size_t index) {
    assert(index < steps.size());
    return *steps[index];
  }

  const BFSStep &operator[](size_t index) const {
    assert(index < steps.size());
    return *steps[index];
  }

  std::shared_ptr<BFSStep> at(size_t index) const {
    assert(index < steps.size());
    return steps[index];
  }

  bool empty() const { return steps.empty(); }

  size_t size() const { return steps.size(); }

  template <typename... Args>
  void emplace_back(Args &&...args) {
    steps.emplace_back(std::make_shared<BFSStep>(std::forward<Args>(args)...));
  }

  void push_back(std::shared_ptr<BFSStep> step) {
    steps.push_back(std::move(step));
  }

  template <typename It>
  void insert(iterator posIt, It beginIt, It endIt) {
    steps.insert(posIt, beginIt, endIt);
  }

  template <typename It>
  void insert(const_iterator posIt, It beginIt, It endIt) {
    steps.insert(posIt, beginIt, endIt);
  }

  void clear() { steps.clear(); }

  void swap(Frontier &other) { steps.swap(other.steps); }

  iterator begin() { return steps.begin(); }

  const_iterator begin() const { return steps.begin(); }

  iterator end() { return steps.end(); }

  const_iterator end() const { return steps.end(); }

private:
  Container<BFSStep> steps;
};

template <typename Graph, typename Variable, typename Equation>
class Flow : public Dumpable {
private:
  using VertexDescriptor = typename Graph::VertexDescriptor;
  using EdgeDescriptor = typename Graph::EdgeDescriptor;

  using VertexProperty = typename Graph::VertexProperty;

public:
  Flow(const Graph &graph, VertexDescriptor source, EdgeDescriptor edge,
       const MCIM &delta)
      : graph(&graph), source(std::move(source)), edge(std::move(edge)),
        delta(std::move(delta)) {
    assert(this->source == this->edge.from || this->source == this->edge.to);
  }

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Flow\n";

    os << "  - Source: ";
    dumpId(os, source);
    os << "\n";

    os << "  - Edge: ";
    dumpId(os, edge.from);
    os << " - ";
    dumpId(os, edge.to);
    os << "\n";

    os << "  - Delta:\n" << delta;
  }

private:
  void dumpId(llvm::raw_ostream &os, VertexDescriptor descriptor) const {
    const VertexProperty &nodeProperty = (*graph)[descriptor];

    if (std::holds_alternative<Variable>(nodeProperty)) {
      os << std::get<Variable>(nodeProperty).getId();
    } else {
      os << std::get<Equation>(nodeProperty).getId();
    }
  }

private:
  // Stored for debugging purpose
  const Graph *graph;

public:
  const VertexDescriptor source;
  const EdgeDescriptor edge;
  const MCIM delta;
};

template <typename Flow>
class AugmentingPath : public Dumpable {
private:
  template <typename T>
  using Container = std::vector<T>;

public:
  using const_iterator = typename Container<Flow>::const_iterator;

  template <typename Flows>
  explicit AugmentingPath(const Flows &flows)
      : flows(flows.begin(), flows.end()) {}

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Augmenting path\n";

    for (const auto &flow : flows) {
      os << "  ";
      flow.dump(os);
      os << "\n";
    }
  }

  const Flow &operator[](size_t index) const {
    assert(index < flows.size());
    return flows[index];
  }

  const_iterator begin() const { return flows.begin(); }

  const_iterator end() const { return flows.end(); }

private:
  Container<Flow> flows;
};

/// The class represents to which indices of a variable an equation is matched.
template <typename EquationProperty, typename VariableProperty>
class MatchingSolution {
public:
  using EquationId =
      typename modeling::matching::EquationTraits<EquationProperty>::Id;

  using VariableId =
      typename modeling::matching::VariableTraits<VariableProperty>::Id;

  MatchingSolution(EquationId equation, IndexSet equationIndices,
                   VariableId variable, IndexSet variableIndices)
      : equation(std::move(equation)),
        equationIndices(std::move(equationIndices)),
        variable(std::move(variable)),
        variableIndices(std::move(variableIndices)) {}

  const EquationId &getEquation() const { return equation; }

  const IndexSet &getEquationIndices() const { return equationIndices; }

  const VariableId &getVariable() const { return variable; }

  const IndexSet &getVariableIndices() const { return variableIndices; }

private:
  EquationId equation;
  IndexSet equationIndices;
  VariableId variable;
  IndexSet variableIndices;
};

template <typename Derived, typename VariableProperty,
          typename EquationProperty>
class MatchingGraphCRTP : public Dumpable {
public:
  using Variable = VariableVertex<VariableProperty>;
  using Equation = EquationVertex<EquationProperty>;
  using Vertex = std::variant<Variable, Equation>;
  using Edge = Edge<Variable, Equation>;

private:
  using Graph = UndirectedGraph<Vertex, Edge>;

public:
  using VertexDescriptor = typename Graph::VertexDescriptor;
  using EdgeDescriptor = typename Graph::EdgeDescriptor;

protected:
  using VertexIterator = typename Graph::VertexIterator;
  using EdgeIterator = typename Graph::EdgeIterator;

  using VisibleIncidentEdgeIterator =
      typename Graph::FilteredIncidentEdgeIterator;

  using TraversableEdges =
      llvm::MapVector<VertexDescriptor, llvm::SetVector<EdgeDescriptor>>;

  using BFSStep = BFSStep<Graph, Variable, Equation>;
  using Frontier = Frontier<BFSStep>;
  using Flow = Flow<Graph, Variable, Equation>;
  using AugmentingPath = AugmentingPath<Flow>;

public:
  using VariableIterator = typename Graph::FilteredVertexIterator;
  using EquationIterator = typename Graph::FilteredVertexIterator;

  using Access = modeling::matching::Access<typename Variable::Id>;
  using MatchingOptions = modeling::matching::MatchingOptions;
  using MatchingSolution = MatchingSolution<EquationProperty, VariableProperty>;

private:
  mlir::MLIRContext *context;
  std::unique_ptr<Graph> graph;

  // Maps used for faster lookups.
  llvm::DenseMap<typename Variable::Id, VertexDescriptor> variablesMap;
  llvm::DenseMap<typename Equation::Id, VertexDescriptor> equationsMap;

  // Multithreading.
  mutable std::mutex mutex;

public:
  explicit MatchingGraphCRTP(mlir::MLIRContext *context)
      : context(context), graph(std::make_unique<Graph>()) {}

  MatchingGraphCRTP(const MatchingGraphCRTP &other) = delete;

  MatchingGraphCRTP(MatchingGraphCRTP &&other) noexcept {
    std::lock_guard<std::mutex> lockGuard(other.mutex);

    context = std::move(other.context);
    graph = std::move(other.graph);
    variablesMap = std::move(other.variablesMap);
    equationsMap = std::move(other.equationsMap);
  }

  ~MatchingGraphCRTP() override = default;

  MatchingGraphCRTP &operator=(const MatchingGraphCRTP &other) = delete;

  MatchingGraphCRTP &operator=(MatchingGraphCRTP &&other) noexcept {
    std::lock_guard<std::mutex> lockGuard(other.mutex);

    context = std::move(other.context);
    graph = std::move(other.graph);
    variablesMap = std::move(other.variablesMap);
    equationsMap = std::move(other.equationsMap);

    return *this;
  }

  virtual Derived newInstance() const = 0;

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "--------------------------------------------------\n";
    os << "Matching graph\n";
    os << "- Nodes:\n";

    for (auto vertexDescriptor : llvm::make_range(
             getBaseGraph().verticesBegin(), getBaseGraph().verticesEnd())) {
      if (isVariable(vertexDescriptor)) {
        const Variable &variable = getVariableFromDescriptor(vertexDescriptor);

        os << "Variable\n";
        os << "  - ID: " << variable.getId() << "\n";
        os << "  - Info: ";
        variable.dump(os);
        os << "\n";
        os << "  - Rank: " << variable.getRank() << "\n";
        os << "  - Indices: " << variable.getIndices() << "\n";
        os << "  - Matched: " << variable.getMatched() << "\n";

        for (auto edgeDescriptor : llvm::make_range(
                 edgesBegin(vertexDescriptor), edgesEnd(vertexDescriptor))) {
          const Equation &equation =
              getEquationFromDescriptor(edgeDescriptor.to);
          const Edge &edge = getBaseGraph()[edgeDescriptor];
          IndexSet matchedVariables = edge.getMatched().flattenRows();

          if (!matchedVariables.empty()) {
            IndexSet matchedEquations = edge.getMatched().flattenColumns();
            os << "  - Match: ";
            os << matchedVariables << " -> " << equation.getId() << " "
               << matchedEquations << "\n";
          }
        }
      } else {
        const Equation &equation = getEquationFromDescriptor(vertexDescriptor);

        os << "Equation\n";
        os << "  - ID: " << equation.getId() << "\n";
        os << "  - Info: ";
        equation.dump(os);
        os << "\n";
        os << "  - Indices: " << equation.getIndices() << "\n";
        os << "  - Matched: " << equation.getMatched() << "\n";

        for (auto edgeDescriptor : llvm::make_range(
                 edgesBegin(vertexDescriptor), edgesEnd(vertexDescriptor))) {
          const Variable &variable =
              getVariableFromDescriptor(edgeDescriptor.to);
          const Edge &edge = getBaseGraph()[edgeDescriptor];
          IndexSet matchedEquations = edge.getMatched().flattenColumns();

          if (!matchedEquations.empty()) {
            IndexSet matchedVariables = edge.getMatched().flattenRows();
            os << "  - Match: ";
            os << matchedEquations << " -> " << variable.getId() << " "
               << matchedVariables << "\n";
          }
        }
      }

      os << "\n";
    }

    for (auto edgeDescriptor : llvm::make_range(getBaseGraph().edgesBegin(),
                                                getBaseGraph().edgesEnd())) {
      getBaseGraph()[edgeDescriptor].dump(os);
      os << "\n";
    }

    os << "--------------------------------------------------\n";
  }

  mlir::MLIRContext *getContext() const {
    assert(context && "MLIR context not set");
    return context;
  }

  Graph &getBaseGraph() {
    assert(graph && "Base graph not set");
    return *graph;
  }

  const Graph &getBaseGraph() const {
    assert(graph && "Base graph not set");
    return *graph;
  }

  bool hasVariable(typename Variable::Id id) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return hasVariableWithId(id);
  }

  VariableProperty *getVariable(typename Variable::Id id) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariablePropertyFromId(id);
  }

  const VariableProperty *getVariable(typename Variable::Id id) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariablePropertyFromId(id);
  }

  Variable *getVariable(VertexDescriptor descriptor) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariableFromDescriptor(descriptor);
  }

  const Variable &getVariable(VertexDescriptor descriptor) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariableFromDescriptor(descriptor);
  }

  VariableIterator variablesBegin() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariablesBeginIt();
  }

  VariableIterator variablesEnd() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariablesEndIt();
  }

  void addVariable(VariableProperty property) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    Variable variable(std::make_shared<VariableProperty>(std::move(property)));
    addVariable(std::move(variable));
  }

private:
  void addVariable(Variable variable) {
    auto id = variable.getId();
    assert(!hasVariableWithId(id) && "Already existing variable");

    VertexDescriptor variableDescriptor =
        getBaseGraph().addVertex(std::move(variable));

    variablesMap[id] = variableDescriptor;
  }

public:
  bool hasEquation(typename Equation::Id id) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return hasEquationWithId(id);
  }

  EquationProperty *getEquation(typename Equation::Id id) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationPropertyFromId(id);
  }

  const EquationProperty *getEquation(typename Equation::Id id) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationPropertyFromId(id);
  }

  Equation &getEquation(VertexDescriptor descriptor) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationFromDescriptor(descriptor);
  }

  const Equation &getEquation(VertexDescriptor descriptor) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationFromDescriptor(descriptor);
  }

  EquationIterator equationsBegin() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationsBeginIt();
  }

  EquationIterator equationsEnd() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationsEndIt();
  }

  void addEquation(EquationProperty property) {
    std::unique_lock<std::mutex> lockGuard(mutex);

    Equation equation(getContext(),
                      std::make_shared<EquationProperty>(std::move(property)));

    VertexDescriptor equationDescriptor = addEquation(std::move(equation));
    discoverAccesses(equationDescriptor);
  }

private:
  VertexDescriptor addEquation(Equation equation) {
    [[maybe_unused]] auto id = equation.getId();

    // Insert the equation into the graph.
    assert(!hasEquationWithId(id) && "Already existing equation");
    VertexDescriptor equationDescriptor = getBaseGraph().addVertex(equation);
    equationsMap[id] = equationDescriptor;

    return equationDescriptor;
  }

  void discoverAccesses(VertexDescriptor equationDescriptor) {
    const Equation &equation = getEquationFromDescriptor(equationDescriptor);

    // Add an edge for each accessed variable.
    llvm::DenseMap<VertexDescriptor, EdgeDescriptor> edges;

    for (const auto &access : equation.getVariableAccesses()) {
      if (auto variableDescriptor =
              getVariableDescriptorFromId(access.getVariable())) {
        Variable &variable = getVariableFromDescriptor(*variableDescriptor);

        if (edges.find(*variableDescriptor) == edges.end()) {
          Edge edge(equation.getId(), variable.getId(),
                    equation.getIndicesPtr(), variable.getIndicesPtr());

          edges[*variableDescriptor] = getBaseGraph().addEdge(
              equationDescriptor, *variableDescriptor, std::move(edge));
        }

        Edge &edge = getBaseGraph()[edges[*variableDescriptor]];
        edge.addAccessFunction(access.getAccessFunction().clone());
      }
    }
  }

public:
  auto getEdgesBegin(VertexDescriptor vertex) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getBaseGraph().outgoingEdgesBegin(vertex);
  }

  auto getEdgesEnd(VertexDescriptor vertex) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getBaseGraph().outgoingEdgesEnd(vertex);
  }

  auto getLinkedNodesBegin(VertexDescriptor vertex) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getBaseGraph().linkedVerticesBegin(vertex);
  }

  auto getLinkedNodesEnd(VertexDescriptor vertex) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getBaseGraph().linkedVerticesEnd(vertex);
  }

  /// Get the total amount of scalar variables inside the graph.
  size_t getNumberOfScalarVariables() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    size_t result = 0;

    auto variables =
        llvm::make_range(getVariablesBeginIt(), getVariablesEndIt());

    for (VertexDescriptor variableDescriptor : variables) {
      result += getVariableFromDescriptor(variableDescriptor).flatSize();
    }

    return result;
  }

  /// Get the total amount of scalar equations inside the graph.
  /// With "scalar equations" we mean the ones generated by unrolling
  /// the loops defining them.
  size_t getNumberOfScalarEquations() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    size_t result = 0;

    auto equations =
        llvm::make_range(getEquationsBeginIt(), getEquationsEndIt());

    for (VertexDescriptor equationDescriptor : equations) {
      result += getEquationFromDescriptor(equationDescriptor).flatSize();
    }

    return result;
  }

  // Warning: highly inefficient, use for testing purposes only.
  bool hasEdge(typename Equation::Id equationId,
               typename Variable::Id variableId) const {
    std::lock_guard<std::mutex> lockGuard(mutex);

    if (findEdge<Equation, Variable>(equationId, variableId).first) {
      return true;
    }

    return findEdge<Variable, Equation>(variableId, equationId).first;
  }

private:
  using MatchingSolutions =
      llvm::MapVector<typename Equation::Id::BaseId,
                      llvm::MapVector<typename Variable::Id::BaseId,
                                      std::pair<IndexSet, IndexSet>>>;

public:
  /// Compute a maximum matching between variables and equations.
  /// Returns true if a full match is obtained, false otherwise.
  bool match(llvm::SmallVectorImpl<MatchingSolution> &result,
             const MatchingOptions &options = MatchingOptions()) {
    assert(options.scalarAccessThreshold >= 0 &&
           "The scalarization threshold must be greater or equal to zero");

    std::lock_guard<std::mutex> lockGuard(mutex);
    MatchingSolutions matchingSolutions;

    if (!match(matchingSolutions, options)) {
      return false;
    }

    for (const auto &equation : matchingSolutions) {
      for (const auto &variable : equation.second) {
        result.emplace_back(equation.first, variable.second.first,
                            variable.first, variable.second.second);
      }
    }

    return true;
  }

private:
  bool match(MatchingSolutions &matchingSolutions,
             const MatchingOptions &options) {
    if (options.enableLeafNodesSimplification) {
      matchLeafNodes();
      collectMatches(matchingSolutions);
    }

    // Operate only on the nodes that are not fully matched.
    Derived unmatchedGraph = getUnmatchedGraph();

    if (options.enableScalarization) {
      if (auto scalarGraph =
              unmatchedGraph.getScalarGraph(options.scalarAccessThreshold)) {
        MatchingOptions disabledScalarizationOptions(options);
        disabledScalarizationOptions.enableScalarization = false;

        return scalarGraph->match(matchingSolutions,
                                  disabledScalarizationOptions);
      }
    }

    // Apply the generic matching algorithm.
    if (!unmatchedGraph.applyHK()) {
      return false;
    }

    unmatchedGraph.collectMatches(matchingSolutions);
    return true;
  }

  /// Build a graph containing only the nodes that are not fully matched.
  Derived getUnmatchedGraph() const {
    Derived result = this->newInstance();

    for (VertexDescriptor variableDescriptor :
         llvm::make_range(getVariablesBeginIt(), getVariablesEndIt())) {
      const Variable &variable = getVariableFromDescriptor(variableDescriptor);

      if (variable.getUnmatched().empty()) {
        continue;
      }

      result.addVariable(variable.withMask(variable.getUnmatched()));
    }

    for (VertexDescriptor equationDescriptor :
         llvm::make_range(getEquationsBeginIt(), getEquationsEndIt())) {
      const Equation &equation = getEquationFromDescriptor(equationDescriptor);

      if (equation.getUnmatched().empty()) {
        continue;
      }

      auto newEquationDescriptor =
          result.addEquation(equation.withMask(equation.getUnmatched()));

      result.discoverAccesses(newEquationDescriptor);
    }

    return result;
  }

  void collectMatches(MatchingSolutions &result) const {
    for (EdgeDescriptor edgeDescriptor :
         llvm::make_range(edgesBegin(), edgesEnd())) {
      const Edge &edge = (*graph)[edgeDescriptor];

      if (edge.getMatched().empty()) {
        continue;
      }

      VertexDescriptor equationDescriptor = isEquation(edgeDescriptor.from)
                                                ? edgeDescriptor.from
                                                : edgeDescriptor.to;

      VertexDescriptor variableDescriptor = isVariable(edgeDescriptor.from)
                                                ? edgeDescriptor.from
                                                : edgeDescriptor.to;

      const Equation &equation = getEquationFromDescriptor(equationDescriptor);
      const Variable &variable = getVariableFromDescriptor(variableDescriptor);

      auto equationId = equation.getId().getBaseId();
      auto variableId = variable.getId().getBaseId();

      auto &match = result[equationId][variableId];
      match.first += edge.getMatched().flattenColumns();
      match.second += edge.getMatched().flattenRows();
    }
  }

  /// Check if a variable with a given ID exists.
  bool hasVariableWithId(typename Variable::Id id) const {
    return variablesMap.find(id) != variablesMap.end();
  }

  bool isVariable(VertexDescriptor vertex) const {
    return std::holds_alternative<Variable>(getBaseGraph()[vertex]);
  }

  std::optional<VertexDescriptor>
  getVariableDescriptorFromId(typename Variable::Id id) const {
    auto it = variablesMap.find(id);

    if (it == variablesMap.end()) {
      return std::nullopt;
    }

    return it->second;
  }

  VariableProperty *getVariablePropertyFromId(typename Variable::Id id) {
    if (auto descriptor = getVariableDescriptorFromId(id)) {
      return &getVariableFromDescriptor(*descriptor).getProperty();
    }

    return nullptr;
  }

  const VariableProperty *
  getVariablePropertyFromId(typename Variable::Id id) const {
    if (auto descriptor = getVariableDescriptorFromId(id)) {
      return &getVariableFromDescriptor(*descriptor).getProperty();
    }

    return nullptr;
  }

  Variable &getVariableFromDescriptor(VertexDescriptor descriptor) {
    Vertex &vertex = getBaseGraph()[descriptor];
    assert(std::holds_alternative<Variable>(vertex));
    return std::get<Variable>(vertex);
  }

  const Variable &getVariableFromDescriptor(VertexDescriptor descriptor) const {
    const Vertex &vertex = getBaseGraph()[descriptor];
    assert(std::holds_alternative<Variable>(vertex));
    return std::get<Variable>(vertex);
  }

  /// Check if an equation with a given ID exists.
  bool hasEquationWithId(typename Equation::Id id) const {
    return equationsMap.find(id) != equationsMap.end();
  }

  bool isEquation(VertexDescriptor vertex) const {
    return std::holds_alternative<Equation>(getBaseGraph()[vertex]);
  }

  std::optional<VertexDescriptor>
  getEquationDescriptorFromId(typename Equation::Id id) const {
    auto it = equationsMap.find(id);

    if (it == equationsMap.end()) {
      return std::nullopt;
    }

    return it->second;
  }

  EquationProperty *getEquationPropertyFromId(typename Equation::Id id) {
    if (auto descriptor = getEquationDescriptorFromId(id)) {
      return &getEquationFromDescriptor(*descriptor).getProperty();
    }

    return nullptr;
  }

  const EquationProperty *
  getEquationPropertyFromId(typename Equation::Id id) const {
    if (auto descriptor = getEquationDescriptorFromId(id)) {
      return &getEquationFromDescriptor(*descriptor).getProperty();
    }

    return nullptr;
  }

  Equation &getEquationFromDescriptor(VertexDescriptor descriptor) {
    Vertex &vertex = getBaseGraph()[descriptor];
    assert(std::holds_alternative<Equation>(vertex));
    return std::get<Equation>(vertex);
  }

  const Equation &getEquationFromDescriptor(VertexDescriptor descriptor) const {
    const Vertex &vertex = getBaseGraph()[descriptor];
    assert(std::holds_alternative<Equation>(vertex));
    return std::get<Equation>(vertex);
  }

  /// Get the begin iterator for the variables of the graph.
  VariableIterator getVariablesBeginIt() const {
    auto filter = [](const Vertex &vertex) -> bool {
      return std::holds_alternative<Variable>(vertex);
    };

    return getBaseGraph().verticesBegin(filter);
  }

  /// Get the end iterator for the variables of the graph.
  VariableIterator getVariablesEndIt() const {
    auto filter = [](const Vertex &vertex) -> bool {
      return std::holds_alternative<Variable>(vertex);
    };

    return getBaseGraph().verticesEnd(filter);
  }

  /// Get the begin iterator for the equations of the graph.
  EquationIterator getEquationsBeginIt() const {
    auto filter = [](const Vertex &vertex) -> bool {
      return std::holds_alternative<Equation>(vertex);
    };

    return getBaseGraph().verticesBegin(filter);
  }

  /// Get the end iterator for the equations of the graph.
  EquationIterator getEquationsEndIt() const {
    auto filter = [](const Vertex &vertex) -> bool {
      return std::holds_alternative<Equation>(vertex);
    };

    return getBaseGraph().verticesEnd(filter);
  }

  /// Check if all the scalar variables and equations have been matched.
  bool allNodesMatched() const {
    auto allComponentsMatchedFn = [](const auto &obj) {
      return obj.allComponentsMatched();
    };

    return mlir::succeeded(mlir::failableParallelForEach(
        getContext(), getBaseGraph().verticesBegin(),
        getBaseGraph().verticesEnd(),
        [&](VertexDescriptor vertex) -> mlir::LogicalResult {
          return mlir::LogicalResult::success(
              std::visit(allComponentsMatchedFn, getBaseGraph()[vertex]));
        }));
  }

  size_t getVertexVisibilityDegree(VertexDescriptor vertex) const {
    auto edges =
        llvm::make_range(visibleEdgesBegin(vertex), visibleEdgesEnd(vertex));
    return std::distance(edges.begin(), edges.end());
  }

  size_t getVertexVisibilityDegreeUpTo(VertexDescriptor vertex,
                                       size_t limit) const {
    auto edgesIt = visibleEdgesBegin(vertex);
    auto edgesEndIt = visibleEdgesEnd(vertex);

    size_t result = 0;

    while (edgesIt != edgesEndIt && result < limit) {
      ++result;
      ++edgesIt;
    }

    return result;
  }

  void remove(VertexDescriptor vertex) {
    std::visit([](auto &obj) -> void { obj.setVisibility(false); },
               getBaseGraph()[vertex]);
  }

  // Warning: highly inefficient, use for testing purposes only.
  template <typename From, typename To>
  std::pair<bool, EdgeIterator> findEdge(typename From::Id from,
                                         typename To::Id to) const {
    auto edges = llvm::make_range(getBaseGraph().edgesBegin(),
                                  getBaseGraph().edgesEnd());

    EdgeIterator it =
        std::find_if(edges.begin(), edges.end(), [&](const EdgeDescriptor &e) {
          const Vertex &source = getBaseGraph()[e.from];
          const Vertex &target = getBaseGraph()[e.to];

          if (!std::holds_alternative<From>(source) ||
              !std::holds_alternative<To>(target)) {
            return false;
          }

          return std::get<From>(source).getId() == from &&
                 std::get<To>(target).getId() == to;
        });

    return std::make_pair(it != edges.end(), it);
  }

  auto edgesBegin() const { return getBaseGraph().edgesBegin(); }

  auto edgesEnd() const { return getBaseGraph().edgesEnd(); }

  auto edgesBegin(VertexDescriptor vertex) const {
    return getBaseGraph().outgoingEdgesBegin(vertex);
  }

  auto edgesEnd(VertexDescriptor vertex) const {
    return getBaseGraph().outgoingEdgesEnd(vertex);
  }

  VisibleIncidentEdgeIterator visibleEdgesBegin(VertexDescriptor vertex) const {
    auto filter = [&](const Edge &edge) -> bool { return edge.isVisible(); };

    return getBaseGraph().outgoingEdgesBegin(vertex, filter);
  }

  VisibleIncidentEdgeIterator visibleEdgesEnd(VertexDescriptor vertex) const {
    auto filter = [&](const Edge &edge) -> bool { return edge.isVisible(); };

    return getBaseGraph().outgoingEdgesEnd(vertex, filter);
  }

  EdgeDescriptor getFirstOutVisibleEdge(VertexDescriptor vertex) const {
    auto edges =
        llvm::make_range(visibleEdgesBegin(vertex), visibleEdgesEnd(vertex));
    assert(edges.begin() != edges.end() && "Vertex doesn't belong to any edge");
    return *edges.begin();
  }

  void remove(EdgeDescriptor edge) {
    getBaseGraph()[edge].setVisibility(false);
  }

  /// Apply the simplification algorithm in order to perform all
  /// the obligatory matches, that is the variables and equations
  /// having only one incident edge.
  void matchLeafNodes() {
    LLVM_DEBUG(llvm::dbgs() << "Matching leaf nodes\n");

    // Vertices that are candidate for the first simplification phase.
    // They are the ones having only one incident edge.
    std::list<VertexDescriptor> candidates;
    collectSimplifiableNodes(candidates);

    // Check that the list of simplifiable nodes does not contain
    // duplicates.
    assert(llvm::all_of(candidates,
                        [&](const VertexDescriptor &vertex) {
                          return llvm::count(candidates, vertex) == 1;
                        }) &&
           "Duplicates found in the list of simplifiable nodes");

    // Iterate on the candidate vertices and apply the simplification algorithm
    auto isVisibleFn = [](const auto &obj) -> bool { return obj.isVisible(); };

    auto allComponentsMatchedFn = [](const auto &vertex) -> bool {
      return vertex.allComponentsMatched();
    };

    while (!candidates.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "Leaf vertices:\n";

        for (VertexDescriptor vertexDescriptor : candidates) {
          std::visit(
              [&](const auto &vertex) {
                llvm::dbgs() << "  - " << vertex.getId() << "\n";
              },
              (*graph)[vertexDescriptor]);
        }
      });

      VertexDescriptor v1 = candidates.front();
      candidates.pop_front();

      LLVM_DEBUG({
        std::visit(
            [&](const auto &vertex) {
              llvm::dbgs() << "First vertex: " << vertex.getId() << "\n";
            },
            (*graph)[v1]);
      });

      if (const Vertex &v = getBaseGraph()[v1]; !std::visit(isVisibleFn, v)) {
        // The current node, which initially had one and only one incident
        // edge, has been removed by simplifications performed in the previous
        // iterations. We could just remove the vertex while the edge was
        // removed, but that would have required iterating over the whole
        // candidates list, thus worsening the overall complexity of the
        // algorithm.

        assert(std::visit(allComponentsMatchedFn, v));
        LLVM_DEBUG(llvm::dbgs() << "First vertex not visible\n");
        continue;
      }

      EdgeDescriptor edgeDescriptor = getFirstOutVisibleEdge(v1);
      Edge &edge = getBaseGraph()[edgeDescriptor];

      VertexDescriptor v2 =
          edgeDescriptor.from == v1 ? edgeDescriptor.to : edgeDescriptor.from;

      LLVM_DEBUG({
        std::visit(
            [&](const auto &vertex) {
              llvm::dbgs() << "Second vertex: " << vertex.getId() << "\n";
            },
            (*graph)[v2]);
      });

      Variable &variable = isVariable(v1) ? getVariableFromDescriptor(v1)
                                          : getVariableFromDescriptor(v2);
      Equation &equation = isEquation(v1) ? getEquationFromDescriptor(v1)
                                          : getEquationFromDescriptor(v2);

      IndexSet equationFilter = equation.getUnmatched();
      IndexSet variableFilter = variable.getUnmatched();

      LLVM_DEBUG(llvm::dbgs() << "Equation filter: " << equationFilter << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Variable filter: " << variableFilter << "\n");

      MCIM filteredUnmatchedMatrix = edge.getUnmatched()
                                         .filterRows(equationFilter)
                                         .filterColumns(variableFilter);

      LLVM_DEBUG(llvm::dbgs() << "Filtered unmatched matrix:\n"
                              << filteredUnmatchedMatrix << "\n");

      auto matchOptions =
          internal::solveLocalMatchingProblem(filteredUnmatchedMatrix);

      LLVM_DEBUG(llvm::dbgs()
                 << "Match options: " << matchOptions.size() << "\n");

      // The simplification steps is executed only in case of a single
      // matching option. In case of multiple ones, in fact, the choice
      // would be arbitrary and may affect the feasibility of the matching
      // problem.

      if (matchOptions.size() == 1) {
        const MCIM &match = matchOptions[0];
        edge.addMatch(match);

        IndexSet variableMatch = match.flattenRows();
        IndexSet equationMatch = match.flattenColumns();

        variable.addMatch(variableMatch);
        equation.addMatch(equationMatch);

        bool shouldRemoveOppositeNode =
            std::visit(allComponentsMatchedFn, getBaseGraph()[v2]);

        // Remove the edge and the current candidate vertex.
        remove(edgeDescriptor);
        remove(v1);

        if (shouldRemoveOppositeNode) {
          // When a node is removed, then also its incident edges are
          // removed. This can lead to new obliged matches, like in the
          // following example:
          //        |-- v3 ----
          // v1 -- v2         |
          //        |-- v4 -- v5
          // v1 is the current candidate and thus is removed.
          // v2 is removed because fully matched.
          // v3 and v4 become new candidates for the simplification pass.

          for (EdgeDescriptor e :
               llvm::make_range(getBaseGraph().outgoingEdgesBegin(v2),
                                getBaseGraph().outgoingEdgesEnd(v2))) {
            remove(e);

            VertexDescriptor v = e.from == v2 ? e.to : e.from;

            if (!std::visit(isVisibleFn, getBaseGraph()[v])) {
              continue;
            }

            size_t visibilityDegree = getVertexVisibilityDegreeUpTo(v, 2);

            if (visibilityDegree == 0) {
              // 'v' will also be present for sure in the candidates list.
              // However, having no outgoing edge, we now must remove it.
              remove(v);
            } else if (visibilityDegree == 1) {
              candidates.push_back(v);
            }
          }

          // Remove the v2 vertex.
          remove(v2);
        } else {
          // When an edge is removed but one of its vertices survives, we must
          // check if the remaining vertex has an obliged match.

          size_t visibilityDegree = getVertexVisibilityDegreeUpTo(v2, 2);

          if (visibilityDegree == 1) {
            candidates.push_back(v2);
          }
        }
      }
    }
  }

  /// Collect the list of vertices with exactly one incident edge.
  /// The function returns 'false' if there exist a node with no incident
  /// edges (which would make the matching process to fail in aby case).
  void collectSimplifiableNodes(std::list<VertexDescriptor> &nodes) const {
    std::mutex resultMutex;

    auto collectFn = [&](VertexDescriptor vertex) {
      size_t incidentEdges = getVertexVisibilityDegreeUpTo(vertex, 2);

      if (incidentEdges == 1) {
        std::lock_guard<std::mutex> resultLockGuard(resultMutex);
        nodes.push_back(vertex);
      }
    };

    mlir::parallelForEach(getContext(), getBaseGraph().verticesBegin(),
                          getBaseGraph().verticesEnd(), collectFn);
  }

  /// Apply an array-aware variant of the Hopcroft-Karp algorithm.
  /// Returns true if a full match is obtained.
  bool applyHK() {
    if (allNodesMatched()) {
      return true;
    }

    bool success;
    bool complete;
    TraversableEdges traversableEdges;

    // Compute the initial set of traversable edges.
    collectTraversableEdges(traversableEdges);

    do {
      success = matchIteration(traversableEdges);
      complete = allNodesMatched();

      LLVM_DEBUG({
        llvm::dbgs() << "Match iteration completed\n";
        dump(llvm::dbgs());
      });
    } while (success && !complete);

    LLVM_DEBUG({
      if (success) {
        llvm::dbgs() << "Matching completed successfully\n";
      } else {
        llvm::dbgs() << "Matching failed\n";
      }
    });

    return complete;
  }

  void collectTraversableEdges(TraversableEdges &traversableEdges) const {
    std::mutex mutex;

    mlir::parallelForEach(
        getContext(), edgesBegin(), edgesEnd(),
        [&](EdgeDescriptor edgeDescriptor) {
          const Edge &edge = getBaseGraph()[edgeDescriptor];

          if (isEquation(edgeDescriptor.from)) {
            if (!edge.getMatched().empty()) {
              std::lock_guard lock(mutex);
              traversableEdges[edgeDescriptor.to].insert(edgeDescriptor);
            }

            if (!edge.getUnmatched().empty()) {
              std::lock_guard lock(mutex);
              traversableEdges[edgeDescriptor.from].insert(edgeDescriptor);
            }
          } else {
            if (!edge.getMatched().empty()) {
              std::lock_guard lock(mutex);
              traversableEdges[edgeDescriptor.from].insert(edgeDescriptor);
            }

            if (!edge.getUnmatched().empty()) {
              std::lock_guard lock(mutex);
              traversableEdges[edgeDescriptor.to].insert(edgeDescriptor);
            }
          }
        });
  }

  std::optional<Derived> getScalarGraph(double scalarAccessThreshold) const {
    // Determine which variables should be scalarized.
    llvm::DenseSet<VertexDescriptor> toScalarize;
    collectVariablesToScalarize(toScalarize, scalarAccessThreshold);

    if (toScalarize.empty()) {
      return std::nullopt;
    }

    // Build the graph with scalarized variables.
    Derived scalarizedGraph = this->newInstance();

    auto scalarVariablesMap =
        std::make_shared<typename Equation::ScalarVariablesMap>();

    // Add the variables.
    for (VertexDescriptor variableDescriptor :
         llvm::make_range(getVariablesBeginIt(), getVariablesEndIt())) {
      const Variable &variable = getVariableFromDescriptor(variableDescriptor);

      if (toScalarize.contains(variableDescriptor)) {
        for (Point point : variable.getIndices()) {
          auto scalarizedVariable = variable.withMask(point);

          auto baseId =
              modeling::matching::VariableTraits<VariableProperty>::getId(
                  &variable.getProperty());

          (*scalarVariablesMap)[baseId].insert(
              {point, scalarizedVariable.getId()});

          scalarizedGraph.addVariable(std::move(scalarizedVariable));
        }
      } else {
        scalarizedGraph.addVariable(variable);
      }
    }

    // Add the equations.
    for (VertexDescriptor equationDescriptor :
         llvm::make_range(getEquationsBeginIt(), getEquationsEndIt())) {
      const Equation &equation = getEquationFromDescriptor(equationDescriptor);

      // Check if the equation accesses a scalarized variable in a non-constant
      // way. If that is the case, then the equation needs to be scalarized too.
      bool shouldScalarize = llvm::any_of(
          equation.getVariableAccesses(), [&](const Access &access) {
            if (!scalarVariablesMap->contains(
                    access.getVariable().getBaseId())) {
              return false;
            }

            return !access.getAccessFunction()
                        .template isa<AccessFunctionAffineConstant>();
          });

      if (shouldScalarize) {
        for (Point point : equation.getIndices()) {
          Equation scalarEquation(getContext(), equation.getPropertyPtr(),
                                  point);

          scalarEquation.setScalarVariablesMap(scalarVariablesMap);

          VertexDescriptor newEquationDescriptor =
              scalarizedGraph.addEquation(std::move(scalarEquation));

          scalarizedGraph.discoverAccesses(newEquationDescriptor);
        }
      } else {
        Equation clonedEquation = equation;
        clonedEquation.setScalarVariablesMap(scalarVariablesMap);

        VertexDescriptor newEquationDescriptor =
            scalarizedGraph.addEquation(std::move(clonedEquation));

        scalarizedGraph.discoverAccesses(newEquationDescriptor);
      }
    }

    return scalarizedGraph;
  }

  void collectVariablesToScalarize(
      llvm::DenseSet<VertexDescriptor> &variableDescriptors,
      double scalarAccessThreshold) const {
    std::mutex mutex;

    mlir::parallelForEach(
        getContext(), getVariablesBeginIt(), getVariablesEndIt(),
        [&](VertexDescriptor variableDescriptor) {
          if (shouldVariableBeScalarized(variableDescriptor,
                                         scalarAccessThreshold)) {
            std::lock_guard<std::mutex> lock(mutex);
            variableDescriptors.insert(variableDescriptor);
          }
        });
  }

  bool shouldVariableBeScalarized(VertexDescriptor variableDescriptor,
                                  double scalarAccessThreshold) const {
    const Variable &variable = getVariableFromDescriptor(variableDescriptor);

    size_t variableSize = variable.getIndices().flatSize();

    if (variableSize <= 1) {
      return false;
    }

    return getVariableScalarAccessFactor(variableDescriptor) >=
           scalarAccessThreshold;
  }

  double
  getVariableScalarAccessFactor(VertexDescriptor variableDescriptor) const {
    const Variable &variable = getVariableFromDescriptor(variableDescriptor);
    IndexSet scalarlyAccessedIndices;

    for (EdgeDescriptor edgeDescriptor : llvm::make_range(
             edgesBegin(variableDescriptor), edgesEnd(variableDescriptor))) {
      const Edge &edge = getBaseGraph()[edgeDescriptor];

      for (const AccessFunction &accessFunction : edge.getAccessFunctions()) {
        if (accessFunction.isConstant()) {
          const Equation &equation = getEquationFromDescriptor(
              isEquation(edgeDescriptor.to) ? edgeDescriptor.to
                                            : edgeDescriptor.from);

          IndexSet mappedIndices = accessFunction.map(equation.getIndices());
          scalarlyAccessedIndices += mappedIndices;
        }
      }
    }

    return static_cast<double>(scalarlyAccessedIndices.flatSize()) /
           variable.getIndices().flatSize();
  }

  bool matchIteration(TraversableEdges &traversableEdges) {
    std::vector<AugmentingPath> augmentingPaths;
    getAugmentingPaths(augmentingPaths, traversableEdges);

    if (augmentingPaths.empty()) {
      return false;
    }

    for (auto &path : augmentingPaths) {
      applyPath(path, traversableEdges);
    }

    return true;
  }

  void getAugmentingPaths(std::vector<AugmentingPath> &augmentingPaths,
                          const TraversableEdges &traversableEdges) const {
    // Get the possible augmenting paths.
    LLVM_DEBUG({ llvm::dbgs() << "Searching augmenting paths\n"; });

    std::vector<std::shared_ptr<BFSStep>> paths =
        getCandidateAugmentingPaths(traversableEdges);

    LLVM_DEBUG({
      llvm::dbgs() << "Number of candidate augmenting paths: " << paths.size()
                   << "\n";
    });

    filterCandidateAugmentingPaths(augmentingPaths, paths);

    LLVM_DEBUG({
      llvm::dbgs() << "Number of accepted augmenting paths: "
                   << augmentingPaths.size() << "\n";
    });
  }

  void filterCandidateAugmentingPaths(
      std::vector<AugmentingPath> &augmentingPaths,
      llvm::ArrayRef<std::shared_ptr<BFSStep>> candidatePaths) const {
    // For each traversed node, keep track of the indices that have already
    // been traversed by some augmenting path. A new candidate path can be
    // accepted only if it does not traverse any of them.
    llvm::DenseMap<VertexDescriptor, IndexSet> visited;
    std::mutex mutex;

    mlir::parallelForEach(
        getContext(), candidatePaths,
        [&](const std::shared_ptr<BFSStep> &pathEnd) {
          LLVM_DEBUG({
            llvm::dbgs() << "Candidate augmenting path: ";
            pathEnd->dump(llvm::dbgs());
            llvm::dbgs() << "\n";
          });

          // All the candidate paths consist in at least two nodes by
          // construction
          assert(pathEnd->hasPrevious());

          std::list<Flow> flows;

          // The path's validity is unknown, so we must avoid polluting the
          // list of visited scalar nodes. If the path will be marked as valid,
          // then the new visits will be merged with the already found ones.
          llvm::DenseMap<VertexDescriptor, IndexSet> newVisits;

          const BFSStep *curStep = pathEnd.get();
          MCIM map = curStep->getMappedFlow();
          bool validPath = true;

          while (curStep && validPath) {
            if (curStep->hasPrevious()) {
              if (!flows.empty()) {
                // Restrict the flow
                const auto &prevMap = flows.front().delta;

                if (isVariable(curStep->getNode())) {
                  map = curStep->getMappedFlow().filterColumns(
                      prevMap.flattenRows());
                } else {
                  map = curStep->getMappedFlow().filterRows(
                      prevMap.flattenColumns());
                }
              }

              flows.emplace(flows.begin(), getBaseGraph(),
                            curStep->getPrevious()->getNode(),
                            curStep->getEdge(), map);
            }

            IndexSet touchedIndices = isVariable(curStep->getNode())
                                          ? map.flattenRows()
                                          : map.flattenColumns();

            // Early check for path validity.
            std::lock_guard lock(mutex);

            if (auto it = visited.find(curStep->getNode());
                it != visited.end()) {
              if (touchedIndices.overlaps(it->second)) {
                // Discard the current path, as it overlaps with another one.
                validPath = false;
              } else {
                newVisits[curStep->getNode()] += touchedIndices;
              }
            } else {
              newVisits[curStep->getNode()] = std::move(touchedIndices);
            }

            // Move backwards inside the candidate augmenting path.
            curStep = curStep->getPrevious();
          }

          LLVM_DEBUG({
            if (validPath) {
              llvm::dbgs() << "Accepted path\n";
            } else {
              llvm::dbgs() << "Discarded path\n";
            }
          });

          std::lock_guard lock(mutex);

          if (validPath) {
            // Check for path validity.
            for (auto &newVisit : newVisits) {
              if (auto it = visited.find(newVisit.first); it != visited.end()) {
                if (newVisit.second.overlaps(it->second)) {
                  // Discard the current path, as it overlaps with another one.
                  validPath = false;
                  break;
                }
              }
            }
          }

          if (validPath) {
            augmentingPaths.emplace_back(std::move(flows));

            for (auto &newVisit : newVisits) {
              visited[newVisit.first] += newVisit.second;
            }
          }
        });
  }

  std::vector<std::shared_ptr<BFSStep>>
  getCandidateAugmentingPaths(const TraversableEdges &traversableEdges) const {
    std::vector<std::shared_ptr<BFSStep>> paths;
    Frontier frontier;

    // Computation of the initial frontier.
    getInitialFrontier(frontier);

    // Breadth-first search.
    Frontier newFrontier;

    std::mutex newFrontierMutex;
    std::mutex pathsMutex;

    while (!frontier.empty() && paths.empty()) {
      sortFrontier(frontier);
      LLVM_DEBUG(frontier.dump(llvm::dbgs()));

      auto stepFn = [&](const std::shared_ptr<BFSStep> &step) {
        std::vector<std::shared_ptr<BFSStep>> localFrontier;
        std::vector<std::shared_ptr<BFSStep>> localPaths;

        expandFrontier(step, traversableEdges, localFrontier, localPaths);

        if (!localFrontier.empty()) {
          std::lock_guard<std::mutex> lock(newFrontierMutex);
          llvm::append_range(newFrontier, std::move(localFrontier));
        }

        if (!localPaths.empty()) {
          std::lock_guard<std::mutex> lock(pathsMutex);
          llvm::append_range(paths, std::move(localPaths));
        }
      };

      LLVM_DEBUG(llvm::dbgs() << "Expanding the frontier\n");
      mlir::parallelForEach(getContext(), frontier, stepFn);

      // Set the new frontier for the next iteration.
      frontier = std::move(newFrontier);
    }

    sortPaths(paths);
    return paths;
  }

  void getInitialFrontier(Frontier &frontier) const {
    std::mutex mutex;

    mlir::parallelForEach(
        getContext(), getEquationsBeginIt(), getEquationsEndIt(),
        [&](VertexDescriptor equationDescriptor) {
          const Equation &equation =
              getEquationFromDescriptor(equationDescriptor);

          if (const IndexSet &unmatchedEquations = equation.getUnmatched();
              !unmatchedEquations.empty()) {
            std::lock_guard lock(mutex);
            frontier.emplace_back(getBaseGraph(), equationDescriptor,
                                  std::move(unmatchedEquations));
          }
        });
  }

  static void sortFrontier(Frontier &frontier) {
    std::vector<std::shared_ptr<BFSStep>> paths;

    for (size_t i = 0, e = frontier.size(); i < e; ++i) {
      paths.push_back(frontier.at(i));
    }

    sortPaths(paths);
    frontier.clear();

    for (auto &path : paths) {
      frontier.push_back(std::move(path));
    }
  }

  static void sortPaths(std::vector<std::shared_ptr<BFSStep>> &paths) {
    std::vector<std::shared_ptr<BFSStep>> result;

    // Split the paths into partitions based on their size.
    llvm::DenseMap<size_t, llvm::SmallVector<size_t>> pathsBySize;
    llvm::DenseSet<size_t> pathSizes;

    for (size_t i = 0, e = paths.size(); i < e; ++i) {
      size_t size = paths[i]->getCandidates().flatSize();
      pathsBySize[size].push_back(i);
      pathSizes.insert(size);
    }

    llvm::SmallVector<size_t> orderedPathsSizes(pathSizes.begin(),
                                                pathSizes.end());

    // Order the partition sizes.
    llvm::sort(orderedPathsSizes, [](size_t a, size_t b) { return a > b; });

    // Merge the partitions.
    for (size_t pathSize : orderedPathsSizes) {
      for (size_t path : pathsBySize[pathSize]) {
        result.push_back(std::move(paths[path]));
      }
    }

    paths = std::move(result);
  }

  void expandFrontier(const std::shared_ptr<BFSStep> &step,
                      const TraversableEdges &traversableEdges,
                      std::vector<std::shared_ptr<BFSStep>> &newFrontier,
                      std::vector<std::shared_ptr<BFSStep>> &paths) const {
    LLVM_DEBUG({
      llvm::dbgs() << "Current traversal: ";
      step->dump(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    const VertexDescriptor &vertexDescriptor = step->getNode();

    auto containsFn = [&](VertexDescriptor node, const IndexSet &indices) {
      const BFSStep *currentStep = step.get();

      while (currentStep) {
        if (currentStep->getNode() == node &&
            currentStep->getCandidates() == indices) {
          return true;
        }

        if (currentStep->hasPrevious()) {
          currentStep = currentStep->getPrevious();
        } else {
          currentStep = nullptr;
        }
      }

      return false;
    };

    for (EdgeDescriptor edgeDescriptor :
         traversableEdges.lookup(vertexDescriptor)) {
      VertexDescriptor nextNode = edgeDescriptor.from == vertexDescriptor
                                      ? edgeDescriptor.to
                                      : edgeDescriptor.from;

      const Edge &edge = getBaseGraph()[edgeDescriptor];

      if (isEquation(vertexDescriptor)) {
        assert(isVariable(nextNode));
        const Variable &var = getVariableFromDescriptor(nextNode);

        LLVM_DEBUG({
          llvm::dbgs() << "Exploring edge from "
                       << getEquationFromDescriptor(vertexDescriptor).getId()
                       << " to " << var.getId() << "\n";
        });

        const MCIM &unmatchedMatrix = edge.getUnmatched();
        MCIM filteredMatrix = unmatchedMatrix.filterRows(step->getCandidates());

        internal::LocalMatchingSolutions solutions =
            internal::solveLocalMatchingProblem(filteredMatrix);

        LLVM_DEBUG({
          llvm::dbgs() << "Number of local matching solutions: "
                       << solutions.size() << "\n";
        });

        for (auto solution : solutions) {
          const IndexSet &unmatchedScalarVariables = var.getUnmatched();
          auto matched = solution.filterColumns(unmatchedScalarVariables);
          IndexSet indices = solution.flattenRows();

          if (!containsFn(nextNode, indices)) {
            if (!matched.empty()) {
              paths.push_back(std::make_shared<BFSStep>(
                  getBaseGraph(), step, edgeDescriptor, nextNode, indices,
                  matched));
            } else {
              newFrontier.push_back(std::make_shared<BFSStep>(
                  getBaseGraph(), step, edgeDescriptor, nextNode, indices,
                  solution));
            }
          }
        }
      } else {
        assert(isEquation(nextNode));

        LLVM_DEBUG({
          llvm::dbgs() << "Exploring edge from "
                       << getVariableFromDescriptor(vertexDescriptor).getId()
                       << " to " << getEquationFromDescriptor(nextNode).getId()
                       << "\n";
        });

        auto filteredMatrix =
            edge.getMatched().filterColumns(step->getCandidates());

        internal::LocalMatchingSolutions solutions =
            internal::solveLocalMatchingProblem(filteredMatrix);

        LLVM_DEBUG({
          llvm::dbgs() << "Number of local matching solutions: "
                       << solutions.size() << "\n";
        });

        for (auto solution : solutions) {
          IndexSet indices = solution.flattenColumns();

          if (!containsFn(nextNode, indices)) {
            newFrontier.push_back(std::make_shared<BFSStep>(
                getBaseGraph(), step, edgeDescriptor, nextNode,
                std::move(indices), solution));
          }
        }
      }
    }
  }

  /// Apply an augmenting path to the graph.
  void applyPath(const AugmentingPath &path,
                 TraversableEdges &traversableEdges) {
    // In order to preserve consistency of the match information among
    // edges and nodes, we need to separately track the modifications
    // created by the augmenting path on the vertices and apply all the
    // removals before the additions.
    // Consider in fact the example path [eq1 -> x -> eq2]: the first
    // move would add some match information to eq1 and x, while the
    // subsequent x -> eq2 would remove some from x. However, being the
    // match matrices made of booleans, the components of x that are
    // matched by eq1 would result as unmatched. If we instead first
    // apply the removals, the new matches are not wrongly erased anymore.

    llvm::DenseMap<VertexDescriptor, IndexSet> removedMatches;
    llvm::DenseMap<VertexDescriptor, IndexSet> newMatches;

    // Update the match matrices on the edges and store the information
    // about the vertices to be updated later.

    for (auto &flow : path) {
      Edge &edge = getBaseGraph()[flow.edge];

      VertexDescriptor from = flow.source;

      VertexDescriptor to =
          flow.edge.from == from ? flow.edge.to : flow.edge.from;

      auto deltaEquations = flow.delta.flattenColumns();
      auto deltaVariables = flow.delta.flattenRows();

      if (isVariable(from)) {
        // Backward arc (from variable to equation).
        removedMatches[from] += deltaVariables;
        removedMatches[to] += deltaEquations;
        edge.removeMatch(flow.delta);

        traversableEdges[to].insert(flow.edge);

        if (edge.getMatched().empty()) {
          traversableEdges[from].remove(flow.edge);
        } else {
          traversableEdges[from].insert(flow.edge);
        }
      } else {
        // Forward arc (from equation to variable).
        newMatches[from] += deltaEquations;
        newMatches[to] += deltaVariables;
        edge.addMatch(flow.delta);

        traversableEdges[to].insert(flow.edge);

        if (edge.getUnmatched().empty()) {
          traversableEdges[from].remove(flow.edge);
        } else {
          traversableEdges[from].insert(flow.edge);
        }
      }
    }

    // Update the match information stored in the vertices.
    mlir::parallelForEach(getContext(), removedMatches, [&](const auto &match) {
      std::visit([&match](auto &node) { node.removeMatch(match.second); },
                 getBaseGraph()[match.first]);
    });

    mlir::parallelForEach(getContext(), newMatches, [&](const auto &match) {
      std::visit([&match](auto &node) { node.addMatch(match.second); },
                 getBaseGraph()[match.first]);
    });
  }
};
} // namespace internal::matching

template <typename VariableProperty, typename EquationProperty>
class MatchingGraph : public internal::matching::MatchingGraphCRTP<
                          MatchingGraph<VariableProperty, EquationProperty>,
                          VariableProperty, EquationProperty> {
public:
  using internal::matching::MatchingGraphCRTP<
      MatchingGraph, VariableProperty, EquationProperty>::MatchingGraphCRTP;

  MatchingGraph newInstance() const override {
    return MatchingGraph{this->getContext()};
  }
};
} // namespace marco::modeling

namespace llvm {
template <typename VertexProperty>
struct DenseMapInfo<
    ::marco::modeling::internal::matching::VariableId<VertexProperty>> {
  using Key = ::marco::modeling::internal::matching::VariableId<VertexProperty>;

  static Key getEmptyKey() {
    return {llvm::DenseMapInfo<typename Key::BaseId>::getEmptyKey(),
            std::nullopt};
  }

  static Key getTombstoneKey() {
    return {llvm::DenseMapInfo<typename Key::BaseId>::getTombstoneKey(),
            std::nullopt};
  }

  static unsigned getHashValue(const Key &val) { return hash_value(val); }

  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
};

template <typename EquationProperty>
struct DenseMapInfo<
    ::marco::modeling::internal::matching::EquationId<EquationProperty>> {
  using Key =
      ::marco::modeling::internal::matching::EquationId<EquationProperty>;

  static Key getEmptyKey() {
    return {llvm::DenseMapInfo<typename Key::BaseId>::getEmptyKey(),
            std::nullopt};
  }

  static Key getTombstoneKey() {
    return {llvm::DenseMapInfo<typename Key::BaseId>::getTombstoneKey(),
            std::nullopt};
  }

  static unsigned getHashValue(const Key &val) { return hash_value(val); }

  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
};
} // namespace llvm

#endif // MARCO_MODELING_MATCHING_H
