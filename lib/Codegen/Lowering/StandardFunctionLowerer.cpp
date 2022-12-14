#include "marco/Codegen/Lowering/StandardFunctionLowerer.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include <set>
#include <stack>

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  /// Directed graph representing the dependencies among the variables with
  /// respect to the usage of variables for the computation of the dynamic
  /// dimensions of their types.
  class DynamicDimensionsGraph
  {
    private:
      class ExpressionVisitor;

    public:
      /// A node of the graph.
      struct Node
      {
        Node();

        Node(const DynamicDimensionsGraph* graph, const Member* member);

        bool operator==(const Node& other) const;

        bool operator!=(const Node& other) const;

        bool operator<(const Node& other) const;

        const DynamicDimensionsGraph* graph;
        const Member* member;
      };

      DynamicDimensionsGraph();

      void addMembersGroup(llvm::ArrayRef<const Member*> group);

      void discoverDependencies();

      /// Get the number of nodes.
      /// Note that nodes consists in the inserted members together with the
      /// entry node.
      size_t getNumOfNodes() const;

      Node getEntryNode() const;

      /// @name Iterators for the children of a node
      /// {

      std::set<Node>::const_iterator childrenBegin(Node node) const;

      std::set<Node>::const_iterator childrenEnd(Node node) const;

      /// }
      /// @name Iterators for the nodes
      /// {

      llvm::SmallVector<Node>::const_iterator nodesBegin() const;

      llvm::SmallVector<Node>::const_iterator nodesEnd() const;

      /// }

      /// Check if the graph contains cycles.
      bool hasCycles() const;

      /// Perform a post-order visit of the graph and get the ordered members.
      llvm::SmallVector<const Member*> postOrder() const;

    private:
      std::set<const Member*> getUsedMembers(const Expression* expression);

    private:
      std::map<const Member*, std::set<Node>> arcs;
      llvm::SmallVector<Node> nodes;
      llvm::DenseMap<llvm::StringRef, Node> symbolTable;
  };
}

namespace llvm
{
  template<>
  struct DenseMapInfo<DynamicDimensionsGraph::Node>
  {
    static inline DynamicDimensionsGraph::Node getEmptyKey()
    {
      return DynamicDimensionsGraph::Node(nullptr, nullptr);
    }

    static inline DynamicDimensionsGraph::Node getTombstoneKey()
    {
      return DynamicDimensionsGraph::Node(nullptr, nullptr);
    }

    static unsigned getHashValue(const DynamicDimensionsGraph::Node& val)
    {
      return std::hash<const Member*>{}(val.member);
    }

    static bool isEqual(
        const DynamicDimensionsGraph::Node& lhs,
        const DynamicDimensionsGraph::Node& rhs)
    {
      return lhs.graph == rhs.graph && lhs.member == rhs.member;
    }
  };

  template<>
  struct GraphTraits<const DynamicDimensionsGraph*>
  {
    using GraphType = const DynamicDimensionsGraph*;
    using NodeRef = DynamicDimensionsGraph::Node;

    using ChildIteratorType =
        std::set<DynamicDimensionsGraph::Node>::const_iterator;

    static NodeRef getEntryNode(const GraphType& graph)
    {
      return graph->getEntryNode();
    }

    static ChildIteratorType child_begin(NodeRef node)
    {
      return node.graph->childrenBegin(node);
    }

    static ChildIteratorType child_end(NodeRef node)
    {
      return node.graph->childrenEnd(node);
    }

    using nodes_iterator = llvm::SmallVector<NodeRef>::const_iterator;

    static nodes_iterator nodes_begin(GraphType* graph)
    {
      return (*graph)->nodesBegin();
    }

    static nodes_iterator nodes_end(GraphType* graph)
    {
      return (*graph)->nodesEnd();
    }

    // There is no need for a dedicated class for the arcs.
    using EdgeRef = DynamicDimensionsGraph::Node;

    using ChildEdgeIteratorType =
        std::set<DynamicDimensionsGraph::Node>::const_iterator;

    static ChildEdgeIteratorType child_edge_begin(NodeRef node)
    {
      return node.graph->childrenBegin(node);
    }

    static ChildEdgeIteratorType child_edge_end(NodeRef node)
    {
      return node.graph->childrenEnd(node);
    }

    static NodeRef edge_dest(EdgeRef edge)
    {
      return edge;
    }

    static unsigned int size(GraphType* graph)
    {
      return (*graph)->getNumOfNodes();
    }
  };
}

namespace
{
  class DynamicDimensionsGraph::ExpressionVisitor
  {
    public:
    std::set<const Expression*> operator()(const Array& array)
    {
      std::set<const Expression*> result;

      for (const auto& value : array) {
        result.insert(value.get());
      }

      return result;
    }

    std::set<const Expression*> operator()(const Call& call)
    {
      std::set<const Expression*> result;

      for (const auto& expression : call.getArgs()) {
        result.insert(expression.get());
      }

      return result;
    }

    std::set<const Expression*> operator()(const Constant& constant)
    {
      return {};
    }

    std::set<const Expression*> operator()(const Operation& operation)
    {
      std::set<const Expression*> result;

      for (const auto& expression : operation.getArguments()) {
        result.insert(expression.get());
      }

      return result;
    }

    std::set<const Expression*> operator()(const ReferenceAccess& reference)
    {
      return {};
    }

    std::set<const Expression*> operator()(const Tuple& tuple)
    {
      std::set<const Expression*> result;

      for (const auto& expression : tuple) {
        result.insert(expression.get());
      }

      return result;
    }

    std::set<const Expression*> operator()(const RecordInstance& recordInstance)
    {
      std::set<const Expression*> result;

      for (const auto& expression : recordInstance) {
        result.insert(expression.get());
      }

      return result;
    }
  };

  DynamicDimensionsGraph::Node::Node()
      : graph(nullptr), member(nullptr)
  {
  }

  DynamicDimensionsGraph::Node::Node(
      const DynamicDimensionsGraph* graph, const Member* member)
      : graph(graph), member(member)
  {
  }

  bool DynamicDimensionsGraph::Node::operator==(
      const DynamicDimensionsGraph::Node& other) const
  {
    return graph == other.graph && member == other.member;
  }

  bool DynamicDimensionsGraph::Node::operator!=(
      const DynamicDimensionsGraph::Node& other) const
  {
    return graph != other.graph || member != other.member;
  }

  bool DynamicDimensionsGraph::Node::operator<(
      const DynamicDimensionsGraph::Node& other) const
  {
    if (member == nullptr) {
      return true;
    }

    if (other.member == nullptr) {
      return false;
    }

    return member < other.member;
  }

  DynamicDimensionsGraph::DynamicDimensionsGraph()
  {
    // Entry node, which is connected to every other node.
    nodes.emplace_back(this, nullptr);
  }

  void DynamicDimensionsGraph::addMembersGroup(
      llvm::ArrayRef<const Member*> group)
  {
    for (const Member* member : group) {
      assert(member != nullptr);

      Node node(this, member);
      nodes.push_back(node);
      symbolTable[member->getName()] = node;

      // Connect the entry node.
      arcs[getEntryNode().member].insert(node);
    }

    for (size_t i = 1, e = group.size(); i < e; ++i) {
      arcs[group[i]].insert(Node(this, group[i - 1]));
    }
  }

  void DynamicDimensionsGraph::discoverDependencies()
  {
    for (const Node& node : nodes) {
      if (node != getEntryNode()) {
        const Member* member = node.member;

        auto& children = arcs[member];

        for (const auto& dimension : member->getType().getDimensions()) {
          if (dimension.hasExpression()) {
            const Expression* expression = dimension.getExpression();
            std::set<const Member*> usedMembers = getUsedMembers(expression);

            for (const Member* dependency : usedMembers) {
              children.insert(Node(this, dependency));
            }
          }
        }
      }
    }
  }

  size_t DynamicDimensionsGraph::getNumOfNodes() const
  {
    return nodes.size();
  }

  DynamicDimensionsGraph::Node DynamicDimensionsGraph::getEntryNode() const
  {
    return nodes[0];
  }

  std::set<DynamicDimensionsGraph::Node>::const_iterator
  DynamicDimensionsGraph::childrenBegin(Node node) const
  {
    auto it = arcs.find(node.member);
    assert(it != arcs.end());
    const auto& children = it->second;
    return children.begin();
  }

  std::set<DynamicDimensionsGraph::Node>::const_iterator
  DynamicDimensionsGraph::childrenEnd(Node node) const
  {
    auto it = arcs.find(node.member);
    assert(it != arcs.end());
    const auto& children = it->second;
    return children.end();
  }

  llvm::SmallVector<DynamicDimensionsGraph::Node>::const_iterator
  DynamicDimensionsGraph::nodesBegin() const
  {
    return nodes.begin();
  }

  llvm::SmallVector<DynamicDimensionsGraph::Node>::const_iterator
  DynamicDimensionsGraph::nodesEnd() const
  {
    return nodes.end();
  }

  bool DynamicDimensionsGraph::hasCycles() const
  {
    auto beginIt = llvm::scc_begin(this);
    auto endIt =  llvm::scc_end(this);

    for (auto it = beginIt; it != endIt; ++it) {
      if (it.hasCycle()) {
        return true;
      }
    }

    return false;
  }

  llvm::SmallVector<const Member*> DynamicDimensionsGraph::postOrder() const
  {
    llvm::SmallVector<const Member*> result;
    std::set<Node> set;

    for (const auto& node : llvm::post_order_ext(this, set)) {
      if (node != getEntryNode()) {
        result.push_back(node.member);
      }
    }

    return result;
  }

  std::set<const Member*> DynamicDimensionsGraph::getUsedMembers(
      const Expression* expression)
  {
    std::set<const Member*> result;

    std::stack<const Expression*> expressions;
    expressions.push(expression);

    while (!expressions.empty()) {
      const Expression* current = expressions.top();
      expressions.pop();

      if (auto* reference = current->dyn_get<ReferenceAccess>()) {
        const Member* member =
            symbolTable.find(reference->getName())->second.member;

        assert(member != nullptr);
        result.insert(member);
      } else {
        ExpressionVisitor visitor;

        for (const Expression* child : current->visit(visitor)) {
          expressions.push(child);
        }
      }
    }

    return result;
  }
}

namespace marco::codegen::lowering
{
  StandardFunctionLowerer::StandardFunctionLowerer(
      LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge)
  {
  }

  std::vector<mlir::Operation*> StandardFunctionLowerer::lower(
      const StandardFunction& function)
  {
    std::vector<mlir::Operation*> result;

    mlir::OpBuilder::InsertionGuard guard(builder());
    Lowerer::SymbolScope scope(symbolTable());

    auto location = loc(function.getLocation());

    // Input variables.
    llvm::SmallVector<llvm::StringRef, 3> argNames;
    llvm::SmallVector<mlir::Type, 3> argTypes;

    for (const auto& member : function.getArgs()) {
      argNames.emplace_back(member->getName());

      mlir::Type type = lower(member->getType());
      argTypes.emplace_back(type);
    }

    // Output variables.
    llvm::SmallVector<llvm::StringRef, 1> returnNames;
    llvm::SmallVector<mlir::Type, 1> returnTypes;
    auto outputMembers = function.getResults();

    for (const auto& member : outputMembers) {
      mlir::Type type = lower(member->getType());
      returnNames.emplace_back(member->getName());
      returnTypes.emplace_back(type);
    }

    // Create the function.
    auto functionType = builder().getFunctionType(argTypes, returnTypes);

    auto functionOp = builder().create<FunctionOp>(
        location, function.getName(), functionType);

    // Process the annotations.
    if (function.hasAnnotation()) {
      const auto* annotation = function.getAnnotation();

      // Inline attribute.
      functionOp->setAttr(
          "inline",
          builder().getBoolAttr(
              function.getAnnotation()->getInlineProperty()));

      // Inverse functions attribute.
      auto inverseFunctionAnnotation =
          annotation->getInverseFunctionAnnotation();

      InverseFunctionsMap map;

      // Create a map of the function members indexes for faster retrieval.
      llvm::StringMap<unsigned int> indexes;

      for (const auto& name : llvm::enumerate(argNames)) {
        indexes[name.value()] = name.index();
      }

      for (const auto& name : llvm::enumerate(returnNames)) {
        indexes[name.value()] = argNames.size() + name.index();
      }

      mlir::StorageUniquer::StorageAllocator allocator;

      // Iterate over the input arguments and for each invertible one
      // add the function to the inverse map.
      for (const auto& arg : argNames) {
        if (!inverseFunctionAnnotation.isInvertible(arg)) {
          continue;
        }

        auto inverseArgs = inverseFunctionAnnotation.getInverseArgs(arg);
        llvm::SmallVector<unsigned int, 3> permutation;

        for (const auto& inverseArg : inverseArgs) {
          assert(indexes.find(inverseArg) != indexes.end());
          permutation.push_back(indexes[inverseArg]);
        }

        map[indexes[arg]] = std::make_pair(
            inverseFunctionAnnotation.getInverseFunction(arg),
            allocator.copyInto(llvm::ArrayRef<unsigned int>(permutation)));
      }

      if (!map.empty()) {
        auto inverseFunctionAttribute =
            InverseFunctionsAttr::get(builder().getContext(), map);

        functionOp->setAttr("inverse", inverseFunctionAttribute);
      }

      if (annotation->hasDerivativeAnnotation()) {
        auto derivativeAnnotation = annotation->getDerivativeAnnotation();

        auto derivativeAttribute = DerivativeAttr::get(
            builder().getContext(),
            derivativeAnnotation.getName(),
            derivativeAnnotation.getOrder());

        functionOp->setAttr("derivative", derivativeAttribute);
      }
    }

    // Create the body of the function.
    mlir::Block* entryBlock = builder().createBlock(&functionOp.getBody());
    builder().setInsertionPointToStart(entryBlock);

    // Create the variables. The order in which the variables are created has
    // to take into account the dependencies of their dynamic dimensions. At
    // the same time, however, the relative order of the input and output
    // variables must be the same as the declared one, in order to preserve
    // the correctness of the calls. This last aspect is already taken into
    // account by the dependency graph.

    DynamicDimensionsGraph dynamicDimensionsGraph;

    dynamicDimensionsGraph.addMembersGroup(function.getArgs());
    dynamicDimensionsGraph.addMembersGroup(function.getResults());
    dynamicDimensionsGraph.addMembersGroup(function.getProtectedMembers());
    dynamicDimensionsGraph.discoverDependencies();

    assert(!dynamicDimensionsGraph.hasCycles());

    for (const auto& member : dynamicDimensionsGraph.postOrder()) {
      lower(*member);
    }

    // Initialize the variables which have a default value.
    // This must be performed after all the variables have been created, so
    // that we can be sure that references to other variables can be resolved
    // (without performing a post-order visit).

    for (const auto& member : function.getMembers()) {
      if (member->hasExpression()) {
        // If the member has an initializer expression, lower and assign it as
        // if it was a regular assignment statement.

        mlir::Value value = *lower(*member->getExpression())[0];
        symbolTable().lookup(member->getName()).set(value);
      }
    }

    // Emit the body of the function.
    if (auto algorithms = function.getAlgorithms(); algorithms.size() > 0) {
      assert(algorithms.size() == 1);

      const auto& algorithm = function.getAlgorithms()[0];

      for (const auto& statement : *algorithm) {
        lower(*statement);
      }
    }

    result.push_back(functionOp);
    return result;
  }

  void StandardFunctionLowerer::lower(const Member& member)
  {
    auto location = loc(member.getLocation());
    mlir::Type type = lower(member.getType());

    llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      auto expressionsCount = llvm::count_if(
          member.getType().getDimensions(),
          [](const auto& dimension) {
            return dimension.hasExpression();
          });

      // If all the dynamic dimensions have an expression to determine their
      // values, then the member can be instantiated from the beginning.

      bool initialized = expressionsCount == arrayType.getNumDynamicDims();

      if (initialized) {
        for (const auto& dimension : member.getType().getDimensions()) {
          if (dimension.hasExpression()) {
            mlir::Value size = *lower(*dimension.getExpression())[0];
            size = builder().create<CastOp>(
                location, builder().getIndexType(), size);

            dynamicDimensions.push_back(size);
          }
        }
      }
    }

    bool isConstant = false;
    IOProperty ioProperty = IOProperty::none;

    if (member.isInput()) {
      ioProperty = IOProperty::input;
    } else if (member.isOutput()) {
      ioProperty = IOProperty::output;
    }

    auto memberType = MemberType::wrap(type, isConstant, ioProperty);

    mlir::Value var = builder().create<MemberCreateOp>(
        location, member.getName(), memberType, dynamicDimensions);

    // Add the member to the symbol table.
    symbolTable().insert(member.getName(), Reference::member(&builder(), var));
  }
}
