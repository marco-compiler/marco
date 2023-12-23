#include "marco/Codegen/Transforms/FunctionInlining.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include <set>

namespace mlir::modelica
{
#define GEN_PASS_DEF_FUNCTIONINLININGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class CallGraph
  {
    public:
      /// A node of the graph.
      struct Node
      {
        Node() : graph(nullptr), op(nullptr)
        {
        }

        Node(const CallGraph* graph, mlir::Operation* op)
            : graph(graph), op(op)
        {
        }

        bool operator==(const Node& other) const
        {
          return graph == other.graph && op == other.op;
        }

        bool operator!=(const Node& other) const
        {
          return graph != other.graph || op != other.op;
        }

        bool operator<(const Node& other) const
        {
          if (op == nullptr) {
            return true;
          }

          if (other.op == nullptr) {
            return false;
          }

          return op < other.op;
        }

        const CallGraph* graph;
        mlir::Operation* op;
      };

      CallGraph()
      {
        // Entry node, which is connected to every other node.
        nodes.emplace_back(this, nullptr);

        // Ensure that the set of children for the entry node exists, even in
        // case of no other nodes.
        arcs[getEntryNode().op] = {};
      }

      ~CallGraph() = default;

      void addNode(FunctionOp functionOp)
      {
        assert(functionOp != nullptr);
        Node node(this, functionOp.getOperation());
        nodes.push_back(node);
        nodesByOp[functionOp.getOperation()] = node;

        // Ensure that the set of children for the node exists.
        arcs[node.op] = {};

        // Connect the entry node.
        arcs[getEntryNode().op].insert(node);
      }

      bool hasNode(FunctionOp functionOp) const
      {
        assert(functionOp != nullptr);
        return nodesByOp.find(functionOp.getOperation()) != nodesByOp.end();
      }

      void addEdge(FunctionOp caller, FunctionOp callee)
      {
        assert(caller != nullptr);
        assert(callee != nullptr);
        assert(arcs.find(caller.getOperation()) != arcs.end());
        assert(nodesByOp.find(callee.getOperation()) != nodesByOp.end());
        arcs[caller.getOperation()].insert(nodesByOp[callee.getOperation()]);
      }

      /// Get the number of nodes.
      /// Note that nodes consists in the inserted operations together with the
      /// entry node.
      size_t getNumOfNodes() const
      {
        return nodes.size();
      }

      /// Get the entry node of the graph.
      const Node& getEntryNode() const
      {
        return nodes[0];
      }

      /// @name Iterators for the children of a node
      /// {

      std::set<Node>::const_iterator childrenBegin(Node node) const
      {
        auto it = arcs.find(node.op);
        assert(it != arcs.end());
        const auto& children = it->second;
        return children.begin();
      }

      std::set<Node>::const_iterator childrenEnd(Node node) const
      {
        auto it = arcs.find(node.op);
        assert(it != arcs.end());
        const auto& children = it->second;
        return children.end();
      }

      /// }
      /// @name Iterators for the nodes
      /// {

      llvm::SmallVector<Node>::const_iterator nodesBegin() const
      {
        return nodes.begin();
      }

      llvm::SmallVector<Node>::const_iterator nodesEnd() const
      {
        return nodes.end();
      }

      /// }

      llvm::DenseSet<FunctionOp> getInlinableFunctions() const;

    private:
      llvm::DenseMap<mlir::Operation*, std::set<Node>> arcs;
      llvm::SmallVector<Node> nodes;
      llvm::DenseMap<mlir::Operation*, Node> nodesByOp;
  };
}

namespace llvm
{
  template<>
  struct DenseMapInfo<::CallGraph::Node>
  {
    static inline ::CallGraph::Node getEmptyKey()
    {
      return {nullptr, nullptr};
    }

    static inline ::CallGraph::Node getTombstoneKey()
    {
      return {nullptr, nullptr};
    }

    static unsigned getHashValue(const ::CallGraph::Node& val)
    {
      return std::hash<mlir::Operation*>{}(val.op);
    }

    static bool isEqual(
        const ::CallGraph::Node& lhs,
        const ::CallGraph::Node& rhs)
    {
      return lhs.graph == rhs.graph && lhs.op == rhs.op;
    }
  };

  template<>
  struct GraphTraits<const ::CallGraph*>
  {
      using GraphType = const ::CallGraph*;
      using NodeRef = ::CallGraph::Node;

      using ChildIteratorType =
          std::set<::CallGraph::Node>::const_iterator;

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
      using EdgeRef = ::CallGraph::Node;

      using ChildEdgeIteratorType =
          std::set<::CallGraph::Node>::const_iterator;

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

llvm::DenseSet<FunctionOp> CallGraph::getInlinableFunctions() const
{
  llvm::DenseSet<FunctionOp> result;

  auto beginIt = llvm::scc_begin(this);
  auto endIt =  llvm::scc_end(this);

  for (auto it = beginIt; it != endIt; ++it) {
    if (it.hasCycle()) {
      continue;
    }

    for (const Node& node : *it) {
      if (node != getEntryNode()) {
        result.insert(mlir::cast<FunctionOp>(node.op));
      }
    }
  }

  return result;
}

static bool canBeInlined(FunctionOp functionOp)
{
  // The function must be explicitly marked as inlinable.
  if (!functionOp.shouldBeInlined()) {
    return false;
  }

  // Check that there is exactly one algorithm section.
  if (auto algorithmOps = functionOp.getOps<AlgorithmOp>();
      std::distance(algorithmOps.begin(), algorithmOps.end()) != 1) {
    return false;
  }

  // Check that operations inside the algorithm section have no regions with
  // side effects.
  llvm::SmallVector<mlir::Operation*> nestedOps;

  for (AlgorithmOp algorithmOp : functionOp.getOps<AlgorithmOp>()) {
    for (auto& nestedOp : algorithmOp.getOps()) {
      for (auto& nestedRegion : nestedOp.getRegions()) {
        for (auto& nestedRegionOp : nestedRegion.getOps()) {
          nestedOps.push_back(&nestedRegionOp);
        }
      }
    }
  }

  while (!nestedOps.empty()) {
    mlir::Operation* nestedOp = nestedOps.pop_back_val();

    if (mlir::isa<VariableSetOp, ComponentSetOp>(nestedOp)) {
      return false;
    }

    for (auto& nestedRegion : nestedOp->getRegions()) {
      for (auto& nestedRegionOp : nestedRegion.getOps()) {
        nestedOps.push_back(&nestedRegionOp);
      }
    }
  }

  return true;
}

namespace
{
  class VariablesDependencyGraph
  {
    public:
      /// A node of the graph.
      struct Node
      {
        Node() : graph(nullptr), variable(nullptr)
        {
        }

        Node(const VariablesDependencyGraph* graph, mlir::Operation* variable)
            : graph(graph), variable(variable)
        {
        }

        bool operator==(const Node& other) const
        {
          return graph == other.graph && variable == other.variable;
        }

        bool operator!=(const Node& other) const
        {
          return graph != other.graph || variable != other.variable;
        }

        bool operator<(const Node& other) const
        {
          if (variable == nullptr) {
            return true;
          }

          if (other.variable == nullptr) {
            return false;
          }

          return variable < other.variable;
        }

        const VariablesDependencyGraph* graph;
        mlir::Operation* variable;
      };

      VariablesDependencyGraph()
      {
        // Entry node, which is connected to every other node.
        nodes.emplace_back(this, nullptr);

        // Ensure that the set of children for the entry node exists, even in
        // case of no other nodes.
        arcs[getEntryNode().variable] = {};
      }

      virtual ~VariablesDependencyGraph() = default;

      /// Add a group of variables to the graph and optionally enforce their
      /// relative order to be preserved.
      void addVariables(llvm::ArrayRef<VariableOp> variables)
      {
        for (VariableOp variable : variables) {
          assert(variable != nullptr);

          Node node(this, variable.getOperation());
          nodes.push_back(node);
          nodesByName[variable.getSymName()] = node;
          arcs[node.variable] = {};
        }
      }

      /// Discover the dependencies of the variables that have been added to
      /// the graph.
      void discoverDependencies()
      {
        for (const Node& node : nodes) {
          if (node == getEntryNode()) {
            continue;
          }

          // Connect the entry node to the current one.
          arcs[getEntryNode().variable].insert(node);

          // Connect the current node to the other nodes to which it depends.
          mlir::Operation* variable = node.variable;
          auto variableOp = mlir::cast<VariableOp>(variable);

          for (llvm::StringRef dependency : getDependencies(variableOp)) {
            auto& children = arcs[nodesByName[dependency].variable];
            children.insert(node);
          }
        }
      }

      /// Get the number of nodes.
      /// Note that nodes consists in the inserted variables together with the
      /// entry node.
      size_t getNumOfNodes() const
      {
        return nodes.size();
      }

      /// Get the entry node of the graph.
      const Node& getEntryNode() const
      {
        return nodes[0];
      }

      /// @name Iterators for the children of a node
      /// {

      std::set<Node>::const_iterator childrenBegin(Node node) const
      {
        auto it = arcs.find(node.variable);
        assert(it != arcs.end());
        const auto& children = it->second;
        return children.begin();
      }

      std::set<Node>::const_iterator childrenEnd(Node node) const
      {
        auto it = arcs.find(node.variable);
        assert(it != arcs.end());
        const auto& children = it->second;
        return children.end();
      }

      /// }
      /// @name Iterators for the nodes
      /// {

      llvm::SmallVector<Node>::const_iterator nodesBegin() const
      {
        return nodes.begin();
      }

      llvm::SmallVector<Node>::const_iterator nodesEnd() const
      {
        return nodes.end();
      }

      /// }

      /// Check if the graph contains cycles.
      bool hasCycles() const;

      /// Perform a post-order visit of the graph and get the ordered
      /// variables.
      llvm::SmallVector<VariableOp> postOrder() const;

      /// Perform a reverse post-order visit of the graph and get the ordered
      /// variables.
      llvm::SmallVector<VariableOp> reversePostOrder() const;

    protected:
      virtual std::set<llvm::StringRef> getDependencies(
          VariableOp variable) = 0;

    private:
      llvm::DenseMap<mlir::Operation*, std::set<Node>> arcs;
      llvm::SmallVector<Node> nodes;
      llvm::DenseMap<llvm::StringRef, Node> nodesByName;
  };

  /// Directed graph representing the dependencies among the variables with
  /// respect to the usage of variables for the computation of the default
  /// value.
  class DefaultValuesGraph : public VariablesDependencyGraph
  {
    public:
      explicit DefaultValuesGraph(const llvm::StringMap<DefaultOp>& defaultOps)
          : defaultOps(&defaultOps)
      {
      }

    protected:
      std::set<llvm::StringRef> getDependencies(VariableOp variable) override;

    private:
      const llvm::StringMap<DefaultOp>* defaultOps;
  };
}

namespace llvm
{
  template<>
  struct DenseMapInfo<::VariablesDependencyGraph::Node>
  {
    static inline ::VariablesDependencyGraph::Node getEmptyKey()
    {
      return {nullptr, nullptr};
    }

    static inline ::VariablesDependencyGraph::Node getTombstoneKey()
    {
      return {nullptr, nullptr};
    }

    static unsigned getHashValue(const ::VariablesDependencyGraph::Node& val)
    {
      return std::hash<mlir::Operation*>{}(val.variable);
    }

    static bool isEqual(
        const ::VariablesDependencyGraph::Node& lhs,
        const ::VariablesDependencyGraph::Node& rhs)
    {
      return lhs.graph == rhs.graph && lhs.variable == rhs.variable;
    }
  };

  template<>
  struct GraphTraits<const ::VariablesDependencyGraph*>
  {
    using GraphType = const ::VariablesDependencyGraph*;
    using NodeRef = ::VariablesDependencyGraph::Node;

    using ChildIteratorType =
        std::set<::VariablesDependencyGraph::Node>::const_iterator;

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
    using EdgeRef = ::VariablesDependencyGraph::Node;

    using ChildEdgeIteratorType =
        std::set<::VariablesDependencyGraph::Node>::const_iterator;

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
  bool VariablesDependencyGraph::hasCycles() const
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

  llvm::SmallVector<VariableOp> VariablesDependencyGraph::postOrder() const
  {
    assert(!hasCycles());

    llvm::SmallVector<VariableOp> result;
    std::set<Node> set;

    for (const auto& node : llvm::post_order_ext(this, set)) {
      if (node != getEntryNode()) {
        result.push_back(mlir::cast<VariableOp>(node.variable));
      }
    }

    return result;
  }

  llvm::SmallVector<VariableOp> VariablesDependencyGraph::reversePostOrder() const
  {
    auto result = postOrder();
    std::reverse(result.begin(), result.end());
    return result;
  }

  std::set<llvm::StringRef> DefaultValuesGraph::getDependencies(
      VariableOp variable)
  {
    std::set<llvm::StringRef> dependencies;
    auto defaultOpIt = defaultOps->find(variable.getSymName());

    if (defaultOpIt != defaultOps->end()) {
      DefaultOp defaultOp = defaultOpIt->getValue();

      defaultOp->walk([&](VariableGetOp getOp) {
        dependencies.insert(getOp.getVariable());
      });
    }

    return dependencies;
  }
}

namespace
{
  class DefaultOpComputationOrderings
  {
    public:
      llvm::ArrayRef<VariableOp> get(FunctionOp functionOp) const
      {
        auto it = orderings.find(functionOp);

        // If the assertion doesn't hold, then verification is wrong.
        assert(it != orderings.end());

        return it->getSecond();
      }

      void set(
          FunctionOp functionOp,
          llvm::ArrayRef<VariableOp> variablesOrder)
      {
        for (VariableOp variableOp : variablesOrder) {
          orderings[functionOp].push_back(variableOp);
        }
      }

    private:
      llvm::DenseMap<FunctionOp, llvm::SmallVector<VariableOp, 3>> orderings;
  };
}

class FunctionInliner : public mlir::OpRewritePattern<CallOp>
{
  public:
    FunctionInliner(
      mlir::MLIRContext* context,
      mlir::SymbolTableCollection& symbolTable,
      const llvm::DenseSet<FunctionOp>& inlinableFunctions,
      const DefaultOpComputationOrderings& orderings)
        : mlir::OpRewritePattern<CallOp>(context),
          symbolTable(&symbolTable),
          inlinableFunctions(&inlinableFunctions),
          orderings(&orderings)
    {
    }

    mlir::LogicalResult matchAndRewrite(
        CallOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

      FunctionOp callee = mlir::cast<FunctionOp>(
          op.getFunction(moduleOp, *symbolTable));

      if (!inlinableFunctions->contains(callee)) {
        return mlir::failure();
      }

      mlir::IRMapping mapping;
      llvm::StringMap<mlir::Value> varMapping;

      // Map the operations providing the default values for the variables.
      llvm::DenseMap<VariableOp, DefaultOp> defaultOps;

      for (DefaultOp defaultOp : callee.getDefaultValues()) {
        VariableOp variableOp = symbolTable->lookupSymbolIn<VariableOp>(
            callee, defaultOp.getVariableAttr());

        defaultOps[variableOp] = defaultOp;
      }

      // Set the default values for variables.
      for (VariableOp variableOp : orderings->get(callee)) {
        auto defaultOpIt = defaultOps.find(variableOp);

        if (defaultOpIt == defaultOps.end()) {
          continue;
        }

        DefaultOp defaultOp = defaultOpIt->getSecond();

        for (auto& nestedOp : defaultOp.getOps()) {
          if (auto yieldOp = mlir::dyn_cast<YieldOp>(nestedOp)) {
            assert(yieldOp.getValues().size() == 1);

            varMapping[variableOp.getSymName()] =
                mapping.lookup(yieldOp.getValues()[0]);
          } else {
            rewriter.clone(nestedOp, mapping);
          }
        }
      }

      // Map the call arguments to the function input variables.
      llvm::SmallVector<VariableOp, 3> inputVariables;

      for (VariableOp variableOp : callee.getVariables()) {
        if (variableOp.isInput()) {
          inputVariables.push_back(variableOp);
        }
      }

      assert(op.getArgs().size() <= inputVariables.size());

      for (const auto& callArg : llvm::enumerate(op.getArgs())) {
        VariableOp variableOp = inputVariables[callArg.index()];
        varMapping[variableOp.getSymName()] = callArg.value();
      }

      // Check that all the input variables have a value.
      assert(llvm::all_of(inputVariables, [&](VariableOp variableOp) {
        return varMapping.find(variableOp.getSymName()) != varMapping.end();
      }));

      // Clone the function body.
      for (AlgorithmOp algorithmOp : callee.getOps<AlgorithmOp>()) {
        for (auto& originalOp : algorithmOp.getOps()) {
          cloneBodyOp(rewriter, mapping, varMapping, &originalOp);
        }
      }

      // Determine the result values.
      llvm::SmallVector<VariableOp, 1> outputVariables;

      for (VariableOp variableOp : callee.getVariables()) {
        if (variableOp.isOutput()) {
          outputVariables.push_back(variableOp);
        }
      }

      assert(op.getResults().size() == outputVariables.size());
      llvm::SmallVector<mlir::Value, 1> newResults;

      for (VariableOp variableOp : outputVariables) {
        newResults.push_back(varMapping.lookup(variableOp.getSymName()));
      }

      rewriter.replaceOp(op, newResults);
      return mlir::success();
    }

  private:
    void cloneBodyOp(
      mlir::OpBuilder& builder,
      mlir::IRMapping& mapping,
      llvm::StringMap<mlir::Value>& varMapping,
      mlir::Operation* op) const
    {
      if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(op)) {
        mlir::Value& mappedValue = varMapping[variableGetOp.getVariable()];
        mapping.map(variableGetOp.getResult(), mappedValue);
        return;
      }

      if (auto variableSetOp = mlir::dyn_cast<VariableSetOp>(op)) {
        varMapping[variableSetOp.getVariable()] =
            mapping.lookup(variableSetOp.getValue());

        return;
      }

      mlir::Operation* clonedOp = builder.clone(*op, mapping);
      mapping.map(op->getResults(), clonedOp->getResults());
    }

  private:
    mlir::SymbolTableCollection* symbolTable;
    const llvm::DenseSet<FunctionOp>* inlinableFunctions;
    const DefaultOpComputationOrderings* orderings;
};

namespace
{
  class FunctionInliningPass
      : public mlir::modelica::impl::FunctionInliningPassBase<
            FunctionInliningPass>
  {
    public:
      using FunctionInliningPassBase::FunctionInliningPassBase;

      void runOnOperation() override;

    private:
      void collectGraphNodes(
        CallGraph& callGraph,
        DefaultOpComputationOrderings& orderings,
        mlir::Operation* op) const;

      void collectGraphEdges(
          CallGraph& callGraph,
          mlir::SymbolTableCollection& symbolTable,
          mlir::ModuleOp moduleOp,
          mlir::Operation* op) const;
  };
}

void FunctionInliningPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTable;

  CallGraph callGraph;
  DefaultOpComputationOrderings orderings;

  collectGraphNodes(callGraph, orderings, moduleOp);
  collectGraphEdges(callGraph, symbolTable, moduleOp, moduleOp);

  mlir::RewritePatternSet patterns(&getContext());
  auto inlinableFunctions = callGraph.getInlinableFunctions();

  patterns.add<FunctionInliner>(
      &getContext(), symbolTable, inlinableFunctions, orderings);

  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;

  if (mlir::failed(applyPatternsAndFoldGreedily(
          moduleOp, std::move(patterns), config))) {
      return signalPassFailure();
  }
}

void FunctionInliningPass::collectGraphNodes(
    CallGraph& callGraph,
    DefaultOpComputationOrderings& orderings,
    mlir::Operation* op) const
{
  if (auto functionOp = mlir::dyn_cast<FunctionOp>(op)) {
    if (canBeInlined(functionOp)) {
      callGraph.addNode(functionOp);

      llvm::StringMap<DefaultOp> defaultOps;

      for (DefaultOp defaultOp : functionOp.getDefaultValues()) {
          defaultOps[defaultOp.getVariable()] = defaultOp;
      }

      llvm::SmallVector<VariableOp, 3> inputVariables;

      for (VariableOp variableOp : functionOp.getVariables()) {
          if (variableOp.isInput()) {
            inputVariables.push_back(variableOp);
          }
      }

      DefaultValuesGraph defaultValuesGraph(defaultOps);
      defaultValuesGraph.addVariables(inputVariables);
      defaultValuesGraph.discoverDependencies();

      orderings.set(functionOp, defaultValuesGraph.reversePostOrder());
    }
  }

  for (auto& region : op->getRegions()) {
    for (auto& nested : region.getOps()) {
      collectGraphNodes(callGraph, orderings, &nested);
    }
  }
}

void FunctionInliningPass::collectGraphEdges(
    CallGraph& callGraph,
    mlir::SymbolTableCollection& symbolTable,
    mlir::ModuleOp moduleOp,
    mlir::Operation* op) const
{
  if (auto functionOp = mlir::dyn_cast<FunctionOp>(op)) {
    if (callGraph.hasNode(functionOp)) {
      functionOp.walk([&](CallOp callOp) {
        FunctionOp callee = mlir::cast<FunctionOp>(
            callOp.getFunction(moduleOp, symbolTable));

        if (callGraph.hasNode(callee)) {
          callGraph.addEdge(functionOp, callee);
        }
      });
    }
  }

  for (auto& region : op->getRegions()) {
    for (auto& nested : region.getOps()) {
      collectGraphEdges(callGraph, symbolTable, moduleOp, &nested);
    }
  }
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createFunctionInliningPass()
  {
    return std::make_unique<FunctionInliningPass>();
  }
}
