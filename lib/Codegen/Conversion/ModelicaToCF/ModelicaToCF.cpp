#include "marco/Codegen/Conversion/ModelicaToCF/ModelicaToCF.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include <set>
#include <stack>

namespace mlir
{
#define GEN_PASS_DEF_MODELICATOCFCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::modelica;

static bool canBePromoted(ArrayType arrayType)
{
  return arrayType.hasStaticShape();
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

      /// A group of variables.
      struct Group
      {
        llvm::SmallVector<Node> nodes;
        bool ordered{false};
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
      void addVariablesGroup(
          llvm::ArrayRef<VariableOp> group,
          bool enforceInternalOrder)
      {
        Group nodesGroup;
        nodesGroup.ordered = enforceInternalOrder;

        for (VariableOp variable : group) {
          assert(variable != nullptr);

          Node node(this, variable.getOperation());
          nodes.push_back(node);
          nodesGroup.nodes.push_back(node);
          nodesByName[variable.getSymName()] = node;
        }

        groups.push_back(std::move(nodesGroup));
      }

      /// Discover the dependencies of the variables that have been added to
      /// the graph.
      void discoverDependencies()
      {
        for (const Group& group : groups) {
          for (const auto& node : llvm::enumerate(group.nodes)) {
            // Connect the entry node to the current one.
            arcs[getEntryNode().variable].insert(node.value());

            // Connect the current node to the other nodes to which it depends.
            mlir::Operation* variable = node.value().variable;
            auto& children = arcs[variable];

            auto variableOp = mlir::cast<VariableOp>(variable);

            for (llvm::StringRef dependency : getDependencies(variableOp)) {
              children.insert(nodesByName[dependency]);
            }

            // If the internal ordering must be preserved, then add the arcs
            // from each node of the group to its predecessor.

            if (size_t i = node.index(); i > 0 && group.ordered) {
              arcs[group.nodes[i].variable].insert(group.nodes[i - 1]);
            }
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

    protected:
      virtual std::set<llvm::StringRef> getDependencies(
          VariableOp variable) = 0;

    private:
      llvm::DenseMap<mlir::Operation*, std::set<Node>> arcs;
      llvm::SmallVector<Node> nodes;
      llvm::SmallVector<Group> groups;
      llvm::DenseMap<llvm::StringRef, Node> nodesByName;
  };

  /// Directed graph representing the dependencies among the variables with
  /// respect to the usage of variables for the computation of the dynamic
  /// dimensions of their types.
  class DynamicDimensionsGraph : public VariablesDependencyGraph
  {
    protected:
      std::set<llvm::StringRef> getDependencies(VariableOp variable) override;
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

  std::set<llvm::StringRef> DynamicDimensionsGraph::getDependencies(
      VariableOp variable)
  {
    std::set<llvm::StringRef> dependencies;
    mlir::Region& region = variable.getConstraintsRegion();

    for (VariableGetOp user : region.getOps<VariableGetOp>()) {
      dependencies.insert(user.getMember());
    }

    return dependencies;
  }

  std::set<llvm::StringRef> DefaultValuesGraph::getDependencies(
      VariableOp variable)
  {
    std::set<llvm::StringRef> dependencies;
    auto defaultOpIt = defaultOps->find(variable.getSymName());

    if (defaultOpIt != defaultOps->end()) {
      DefaultOp defaultOp = defaultOpIt->getValue();

      defaultOp->walk([&](VariableGetOp getOp) {
        dependencies.insert(getOp.getMember());
      });
    }

    return dependencies;
  }
}

class CFGLowering : public mlir::OpRewritePattern<FunctionOp>
{
  public:
    CFGLowering(mlir::MLIRContext* context, bool outputArraysPromotion)
        : mlir::OpRewritePattern<FunctionOp>(context),
          outputArraysPromotion(outputArraysPromotion)
    {
    }

    mlir::LogicalResult matchAndRewrite(
        FunctionOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      mlir::SymbolTable symbolTable(op.getOperation());

      // Discover the variables.
      llvm::SmallVector<VariableOp> inputVariables;
      llvm::SmallVector<VariableOp> outputVariables;
      llvm::SmallVector<VariableOp> promotedOutputVariables;
      llvm::SmallVector<VariableOp> protectedVariables;

      collectVariables(
          op, inputVariables, outputVariables, protectedVariables);

      // Determine the signature of the function.
      llvm::SmallVector<mlir::Type> argTypes;
      llvm::SmallVector<mlir::Type> resultTypes;

      llvm::DenseSet<VariableOp> promotedVariables;

      for (VariableOp variableOp : inputVariables) {
        mlir::Type unwrappedType = variableOp.getMemberType().unwrap();
        argTypes.push_back(unwrappedType);
      }

      for (VariableOp variableOp : outputVariables) {
        mlir::Type unwrappedType = variableOp.getMemberType().unwrap();
        resultTypes.push_back(unwrappedType);
      }

      // Create the raw function.
      auto rawFunctionOp = rewriter.create<RawFunctionOp>(
          op.getLoc(), op.getSymName(),
          rewriter.getFunctionType(argTypes, resultTypes));

      // Add the entry block and map the arguments to the input variables.
      mlir::Block* entryBlock = rawFunctionOp.addEntryBlock();
      mlir::Block* lastBlockBeforeExitBlock = entryBlock;

      llvm::StringMap<mlir::BlockArgument> argsMapping;

      for (const auto& [variableOp, arg] : llvm::zip(
               inputVariables, rawFunctionOp.getArguments())) {
        argsMapping[variableOp.getSymName()] = arg;
      }

      // Create the return block.
      mlir::Block* exitBlock = rewriter.createBlock(
          &rawFunctionOp.getFunctionBody(),
          rawFunctionOp.getFunctionBody().end());

      // Create the variables. The order in which the variables are created has
      // to take into account the dependencies of their dynamic dimensions. At
      // the same time, however, the relative order of the input and output
      // variables must be the same as the declared one, in order to preserve
      // the correctness of the calls. This last aspect is already taken into
      // account by the dependency graph.
      DynamicDimensionsGraph dynamicDimensionsGraph;

      dynamicDimensionsGraph.addVariablesGroup(inputVariables, true);
      dynamicDimensionsGraph.addVariablesGroup(outputVariables, true);
      dynamicDimensionsGraph.addVariablesGroup(protectedVariables, false);

      dynamicDimensionsGraph.discoverDependencies();

      llvm::StringMap<RawVariableOp> rawVariables;

      for (VariableOp variable : dynamicDimensionsGraph.postOrder()) {
        if (variable.isInput()) {
          // Input variables are already represented by the function arguments.
          continue;
        }

        rawVariables[variable.getSymName()] = createVariable(
            rewriter, variable, exitBlock, lastBlockBeforeExitBlock);
      }

      // Set the default values.
      llvm::StringMap<DefaultOp> defaultOps;

      for (DefaultOp defaultOp : op.getOps<DefaultOp>()) {
        defaultOps[defaultOp.getVariable()] = defaultOp;
      }

      DefaultValuesGraph defaultValuesGraph(defaultOps);

      defaultValuesGraph.addVariablesGroup(inputVariables, true);
      defaultValuesGraph.addVariablesGroup(outputVariables, true);
      defaultValuesGraph.addVariablesGroup(protectedVariables, false);

      defaultValuesGraph.discoverDependencies();

      for (VariableOp variable : defaultValuesGraph.postOrder()) {
        auto defaultOpIt = defaultOps.find(variable.getSymName());

        if (defaultOpIt != defaultOps.end()) {
          setDefaultValue(
              rewriter,
              variable,
              defaultOpIt->getValue(),
              rawVariables,
              exitBlock, lastBlockBeforeExitBlock);
        }
      }

      // Convert the algorithms.
      for (AlgorithmOp algorithmOp : op.getOps<AlgorithmOp>()) {
        mlir::Region& algorithmRegion = algorithmOp.getBodyRegion();

        if (algorithmRegion.empty()) {
          continue;
        }

        rewriter.setInsertionPointToEnd(lastBlockBeforeExitBlock);
        rewriter.create<mlir::cf::BranchOp>(loc, &algorithmRegion.front());

        if (mlir::failed(recurse(
                rewriter,
                &algorithmRegion.front(), &algorithmRegion.back(),
                nullptr, exitBlock))) {
          return mlir::failure();
        }

        lastBlockBeforeExitBlock = &algorithmRegion.back();
        rewriter.inlineRegionBefore(algorithmRegion, exitBlock);
      }

      rewriter.setInsertionPointToEnd(lastBlockBeforeExitBlock);
      rewriter.create<mlir::cf::BranchOp>(loc, exitBlock);

      // Replace symbol uses with SSA uses.
      replaceSymbolAccesses(
          rewriter, rawFunctionOp.getFunctionBody(),
          argsMapping, rawVariables);

      // Populate the exit block.
      rewriter.setInsertionPointToStart(exitBlock);
      llvm::SmallVector<mlir::Value> results;

      for (VariableOp variableOp : outputVariables) {
        RawVariableOp rawVariableOp = rawVariables[variableOp.getSymName()];

        results.push_back(rewriter.create<RawVariableGetOp>(
            variableOp->getLoc(), rawVariableOp));
      }

      rewriter.create<RawReturnOp>(loc, results);

      // Erase the original function.
      rewriter.eraseOp(op);

      return mlir::success();
    }

  private:
    void collectVariables(
      FunctionOp functionOp,
      llvm::SmallVectorImpl<VariableOp>& inputVariables,
      llvm::SmallVectorImpl<VariableOp>& outputVariables,
      llvm::SmallVectorImpl<VariableOp>& protectedVariables) const
    {
      // Keep the promoted variables in a separate list, so that they can be
      // then appended to the list of input ones.
      llvm::SmallVector<VariableOp> promotedVariables;

      for (VariableOp variableOp : functionOp.getVariables()) {
        if (variableOp.isInput()) {
          inputVariables.push_back(variableOp);
        } else if (variableOp.isOutput()) {
          mlir::Type unwrappedType = variableOp.getMemberType().unwrap();

          if (auto arrayType = unwrappedType.dyn_cast<ArrayType>();
              arrayType && outputArraysPromotion && canBePromoted(arrayType)) {
            promotedVariables.push_back(variableOp);
          } else {
            outputVariables.push_back(variableOp);
          }
        } else {
          protectedVariables.push_back(variableOp);
        }
      }

      // Append the promoted output variables to the list of input variables.
      inputVariables.append(promotedVariables);
    }

    RawVariableOp createVariable(
        mlir::PatternRewriter& rewriter,
        VariableOp variableOp,
        mlir::Block* exitBlock,
        mlir::Block*& lastBlockBeforeExitBlock) const
    {
      // Inline the operations to compute the dimensions constraints, if any.
      mlir::Region& constraintsRegion = variableOp.getConstraintsRegion();

      // The YieldOp of the constraints region will be erased, so we need to
      // store the list of its operands elsewhere.
      llvm::SmallVector<mlir::Value> constraints;

      if (!constraintsRegion.empty()) {
        auto constraintsTerminator =
            mlir::cast<YieldOp>(constraintsRegion.back().getTerminator());

        for (mlir::Value constraint : constraintsTerminator.getValues()) {
          constraints.push_back(constraint);
        }

        rewriter.eraseOp(constraintsTerminator);

        rewriter.setInsertionPointToEnd(lastBlockBeforeExitBlock);

        rewriter.create<mlir::cf::BranchOp>(
            constraintsRegion.getLoc(), &constraintsRegion.back());

        lastBlockBeforeExitBlock = &constraintsRegion.back();
        rewriter.inlineRegionBefore(constraintsRegion, exitBlock);
      }

      // Create the block containing the declaration of the variable.
      mlir::Block* variableBlock = rewriter.createBlock(exitBlock);
      rewriter.setInsertionPointToEnd(lastBlockBeforeExitBlock);
      rewriter.create<mlir::cf::BranchOp>(variableOp.getLoc(), variableBlock);
      lastBlockBeforeExitBlock = variableBlock;

      rewriter.setInsertionPointToStart(variableBlock);

      return rewriter.create<RawVariableOp>(
          variableOp.getLoc(),
          variableOp.getSymName(),
          variableOp.getMemberType(),
          variableOp.getDimensionsConstraints(),
          constraints);
    }

    void setDefaultValue(
        mlir::PatternRewriter& rewriter,
        VariableOp variableOp,
        DefaultOp defaultOp,
        const llvm::StringMap<RawVariableOp>& rawVariables,
        mlir::Block* exitBlock,
        mlir::Block*& lastBlockBeforeExitBlock) const
    {
      // Inline the operations to compute the default value, if any.
      mlir::Region& region = defaultOp.getBodyRegion();

      if (region.empty()) {
        return;
      }

      // Create the block containing the assignment of the default value.
      mlir::Block* valueAssignmentBlock = rewriter.createBlock(exitBlock);

      // Create the branch to the block computing the default value.
      rewriter.setInsertionPointToEnd(lastBlockBeforeExitBlock);
      rewriter.create<mlir::cf::BranchOp>(defaultOp.getLoc(), &region.front());

      // Inline the blocks computing the default value.
      auto terminator = mlir::cast<YieldOp>(region.back().getTerminator());
      mlir::Value value = terminator.getValues()[0];
      rewriter.setInsertionPoint(terminator);

      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
          terminator, valueAssignmentBlock);

      rewriter.inlineRegionBefore(region, valueAssignmentBlock);

      // Assign the value.
      rewriter.setInsertionPointToStart(valueAssignmentBlock);
      auto it = rawVariables.find(variableOp.getSymName());
      assert(it != rawVariables.end());
      RawVariableOp rawVariableOp = it->second;
      rewriter.create<RawVariableSetOp>(value.getLoc(), rawVariableOp, value);

      // Set the last block before the exit one.
      lastBlockBeforeExitBlock = valueAssignmentBlock;
    }

    /// Replace the references to the symbol of a variable with references to
    /// the SSA value of its equivalent "raw" variable.
    void replaceSymbolAccesses(
        mlir::PatternRewriter& rewriter,
        mlir::Region& region,
        const llvm::StringMap<mlir::BlockArgument>& inputVars,
        const llvm::StringMap<RawVariableOp>& outputAndProtectedVars) const
    {
      region.walk([&](VariableGetOp op) {
        rewriter.setInsertionPoint(op);
        auto inputVarIt = inputVars.find(op.getMember());

        if (inputVarIt != inputVars.end()) {
          rewriter.replaceOp(op, inputVarIt->getValue());
        } else {
          auto writableVarIt = outputAndProtectedVars.find(op.getMember());
          assert(writableVarIt != outputAndProtectedVars.end());
          RawVariableOp rawVariableOp = writableVarIt->getValue();
          rewriter.replaceOpWithNewOp<RawVariableGetOp>(op, rawVariableOp);
        }
      });

      region.walk([&](VariableSetOp op) {
        rewriter.setInsertionPoint(op);
        auto it = outputAndProtectedVars.find(op.getMember());
        assert(it != outputAndProtectedVars.end());
        RawVariableOp rawVariableOp = it->getValue();

        rewriter.replaceOpWithNewOp<RawVariableSetOp>(
            op, rawVariableOp, op.getValue());
      });
    }

    mlir::LogicalResult createCFG(
        mlir::PatternRewriter& rewriter,
        mlir::Operation* op,
        mlir::Block* loopExitBlock,
        mlir::Block* functionReturnBlock) const
    {
      if (auto breakOp = mlir::dyn_cast<BreakOp>(op)) {
        return createCFG(rewriter, breakOp, loopExitBlock);
      }

      if (auto forOp = mlir::dyn_cast<ForOp>(op)) {
        return createCFG(rewriter, forOp, functionReturnBlock);
      }

      if (auto ifOp = mlir::dyn_cast<IfOp>(op)) {
        return createCFG(rewriter, ifOp, loopExitBlock, functionReturnBlock);
      }

      if (auto whileOp = mlir::dyn_cast<WhileOp>(op)) {
        return createCFG(rewriter, whileOp, functionReturnBlock);
      }

      if (auto returnOp = mlir::dyn_cast<ReturnOp>(op)) {
        return createCFG(rewriter, returnOp, functionReturnBlock);
      }

      return mlir::success();
    }

    mlir::LogicalResult createCFG(
        mlir::PatternRewriter& rewriter,
        BreakOp op,
        mlir::Block* loopExitBlock) const
    {
      if (loopExitBlock == nullptr) {
        return mlir::failure();
      }

      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);

      mlir::Block* currentBlock = rewriter.getInsertionBlock();
      rewriter.splitBlock(currentBlock, op->getIterator());

      rewriter.setInsertionPointToEnd(currentBlock);
      rewriter.create<mlir::cf::BranchOp>(op->getLoc(), loopExitBlock);

      rewriter.eraseOp(op);
      return mlir::success();
    }

    mlir::LogicalResult createCFG(
        mlir::PatternRewriter& rewriter,
        ForOp op,
        mlir::Block* functionReturnBlock) const
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);

      // Split the current block.
      mlir::Block* currentBlock = rewriter.getInsertionBlock();

      mlir::Block* continuation =
          rewriter.splitBlock(currentBlock, op->getIterator());

      // Keep the references to the op blocks.
      mlir::Block* conditionFirst = &op.getConditionRegion().front();
      mlir::Block* conditionLast = &op.getConditionRegion().back();
      mlir::Block* bodyFirst = &op.getBodyRegion().front();
      mlir::Block* bodyLast = &op.getBodyRegion().back();
      mlir::Block* stepFirst = &op.getStepRegion().front();
      mlir::Block* stepLast = &op.getStepRegion().back();

      // Inline the regions.
      rewriter.inlineRegionBefore(op.getConditionRegion(), continuation);
      rewriter.inlineRegionBefore(op.getBodyRegion(), continuation);
      rewriter.inlineRegionBefore(op.getStepRegion(), continuation);

      // Start the for loop by branching to the "condition" region.
      rewriter.setInsertionPointToEnd(currentBlock);

      rewriter.create<mlir::cf::BranchOp>(
          op->getLoc(), conditionFirst, op.getArgs());

      // Check the condition.
      auto conditionOp =
          mlir::cast<ConditionOp>(conditionLast->getTerminator());

      rewriter.setInsertionPoint(conditionOp);

      mlir::Value conditionValue = rewriter.create<CastOp>(
          conditionOp.getCondition().getLoc(),
          rewriter.getI1Type(),
          conditionOp.getCondition());

      rewriter.create<mlir::cf::CondBranchOp>(
          conditionOp->getLoc(),
          conditionValue,
          bodyFirst, conditionOp.getValues(),
          continuation, llvm::None);

      rewriter.eraseOp(conditionOp);

      // If present, replace "body" block terminator with a branch to the
      // "step" block. If not present, just place the branch.
      rewriter.setInsertionPointToEnd(bodyLast);
      llvm::SmallVector<mlir::Value, 3> bodyYieldValues;

      if (auto yieldOp = mlir::dyn_cast<YieldOp>(bodyLast->back())) {
        for (mlir::Value value : yieldOp.getValues()) {
          bodyYieldValues.push_back(value);
        }

        rewriter.eraseOp(yieldOp);
      }

      rewriter.create<mlir::cf::BranchOp>(
          op->getLoc(), stepFirst, bodyYieldValues);

      // Branch to the condition check after incrementing the induction
      // variable.
      rewriter.setInsertionPointToEnd(stepLast);
      llvm::SmallVector<mlir::Value, 3> stepYieldValues;

      if (auto yieldOp = mlir::dyn_cast<YieldOp>(stepLast->back())) {
        for (mlir::Value value : yieldOp.getValues()) {
          stepYieldValues.push_back(value);
        }

        rewriter.eraseOp(yieldOp);
      }

      rewriter.create<mlir::cf::BranchOp>(
          op->getLoc(), conditionFirst, stepYieldValues);

      // Erase the operation.
      rewriter.eraseOp(op);

      // Recurse on the body operations.
      return recurse(
          rewriter, bodyFirst, bodyLast, continuation, functionReturnBlock);
    }

    mlir::LogicalResult createCFG(
        mlir::PatternRewriter& rewriter,
        IfOp op,
        mlir::Block* loopExitBlock,
        mlir::Block* functionReturnBlock) const
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);

      // Split the current block.
      mlir::Block* currentBlock = rewriter.getInsertionBlock();

      mlir::Block* continuation =
          rewriter.splitBlock(currentBlock, op->getIterator());

      // Keep the references to the op blocks.
      mlir::Block* thenFirst = &op.getThenRegion().front();
      mlir::Block* thenLast = &op.getThenRegion().back();

      // Inline the regions.
      rewriter.inlineRegionBefore(op.getThenRegion(), continuation);
      rewriter.setInsertionPointToEnd(currentBlock);

      mlir::Value conditionValue = rewriter.create<CastOp>(
          op.getCondition().getLoc(), rewriter.getI1Type(), op.getCondition());

      if (op.getElseRegion().empty()) {
        // Branch to the "then" region or to the continuation block according
        // to the condition.

        rewriter.create<mlir::cf::CondBranchOp>(
            op->getLoc(),
            conditionValue,
            thenFirst, llvm::None,
            continuation, llvm::None);

        rewriter.setInsertionPointToEnd(thenLast);
        rewriter.create<mlir::cf::BranchOp>(op->getLoc(), continuation);

        // Erase the operation.
        rewriter.eraseOp(op);

        // Recurse on the body operations.
        if (mlir::failed(recurse(
                rewriter,
                thenFirst, thenLast,
                loopExitBlock, functionReturnBlock))) {
          return mlir::failure();
        }
      } else {
        // Branch to the "then" region or to the "else" region according
        // to the condition.
        mlir::Block* elseFirst = &op.getElseRegion().front();
        mlir::Block* elseLast = &op.getElseRegion().back();

        rewriter.inlineRegionBefore(op.getElseRegion(), continuation);

        rewriter.create<mlir::cf::CondBranchOp>(
            op->getLoc(),
            conditionValue,
            thenFirst, llvm::None,
            elseFirst, llvm::None);

        // Branch to the continuation block.
        rewriter.setInsertionPointToEnd(thenLast);
        rewriter.create<mlir::cf::BranchOp>(op->getLoc(), continuation);

        rewriter.setInsertionPointToEnd(elseLast);
        rewriter.create<mlir::cf::BranchOp>(op->getLoc(), continuation);

        // Erase the operation.
        rewriter.eraseOp(op);

        if (mlir::failed(recurse(
                rewriter,
                elseFirst, elseLast,
                loopExitBlock, functionReturnBlock))) {
          return mlir::failure();
        }
      }

      return mlir::success();
    }

    mlir::LogicalResult createCFG(
        mlir::PatternRewriter& rewriter,
        WhileOp op,
        mlir::Block* functionReturnBlock) const
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);

      // Split the current block.
      mlir::Block* currentBlock = rewriter.getInsertionBlock();

      mlir::Block* continuation =
          rewriter.splitBlock(currentBlock, op->getIterator());

      // Keep the references to the op blocks.
      mlir::Block* conditionFirst = &op.getConditionRegion().front();
      mlir::Block* conditionLast = &op.getConditionRegion().back();

      mlir::Block* bodyFirst = &op.getBodyRegion().front();
      mlir::Block* bodyLast = &op.getBodyRegion().back();

      // Inline the regions.
      rewriter.inlineRegionBefore(op.getConditionRegion(), continuation);
      rewriter.inlineRegionBefore(op.getBodyRegion(), continuation);

      // Branch to the "condition" region.
      rewriter.setInsertionPointToEnd(currentBlock);
      rewriter.create<mlir::cf::BranchOp>(op->getLoc(), conditionFirst);

      // Branch to the "body" region.
      rewriter.setInsertionPointToEnd(conditionLast);

      auto conditionOp = mlir::cast<ConditionOp>(
          conditionLast->getTerminator());

      mlir::Value conditionValue = rewriter.create<CastOp>(
          conditionOp->getLoc(),
          rewriter.getI1Type(),
          conditionOp.getCondition());

      rewriter.create<mlir::cf::CondBranchOp>(
          op->getLoc(),
          conditionValue,
          bodyFirst, llvm::None,
          continuation, llvm::None);

      rewriter.eraseOp(conditionOp);

      // Branch back to the "condition" region.
      rewriter.setInsertionPointToEnd(bodyLast);
      rewriter.create<mlir::cf::BranchOp>(op->getLoc(), conditionFirst);

      // Erase the operation.
      rewriter.eraseOp(op);

      // Recurse on the body operations.
      return recurse(
          rewriter, bodyFirst, bodyLast, continuation, functionReturnBlock);
    }

    mlir::LogicalResult createCFG(
        mlir::PatternRewriter& rewriter,
        ReturnOp op,
        mlir::Block* functionReturnBlock) const
    {
      if (functionReturnBlock == nullptr) {
        return mlir::failure();
      }

      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);

      mlir::Block* currentBlock = rewriter.getInsertionBlock();
      rewriter.splitBlock(currentBlock, op->getIterator());

      rewriter.setInsertionPointToEnd(currentBlock);
      rewriter.create<mlir::cf::BranchOp>(op->getLoc(), functionReturnBlock);

      rewriter.eraseOp(op);
      return mlir::success();
    }

    mlir::LogicalResult recurse(
        mlir::PatternRewriter& rewriter,
        mlir::Block* first,
        mlir::Block* last,
        mlir::Block* loopExitBlock,
        mlir::Block* functionReturnBlock) const
    {
      llvm::SmallVector<mlir::Operation*> ops;
      auto it = first->getIterator();

      do {
        for (auto& op : it->getOperations()) {
          ops.push_back(&op);
        }
      } while (it++ != last->getIterator());

      for (auto& op : ops) {
        if (mlir::failed(
                createCFG(rewriter, op, loopExitBlock, functionReturnBlock))) {
          return mlir::failure();
        }
      }

      return mlir::success();
    }

  private:
    bool outputArraysPromotion;
};

namespace
{
  class ModelicaToCFConversionPass
      : public mlir::impl::ModelicaToCFConversionPassBase<
          ModelicaToCFConversionPass>
  {
    public:
      using ModelicaToCFConversionPassBase::ModelicaToCFConversionPassBase;

      void runOnOperation() override
      {
        mlir::ModuleOp moduleOp = getOperation();

        if (mlir::failed(convertModelicaToCFG(moduleOp))) {
          mlir::emitError(
              getOperation().getLoc(),
              "Can't compute CFG of Modelica functions");

          return signalPassFailure();
        }

        if (outputArraysPromotion) {
          if (mlir::failed(promoteCallResults(moduleOp))) {
            return signalPassFailure();
          }
        }
      }

      mlir::LogicalResult convertModelicaToCFG(mlir::ModuleOp moduleOp)
      {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<CFGLowering>(&getContext(), outputArraysPromotion);

        mlir::GreedyRewriteConfig config;
        config.useTopDownTraversal = true;

        return applyPatternsAndFoldGreedily(
           moduleOp, std::move(patterns), config);
      }

      mlir::LogicalResult promoteCallResults(mlir::ModuleOp moduleOp)
      {
        mlir::OpBuilder builder(moduleOp);

        moduleOp.walk([&](CallOp callOp) {
          builder.setInsertionPoint(callOp);

          llvm::SmallVector<mlir::Value> args;

          for (mlir::Value arg : callOp.getArgs()) {
            args.push_back(arg);
          }

          llvm::SmallVector<mlir::Type> resultTypes;
          llvm::DenseMap<size_t, size_t> resultsMap;
          size_t newResultsCounter = 0;

          for (const auto& result : llvm::enumerate(callOp->getResults())) {
            mlir::Type resultType = result.value().getType();

            if (auto arrayType = resultType.dyn_cast<ArrayType>();
                arrayType && canBePromoted(arrayType)) {
              // Allocate the array inside the caller body.
              mlir::Value array = builder.create<AllocOp>(
                  callOp.getLoc(), arrayType, llvm::None);

              // Add the array to the arguments.
              args.push_back(array);

              // Replace the usages of the old result.
              result.value().replaceAllUsesWith(array);
            } else {
              resultTypes.push_back(resultType);
              resultsMap[newResultsCounter++] = result.index();
            }
          }

          // Create the new function call.
          auto newCallOp = builder.create<CallOp>(
              callOp.getLoc(), callOp.getCallee(), resultTypes, args);

          // Replace the non-promoted old results.
          for (size_t i = 0; i < newResultsCounter; ++i) {
            callOp.getResult(resultsMap[i])
                .replaceAllUsesWith(newCallOp.getResult(i));
          }

          // Erase the old function call.
          callOp.erase();
        });

        return mlir::success();
      }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createModelicaToCFConversionPass()
  {
    return std::make_unique<ModelicaToCFConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelicaToCFConversionPass(
      const ModelicaToCFConversionPassOptions& options)
  {
    return std::make_unique<ModelicaToCFConversionPass>(options);
  }
}
