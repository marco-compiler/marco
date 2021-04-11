#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/mlirlowerer/passes/matching/SCCDependencyGraph.h>
#include <modelica/mlirlowerer/passes/matching/SCCLookup.h>
#include <modelica/mlirlowerer/passes/matching/SVarDependencyGraph.h>
#include <modelica/mlirlowerer/passes/matching/Schedule.h>
#include <modelica/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/utils/IndexSet.hpp>
#include <modelica/utils/ThreadPool.hpp>

using namespace modelica::codegen::model;

static llvm::SmallVector<Equation, 3> collapseEquations(
		const SVarDepencyGraph& originalGraph)
{
	llvm::SmallVector<Equation, 3> out;

	modelica::IndexSet currentSet;

	const auto onSched = [&](size_t node) {
		const auto& currentNode = originalGraph[node];
		const auto& eq = currentNode.getCollapsedVertex();
		currentSet.insert(currentNode.getIndexes());
	};

	const auto onGrupEnd = [&](size_t node,
														 khanNextPreferred schedulingDirection) {
		assert(schedulingDirection != khanNextPreferred::cannotBeOptimized);

		const bool backward = schedulingDirection == khanNextPreferred::backwardPreferred;
		const auto& currentNode = originalGraph[node];
		const auto& eq = currentNode.getCollapsedVertex();

		for (const auto& set : currentSet)
			out.emplace_back(eq.getOp(), eq.lhs(), eq.rhs(), set, !backward);

		currentSet = modelica::IndexSet();
	};

	originalGraph.topoOrder(onSched, onGrupEnd);

	return out;
}

static bool isForward(const VectorAccess* access)
{
	for (const auto& varAcc : *access)
		if (varAcc.isOffset() && varAcc.getOffset() >= 0)
			return false;

	return true;
}

static bool isBacward(const VectorAccess* access)
{
	for (const auto& varAcc : *access)
		if (varAcc.isOffset() && varAcc.getOffset() <= 0)
			return false;

	return true;
}

static llvm::SmallVector<Equation, 3> trivialScheduling(
		const SCC<VVarDependencyGraph>& scc,
		const VVarDependencyGraph& originalGraph)
{
	if (scc.size() != 1)
		return {};

	llvm::SmallVector<const VectorAccess*, 3> internalEdges;
	for (const auto& edge : originalGraph.outEdges(scc[0]))
		if (originalGraph.target(edge) == scc[0])
			internalEdges.push_back(&originalGraph[edge]);

	if (all_of(internalEdges, isForward))
	{
		auto eq = originalGraph[scc[0]].getEquation();
		eq.setForward(true);
		return { eq };
	}

	if (all_of(internalEdges, isBacward))
	{
		auto eq = originalGraph[scc[0]].getEquation();
		eq.setForward(false);
		return { eq };
	}

	return {};
}

static llvm::SmallVector<Equation, 3> sched(
		const SCC<VVarDependencyGraph>& scc,
		const VVarDependencyGraph& originalGraph)
{
	if (auto sched = trivialScheduling(scc, originalGraph); !sched.empty())
		return sched;

	SVarDepencyGraph scalarGraph(originalGraph, scc);

	return collapseEquations(scalarGraph);
}

using ResultVector = llvm::SmallVector<llvm::SmallVector<Equation, 3>, 0>;
using SortedScc = llvm::SmallVector<const SCC<VVarDependencyGraph>*, 0>;

static ResultVector parallelMap(
		const VVarDependencyGraph& vectorGraph, const SortedScc& sortedScc)
{
	ResultVector results(sortedScc.size(), {});
	modelica::ThreadPool pool;

	for (size_t i : modelica::irange(sortedScc.size()))
		pool.addTask([i, &sortedScc, &vectorGraph, &results]() {
			results[i] = sched(*sortedScc[i], vectorGraph);
		});

	return results;
}

mlir::LogicalResult modelica::codegen::model::schedule(Model& model)
{
	VVarDependencyGraph vectorGraph(model);
	SCCDependencyGraph sccDependency(vectorGraph);

	SortedScc sortedScc = sccDependency.topologicalSort();

	auto results = parallelMap(vectorGraph, sortedScc);
	llvm::SmallVector<Equation, 3> equations;

	assert(model.getOp().body().getBlocks().size() == 1);
	mlir::Operation* op = model.getOp().body().front().getTerminator();

	for (const auto& res : results)
		for (const auto& equation : res)
		{
			equation.getOp()->moveBefore(op);
			op = equation.getOp();
			equations.push_back(equation);
		}

	Model result(model.getOp(), model.getVariables(), equations);
	model = result;
	return mlir::success();
}
