#include "marco/matching/Schedule.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/matching/SCCDependencyGraph.hpp"
#include "marco/matching/SVarDependencyGraph.hpp"
#include "marco/matching/SccLookup.hpp"
#include "marco/matching/VVarDependencyGraph.hpp"
#include "marco/model/Assigment.hpp"
#include "marco/model/AssignModel.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/VectorAccess.hpp"
#include "marco/utils/IndexSet.hpp"
#include "marco/utils/ThreadPool.hpp"

using namespace marco;
using namespace llvm;

static SmallVector<ModEquation, 3> collapseEquations(
		const SVarDepencyGraph& originalGraph)
{
	SmallVector<ModEquation, 3> out;

	IndexSet currentSet;

	const auto onSched = [&](size_t node) {
		const auto& currentNode = originalGraph[node];
		const auto& eq = currentNode.getCollapsedVertex();
		currentSet.insert(currentNode.getIndexes());
	};

	const auto onGrupEnd = [&](size_t node,
														 khanNextPreferred schedulingDirection) {
		assert(schedulingDirection != khanNextPreferred::cannotBeOptimized);

		const bool backward =
				schedulingDirection == khanNextPreferred::backwardPreferred;
		const auto& currentNode = originalGraph[node];
		const auto& eq = currentNode.getCollapsedVertex();
		for (const auto& set : currentSet)
			out.emplace_back(eq.getTemplate(), set, !backward);

		currentSet = IndexSet();
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

static SmallVector<ModEquation, 3> trivialScheduling(
		const Scc<VVarDependencyGraph>& scc,
		const VVarDependencyGraph& originalGraph)
{
	if (scc.size() != 1)
		return {};

	SmallVector<const VectorAccess*, 3> internalEdges;
	for (const auto& edge : originalGraph.outEdges(scc[0]))
		if (originalGraph.target(edge) == scc[0])
			internalEdges.push_back(&originalGraph[edge]);

	if (all_of(internalEdges, isForward))
	{
		auto eq = originalGraph[scc[0]].getEquation();
		eq.setForward(true);
		return { std::move(eq) };
	}

	if (all_of(internalEdges, isBacward))
	{
		auto eq = originalGraph[scc[0]].getEquation();
		eq.setForward(false);
		return { std::move(eq) };
	}

	return {};
}

static SmallVector<ModEquation, 3> sched(
		const Scc<VVarDependencyGraph>& scc,
		const VVarDependencyGraph& originalGraph)
{
	if (auto sched = trivialScheduling(scc, originalGraph); !sched.empty())
		return sched;

	SVarDepencyGraph scalarGraph(originalGraph, scc);

	return collapseEquations(scalarGraph);
}

using ResultVector = SmallVector<SmallVector<ModEquation, 3>, 0>;
using SortedScc = SmallVector<const Scc<VVarDependencyGraph>*, 0>;

static ResultVector parallelMap(
		const VVarDependencyGraph& vectorGraph, const SortedScc& sortedScc)
{
	ResultVector results(sortedScc.size(), {});

	ThreadPool pool;
	for (size_t i : irange(sortedScc.size()))
		pool.addTask([i, &sortedScc, &vectorGraph, &results]() {
			results[i] = sched(*sortedScc[i], vectorGraph);
		});

	return results;
}

Model marco::schedule(const Model& model)
{
	VVarDependencyGraph vectorGraph(model);
	SCCDependencyGraph sccDependency(vectorGraph);

	SortedScc sortedScc = sccDependency.topologicalSort();

	auto results = parallelMap(vectorGraph, sortedScc);

	Model scheduledModel({}, std::move(model.getVars()));
	for (const auto& res : results)
		for (const auto& eq : res)
			scheduledModel.addEquation(std::move(eq));

	return scheduledModel;
}
