#include "modelica/matching/Schedule.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/SCCDependencyGraph.hpp"
#include "modelica/matching/SVarDependencyGraph.hpp"
#include "modelica/matching/SccLookup.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/ThreadPool.hpp"

using namespace modelica;
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
		const Scc<size_t>& scc, const VVarDependencyGraph& originalGraph)
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
		const Scc<size_t>& scc, const VVarDependencyGraph& originalGraph)
{
	if (auto sched = trivialScheduling(scc, originalGraph); !sched.empty())
		return sched;

	SVarDepencyGraph scalarGraph(originalGraph, scc);

	return collapseEquations(scalarGraph);
}

using ResultVector = SmallVector<SmallVector<ModEquation, 3>, 0>;
using SortedScc = SmallVector<const Scc<size_t>*, 0>;

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

EntryModel modelica::schedule(const EntryModel& model)
{
	VVarDependencyGraph vectorGraph(model);
	auto sccs = vectorGraph.getSCC();
	SCCDependencyGraph sccDependency(sccs, vectorGraph);

	SortedScc sortedScc = sccDependency.topologicalSort();

	auto results = parallelMap(vectorGraph, sortedScc);

	EntryModel scheduledModel({}, std::move(model.getVars()));
	for (const auto& res : results)
		for (const auto& eq : res)
			scheduledModel.addEquation(std::move(eq));

	return scheduledModel;
}
