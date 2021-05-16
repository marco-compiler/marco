#include "marco/matching/Schedule.hpp"

#include <variant>

#include "llvm/ADT/SmallVector.h"
#include "marco/matching/SCCDependencyGraph.hpp"
#include "marco/matching/SVarDependencyGraph.hpp"
#include "marco/matching/VVarDependencyGraph.hpp"
#include "modelica/model/ModBltBlock.hpp"
#include "marco/model/ModEquation.hpp"
#include "modelica/model/ScheduledModel.hpp"
#include "marco/model/VectorAccess.hpp"
#include "marco/utils/IndexSet.hpp"
#include "marco/utils/ThreadPool.hpp"

using namespace marco;
using namespace llvm;
using namespace std;

static SmallVector<variant<ModEquation, ModBltBlock>, 3> collapseEquations(
		const SVarDepencyGraph& originalGraph)
{
	SmallVector<variant<ModEquation, ModBltBlock>, 3> out;

	IndexSet currentSet;

	const auto onSched = [&](size_t node) {
		const auto& currentNode = originalGraph[node];
		const auto& eq = currentNode.getCollapsedVertex();
		currentSet.insert(currentNode.getIndexes());
	};

	const auto onGroupEnd = [&](size_t node,
															khanNextPreferred schedulingDirection) {
		assert(schedulingDirection != khanNextPreferred::cannotBeOptimized);

		const bool backward =
				schedulingDirection == khanNextPreferred::backwardPreferred;
		const auto& currentNode = originalGraph[node];
		const auto& content = currentNode.getCollapsedVertex();
		if (holds_alternative<ModEquation>(content))
		{
			const ModEquation& eq = get<ModEquation>(content);
			for (const auto& set : currentSet)
				out.emplace_back(ModEquation(eq.getTemplate(), set, !backward));
		}
		else	// TODO: Check if ModEquation/ModBltBlock differentiation is correct
		{
			assert(false && "Need to be checked");
			ModBltBlock bltBlock = get<ModBltBlock>(content);
			bltBlock.setForward(!backward);
			out.emplace_back(bltBlock);
		}

		currentSet = IndexSet();
	};

	originalGraph.topoOrder(onSched, onGroupEnd);

	return out;
}

static bool isForward(const VectorAccess* access)
{
	for (const auto& varAcc : *access)
		if (varAcc.isOffset() && varAcc.getOffset() >= 0)
			return false;

	return true;
}

static bool isBackward(const VectorAccess* access)
{
	for (const auto& varAcc : *access)
		if (varAcc.isOffset() && varAcc.getOffset() <= 0)
			return false;

	return true;
}

static SmallVector<variant<ModEquation, ModBltBlock>, 3> trivialScheduling(
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
		auto content = originalGraph[scc[0]].getContent();
		if (holds_alternative<ModEquation>(content))
			get<ModEquation>(content).setForward(true);
		else
			get<ModBltBlock>(content).setForward(true);
		return { move(content) };
	}

	if (all_of(internalEdges, isBackward))
	{
		auto content = originalGraph[scc[0]].getContent();
		if (holds_alternative<ModEquation>(content))
			get<ModEquation>(content).setForward(false);
		else
			get<ModBltBlock>(content).setForward(false);
		return { move(content) };
	}

	return {};
}

static SmallVector<variant<ModEquation, ModBltBlock>, 3> sched(
		const Scc<VVarDependencyGraph>& scc,
		const VVarDependencyGraph& originalGraph)
{
	if (auto sched = trivialScheduling(scc, originalGraph); !sched.empty())
		return sched;

	SVarDepencyGraph scalarGraph(originalGraph, scc);

	return collapseEquations(scalarGraph);
}

using ResultVector =
		SmallVector<SmallVector<variant<ModEquation, ModBltBlock>, 3>, 0>;
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

ScheduledModel marco::schedule(const Model& model)
{
	VVarDependencyGraph vectorGraph(model);
	SCCDependencyGraph sccDependency(vectorGraph);

	SortedScc sortedScc = sccDependency.topologicalSort();

	auto results = parallelMap(vectorGraph, sortedScc);

	ScheduledModel scheduledModel(move(model.getVars()));
	for (const auto& res : results)
		for (const auto& eq : res)
			scheduledModel.addUpdate(move(eq));

	return scheduledModel;
}
