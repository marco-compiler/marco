#include "modelica/matching/Schedule.hpp"

#include <mutex>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/SCCDependencyGraph.hpp"
#include "modelica/matching/SVarDependencyGraph.hpp"
#include "modelica/matching/SccLookup.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/ThreadPool.hpp"

using namespace modelica;
using namespace llvm;

std::mutex printerLock;

static SmallVector<Assigment, 3> collapseEquations(
		const SVarDepencyGraph& originalGraph)
{
	SmallVector<Assigment, 3> out;

	IndexSet currentSet;

	const auto onSched = [&](size_t node) {
		const auto& currentNode = originalGraph[node];
		const auto& eq = currentNode.getCollapsedVertex();
		currentSet.insert(currentNode.getIndexes());
	};

	const auto onGrupEnd = [&](size_t node) {
		const auto& currentNode = originalGraph[node];
		const auto& eq = currentNode.getCollapsedVertex();
		for (const auto& set : currentSet)
			out.emplace_back(eq.getLeft(), eq.getRight(), set);

		currentSet = IndexSet();
	};

	originalGraph.topoOrder(onSched, onGrupEnd);

	return out;
}

static void sched(
		const Scc<size_t>& scc, const VVarDependencyGraph& originalGraph)
{
	SVarDepencyGraph scalarGraph(originalGraph, scc);

	SmallVector<Assigment, 3> equations(collapseEquations(scalarGraph));

	std::lock_guard guard(printerLock);
	for (const auto& eq : equations)
		eq.dump();
}

void modelica::schedule(const EntryModel& model)
{
	VVarDependencyGraph vectorGraph(model);
	auto sccs = vectorGraph.getSCC();
	SCCDependencyGraph sccDependency(sccs, vectorGraph);

	auto sortedScc = sccDependency.topologicalSort();

	{
		ThreadPool pool;
		for (auto scc : sortedScc)
			pool.addTask([scc, &vectorGraph]() { sched(*scc, vectorGraph); });
	}
}
