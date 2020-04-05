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
		const SVarDepencyGraph& originalGraph, ArrayRef<size_t> schedule)
{
	SmallVector<Assigment, 3> out;

	const ModEquation* lastEquation = nullptr;
	IndexSet currentSet;

	for (size_t node : schedule)
	{
		const auto& currentNode = originalGraph[node];
		const auto& eq = currentNode.getCollapsedVertex();

		if (lastEquation != &eq && lastEquation != nullptr)
		{
			for (const auto& set : currentSet)
				out.emplace_back(eq.getLeft(), eq.getRight(), set);

			currentSet = IndexSet();
		}

		currentSet.insert(currentNode.getIndexes());
		lastEquation = &eq;
	}

	return out;
}

static void sched(
		const Scc<size_t>& scc, const VVarDependencyGraph& originalGraph)
{
	SVarDepencyGraph scalarGraph(originalGraph, scc);
	SmallVector<size_t, 0> out(scalarGraph.count());
	scalarGraph.topoOrder(out.rbegin());

	SmallVector<Assigment, 3> equations(collapseEquations(scalarGraph, out));

	std::lock_guard guard(printerLock);
	for (size_t vertex : out)
		scalarGraph[vertex].dump();
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
