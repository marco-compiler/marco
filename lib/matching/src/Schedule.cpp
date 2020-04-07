#include "modelica/matching/Schedule.hpp"

#include "llvm/ADT/ArrayRef.h"
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
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/ThreadPool.hpp"

using namespace modelica;
using namespace llvm;

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

static SmallVector<Assigment, 3> sched(
		const Scc<size_t>& scc, const VVarDependencyGraph& originalGraph)
{
	SVarDepencyGraph scalarGraph(originalGraph, scc);

	return collapseEquations(scalarGraph);
}

using ResultVector = SmallVector<SmallVector<Assigment, 3>, 0>;
using SortedScc = SmallVector<const Scc<size_t>*, 0>;

static ResultVector parallelMap(
		const VVarDependencyGraph& vectorGraph, const SortedScc& sortedScc)
{
	ResultVector results(sortedScc.size(), {});

	ThreadPool pool;
	llvm::outs() << "avialable scc " << sortedScc.size() << "\n";
	for (size_t i : irange(sortedScc.size()))
		pool.addTask([i, &sortedScc, &vectorGraph, &results]() {
			results[i] = sched(*sortedScc[i], vectorGraph);
			llvm::outs() << "i: " << i << "\n";
		});

	return results;
}

AssignModel modelica::schedule(const EntryModel& model)
{
	VVarDependencyGraph vectorGraph(model);
	auto sccs = vectorGraph.getSCC();
	SCCDependencyGraph sccDependency(sccs, vectorGraph);

	SortedScc sortedScc = sccDependency.topologicalSort();

	auto results = parallelMap(vectorGraph, sortedScc);
	llvm::outs() << results.size() << "asd \n";

	AssignModel scheduledModel(std::move(model.getVars()));
	for (const auto& res : results)
		for (const auto& eq : res)
			scheduledModel.addUpdate(std::move(eq));

	return scheduledModel;
}
