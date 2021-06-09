#include "marco/matching/SccCollapsing.hpp"

#include <algorithm>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <numeric>
#include <vector>

template<typename Graph>
void renumber_vertex_indices(const Graph& graph)
{
	assert(
			false && "YOU CANNOT USE THIS ALGORITHM BECAUSE RENUMBER_VERTEX_INDICIES "
							 "IS NOT SUPPORTED");
}
#include "boost/graph/lookup_edge.hpp"
#include "boost/graph/tiernan_all_cycles.hpp"
#include "marco/matching/SccLookup.hpp"
#include "marco/matching/VVarDependencyGraph.hpp"
#include "marco/model/LinSolver.hpp"
#include "modelica/model/ModBltBlock.hpp"
#include "marco/model/ModEquation.hpp"
#include "modelica/model/ModErrors.hpp"
#include "marco/model/ModVariable.hpp"
#include "marco/model/Model.hpp"
#include "marco/model/VectorAccess.hpp"
#include "marco/utils/IndexSet.hpp"
#include "marco/utils/Interval.hpp"

using namespace std;
using namespace marco;
using namespace llvm;

using EqVector = SmallVector<ModEquation, 3>;

using EdDescVector = SmallVector<VVarDependencyGraph::EdgeDesc, 3>;
using DependenciesVector = SmallVector<VectorAccess, 3>;
using IndexSetVector = SmallVector<MultiDimInterval, 3>;

static EdDescVector cycleToEdgeVec(
		vector<VVarDependencyGraph::VertexDesc> c, const VVarDependencyGraph& graph)
{
	EdDescVector v;
	for (size_t a : irange(c.size()))
	{
		auto vertex = c[a];
		auto nextVertex = c[(a + 1) % c.size()];
		auto [d, exists] = lookup_edge(vertex, nextVertex, graph.getImpl());
		assert(exists);
		v.emplace_back(d);
	}

	return v;
}

static DependenciesVector cycleToDependenciesVector(
		const EdDescVector& c, const VVarDependencyGraph& graph)
{
	DependenciesVector v;

	for (auto e : c)
	{
		const IndexesOfEquation& varToEq = graph[source(e, graph.getImpl())];
		v.emplace_back(varToEq.getVarToEq() * graph[e]);
	}

	return v;
}

static bool cycleHasIndentityDependency(
		const EdDescVector& c,
		const VVarDependencyGraph& graph,
		const DependenciesVector& dep)
{
	VectorAccess fin = accumulate(
			dep.begin() + 1, dep.end(), dep[0], [](const auto& l, const auto& r) {
				return l * r;
			});

	return fin.isIdentity();
}

static MultiDimInterval cyclicDependentSet(
		const EdDescVector& c,
		const VVarDependencyGraph& graph,
		const DependenciesVector& dep)
{
	const IndexesOfEquation& firstEq = graph[source(c[0], graph.getImpl())];
	MultiDimInterval set = firstEq.getInterval();
	for (size_t i : irange(c.size()))
	{
		IndexesOfEquation eq = graph[target(c[i], graph.getImpl())];
		set = intersection(dep[i].map(set), eq.getInterval());
	}

	return set;
}

static IndexSetVector cyclicDependentSets(
		const EdDescVector& c, const VVarDependencyGraph& graph)
{
	DependenciesVector dep = cycleToDependenciesVector(c, graph);
	MultiDimInterval cyclicSet = cyclicDependentSet(c, graph, dep);
	IndexSetVector v({ cyclicSet });

	for (size_t i : irange(c.size() - 1))
		v.emplace_back(dep[i].map(v.back()));

	assert(dep.size() == c.size() && v.size() == c.size());

	if (!cycleHasIndentityDependency(c, graph, dep))
		return v;

	for (size_t i : irange(v.size()))
	{
		const IndexesOfEquation& eq = graph[source(c[i], graph.getImpl())];
		v[i] = eq.getVarToEq().map(v[i]);
		assert(eq.getEquation().getInductions().contains(v[i]));
	}

	return v;
}

static int a = 0;
static llvm::Expected<bool> extractEquationWithDependencies(
		EqVector& source,
		EqVector& filtered,
		EqVector& untouched,
		const vector<VVarDependencyGraph::VertexDesc>& cycle,
		const VVarDependencyGraph& g)
{
	EdDescVector c = cycleToEdgeVec(cycle, g);
	IndexSetVector vecSet = cyclicDependentSets(c, g);
	if (vecSet[0].empty())
		return false;

	// for each equation in the cycle
	for (size_t i : irange(cycle.size()))
	{
		const ModEquation& eq = g[boost::source(c[i], g.getImpl())].getEquation();
		// copy the equation
		ModEquation toFuseEq =
				eq.clone(eq.getTemplate()->getName() + "merged" + to_string(a++));

		// set induction to those that generate the circular dependency
		if (!toFuseEq.getInductions().contains(vecSet[i]))
			return make_error<UnsolvableAlgebraicLoop>();

		toFuseEq.setInductionVars(vecSet[i]);
		if (auto error = toFuseEq.explicitate(); error)
			return move(error);

		// add it to the list of filtered with normalized body if there is no loop
		auto normEq = toFuseEq.normalizeMatched();
		if (!normEq)
			return normEq.takeError();
		filtered.emplace_back(*normEq);

		// then for all other index set that
		// are not in the circular set
		IndexSet nonUsed = remove(eq.getInductions(), vecSet[i]);
		for (MultiDimInterval set : nonUsed)
		{
			// add the equation to the untouched set
			untouched.emplace_back(eq);
			// and set the inductions to the ones  that have no circular dependencies
			untouched.back().setInductionVars(set);
		}
	}

	// for all equations that were not in the circular set, add it to the
	// untouched set.
	for (size_t i : irange(source.size()))
	{
		if (find(cycle, i) == cycle.end())
			untouched.emplace_back(move(source[i]));
	}
	return true;
}

class CycleFuser
{
	public:
	CycleFuser(
			bool& f,
			EqVector& equs,
			const VVarDependencyGraph& graph,
			llvm::Error* e)
			: foundOne(&f), equs(&equs), graph(&graph), error(e)
	{
	}

	template<typename Graph>
	void cycle(
			const vector<VVarDependencyGraph::VertexDesc>& cycle, const Graph& g)
	{
		if (*foundOne)
			return;

		EqVector newEqus;
		EqVector filtered;
		auto err = extractEquationWithDependencies(
				*equs, filtered, newEqus, cycle, *graph);
		if (!err)
		{
			*error = err.takeError();
			*foundOne = true;
			return;
		}

		if (!*err)
			return;

		auto e = linearySolve(filtered);
		if (e)
		{
			*error = move(e);
			*foundOne = true;
			return;
		}

		for (ModEquation& eq : filtered)
			newEqus.emplace_back(move(eq));

		*foundOne = true;
		*equs = move(newEqus);
	}

	private:
	bool* foundOne;
	EqVector* equs;
	const VVarDependencyGraph* graph;
	llvm::Error* error;
};

static Error fuseEquations(
		EqVector& equs, const Model& sourceModel, size_t maxIterations)
{
	bool atLeastOneCollapse = false;
	size_t currIterations = 0;
	do
	{
		atLeastOneCollapse = false;
		VVarDependencyGraph vectorGraph(sourceModel, equs);
		llvm::Error e(llvm::Error::success());
		if (e)
			return e;
		tiernan_all_cycles(
				vectorGraph.getImpl(),
				CycleFuser(atLeastOneCollapse, equs, vectorGraph, &e));
		if (e)
			return e;

		if (++currIterations == maxIterations)
			return Error::success();
	} while (atLeastOneCollapse);
	return Error::success();
}

static Error fuseScc(
		const Scc<VVarDependencyGraph>& scc,
		const VVarDependencyGraph& vectorGraph,
		EqVector& out,
		size_t maxIterations)
{
	out.reserve(scc.size());

	for (const IndexesOfEquation& eq : scc.range(vectorGraph))
		out.push_back(eq.getEquation());

	if (auto error = fuseEquations(out, vectorGraph.getModel(), maxIterations);
			error)
		return error;

	return Error::success();
}

Expected<Model> marco::solveScc(Model&& model, size_t maxIterations)
{
	VVarDependencyGraph vectorGraph(model);
	SccLookup sccs(vectorGraph);

	SmallVector<EqVector, 3> possibleEq(sccs.count());
	SmallVector<ModBltBlock, 3> algebraicLoops;

	// For each Scc, try to collapse it, otherwise add it to the loop list
	for (size_t i : irange(sccs.count()))
	{
		if (auto error =
						fuseScc(sccs[i], vectorGraph, possibleEq[i], maxIterations);
				error)
		{
			if (!error.isA<UnsolvableAlgebraicLoop>())
				return move(error);

			// If the Scc Collapsing algorithm fails, it means that we have
			// an Algebraic Loop, which must be handled by a solver afterwards.
			llvm::SmallVector<ModEquation, 3> bltEquations;
			for (const IndexesOfEquation& eq : sccs[i].range(vectorGraph))
				bltEquations.push_back(eq.getEquation());

			algebraicLoops.push_back(
					ModBltBlock(bltEquations, "loop" + to_string(algebraicLoops.size())));

			possibleEq[i].clear();
			consumeError(move(error));
		}
	}

	Model outModel({}, move(model.getVars()));
	for (EqVector& eqList : possibleEq)
		for (ModEquation& eq : eqList)
			outModel.addEquation(move(eq));
	for (ModBltBlock& algebraicLoop : algebraicLoops)
		outModel.addBltBlock(algebraicLoop);

	return outModel;
}
