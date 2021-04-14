#include "modelica/matching/SccCollapsing.hpp"

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
#include "modelica/matching/SccLookup.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"
#include "modelica/model/LinSolver.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/Model.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/Interval.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

using EqVector = SmallVector<ModEquation, 3>;

using EdDescVector = SmallVector<VVarDependencyGraph::EdgeDesc, 3>;
using DendenciesVector = SmallVector<VectorAccess, 3>;
using InexSetVector = SmallVector<MultiDimInterval, 3>;

static EdDescVector cycleToEdgeVec(
		vector<VVarDependencyGraph::VertexDesc> c, const VVarDependencyGraph& graph)
{
	EdDescVector v;
	for (auto a : irange(c.size()))
	{
		auto vertex = c[a];
		auto nextVertex = c[(a + 1) % c.size()];
		auto [d, exists] = lookup_edge(vertex, nextVertex, graph.getImpl());
		assert(exists);
		v.emplace_back(d);
	}

	return v;
}

static DendenciesVector cycleToDependencieVector(
		const EdDescVector& c, const VVarDependencyGraph& graph)
{
	DendenciesVector v;

	for (auto e : c)
	{
		const auto& varToEq = graph[source(e, graph.getImpl())];
		v.emplace_back(varToEq.getVarToEq() * graph[e]);
	}

	return v;
}

static bool cycleHasIndentityDependency(
		const EdDescVector& c,
		const VVarDependencyGraph& graph,
		const DendenciesVector& dep)
{
	auto fin = std::accumulate(
			dep.begin() + 1, dep.end(), dep[0], [](const auto& l, const auto& r) {
				return l * r;
			});

	return fin.isIdentity();
}

static MultiDimInterval cyclicDependetSet(
		const EdDescVector& c,
		const VVarDependencyGraph& graph,
		const DendenciesVector& dep)
{
	const auto& firstEq = graph[source(c[0], graph.getImpl())];
	auto set = firstEq.getInterval();
	for (auto i : irange(c.size()))
	{
		const auto& edge = graph[c[i]];
		auto eq = graph[target(c[i], graph.getImpl())];

		set = intersection(dep[i].map(set), eq.getInterval());
	}

	return set;
}

static InexSetVector cyclicDependetSets(
		const EdDescVector& c, const VVarDependencyGraph& graph)
{
	auto dep = cycleToDependencieVector(c, graph);
	auto cyclicSet = cyclicDependetSet(c, graph, dep);
	InexSetVector v({ cyclicSet });
	if (!cycleHasIndentityDependency(c, graph, dep))
		return v;

	for (auto i : irange(c.size() - 1))
		v.emplace_back(dep[i].map(v.back()));

	for (auto i : irange(v.size()))
	{
		const auto& edge = graph[c[i]];
		const auto& eq = graph[source(c[i], graph.getImpl())];
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
		const std::vector<VVarDependencyGraph::VertexDesc>& cycle,
		const VVarDependencyGraph& g)
{
	auto c = cycleToEdgeVec(cycle, g);
	auto vecSet = cyclicDependetSets(c, g);
	if (vecSet[0].empty())
		return false;

	// for each equation in the cycle
	for (auto i : irange(cycle.size()))
	{
		const auto& eq = g[boost::source(c[i], g.getImpl())].getEquation();
		// copy the equation
		auto toFuseEq =
				eq.clone(eq.getTemplate()->getName() + "merged" + to_string(a++));
		// set induction to those that generate the circular dependency
		assert(toFuseEq.getInductions().contains(vecSet[i]));
		toFuseEq.setInductionVars(vecSet[i]);
		if (auto error = toFuseEq.explicitate(); error)
			return move(error);
		// add it to the list of filterd with normalized body
		filtered.emplace_back(toFuseEq.normalizeMatched());

		// then for all other index set that
		// are not in the circular set
		auto nonUsed = remove(eq.getInductions(), vecSet[i]);
		for (auto set : nonUsed)
		{
			// add the equation to the untouched set
			untouched.emplace_back(eq);
			// and set the inductions to the ones  that have no circular dependencies
			untouched.back().setInductionVars(set);
		}
	}

	// for all equations that were not in the circluar set, add it to the
	// untouched set.
	for (auto i : irange(source.size()))
	{
		if (find(cycle, i) == cycle.end())
			untouched.emplace_back(std::move(source[i]));
	}
	return true;
}

class CycleFuser
{
	public:
	CycleFuser(
			bool& f,
			EqVector& equs,
			const Model& model,
			const VVarDependencyGraph& graph,
			llvm::Error* e)
			: foundOne(&f), equs(&equs), model(&model), graph(&graph), error(e)
	{
	}

	template<typename Graph>
	void cycle(
			const std::vector<VVarDependencyGraph::VertexDesc>& cycle, const Graph&)
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

		auto e = linearySolve(filtered, *model);
		if (e)
		{
			*error = move(e);
			*foundOne = true;
			return;
		}

		for (auto& eq : filtered)
			newEqus.emplace_back(std::move(eq));

		*foundOne = true;
		*equs = std::move(newEqus);
	}

	private:
	bool* foundOne;
	EqVector* equs;
	const Model* model;
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
				CycleFuser(atLeastOneCollapse, equs, sourceModel, vectorGraph, &e));
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

	for (const auto& eq : scc.range(vectorGraph))
		out.push_back(eq.getEquation());

	if (auto error = fuseEquations(out, vectorGraph.getModel(), maxIterations);
			error)
		return error;

	return Error::success();
}

Expected<Model> modelica::solveScc(Model&& model, size_t maxIterations)
{
	VVarDependencyGraph vectorGraph(model);
	SccLookup sccs(vectorGraph);

	SmallVector<EqVector, 3> possibleEq(sccs.count());

	for (auto i : irange(sccs.count()))
		if (auto error =
						fuseScc(sccs[i], vectorGraph, possibleEq[i], maxIterations);
				error)
			return move(error);

	Model outModel({}, std::move(model.getVars()));
	for (auto& eqList : possibleEq)
		for (auto& eq : eqList)
			outModel.addEquation(std::move(eq));

	VVarDependencyGraph g(outModel);
	g.dump(llvm::errs());

	outModel.dump();

	return outModel;
}
