#include <algorithm>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <modelica/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <modelica/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <modelica/mlirlowerer/passes/model/LinSolver.h>
#include <modelica/utils/Interval.hpp>
#include <modelica/utils/IRange.hpp>
#include <numeric>
#include <vector>

namespace modelica::codegen::model
{
	template<typename Graph>
	void renumber_vertex_indices(const Graph& graph)
	{
		assert(
				false && "YOU CANNOT USE THIS ALGORITHM BECAUSE RENUMBER_VERTEX_INDICES "
								 "IS NOT SUPPORTED");
	}
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

using namespace modelica::codegen::model;

using EquationsVector = llvm::SmallVector<Equation, 3>;

using EdDescVector = llvm::SmallVector<VVarDependencyGraph::EdgeDesc, 3>;
using DendenciesVector = llvm::SmallVector<VectorAccess, 3>;
using InexSetVector = llvm::SmallVector<modelica::MultiDimInterval, 3>;

static EdDescVector cycleToEdgeVec(
		std::vector<VVarDependencyGraph::VertexDesc> c, const VVarDependencyGraph& graph)
{
	EdDescVector v;
	for (auto a : modelica::irange(c.size()))
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

static modelica::MultiDimInterval cyclicDependetSet(
		const EdDescVector& c,
		const VVarDependencyGraph& graph,
		const DendenciesVector& dep)
{
	const auto& firstEq = graph[source(c[0], graph.getImpl())];
	auto set = firstEq.getInterval();
	for (auto i : modelica::irange(c.size()))
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

	for (auto i : modelica::irange(c.size() - 1))
		v.emplace_back(dep[i].map(v.back()));

	for (auto i : modelica::irange(v.size()))
	{
		const auto& edge = graph[c[i]];
		const auto& eq = graph[source(c[i], graph.getImpl())];
		v[i] = eq.getVarToEq().map(v[i]);
		assert(eq.getEquation().getInductions().contains(v[i]));
	}

	return v;
}

static mlir::LogicalResult extractEquationWithDependencies(
		EquationsVector& source,
		EquationsVector& filtered,
		EquationsVector& untouched,
		const std::vector<VVarDependencyGraph::VertexDesc>& cycle,
		const VVarDependencyGraph& g)
{
	auto c = cycleToEdgeVec(cycle, g);
	auto vecSet = cyclicDependetSets(c, g);

	if (vecSet[0].empty())
		return mlir::failure();

	// for each equation in the cycle
	for (auto i : modelica::irange(cycle.size()))
	{
		const auto& eq = g[boost::source(c[i], g.getImpl())].getEquation();

		// copy the equation
		auto toFuseEq = eq.clone();

		// set induction to those that generate the circular dependency
		assert(toFuseEq.getInductions().contains(vecSet[i]));
		toFuseEq.setInductionVars(vecSet[i]);

		if (auto res = toFuseEq.explicitate(); failed(res))
			return res;

		// add it to the list of filtered with normalized body
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

	// for all equations that were not in the circular set, add it to the
	// untouched set.
	for (auto i : modelica::irange(source.size()))
	{
		if (llvm::find(cycle, i) == cycle.end())
			untouched.emplace_back(std::move(source[i]));
	}

	return mlir::success();
}

class CycleFuser
{
	public:
	CycleFuser(
			bool& f,
			EquationsVector& equations,
			const Model& model,
			const VVarDependencyGraph& graph,
			mlir::LogicalResult* e)
			: foundOne(&f), equations(&equations), model(&model), graph(&graph), error(e)
	{
	}

	template<typename Graph>
	void cycle(const std::vector<VVarDependencyGraph::VertexDesc>& cycle, const Graph&)
	{
		if (*foundOne)
			return;

		EquationsVector newEqus;
		EquationsVector filtered;

		auto err = extractEquationWithDependencies(*equations, filtered, newEqus, cycle, *graph);

		if (succeeded(err))
		{
			*error = err;
			*foundOne = true;
			return;
		}

		if (succeeded(err))
			return;

		auto e = linearySolve(filtered, *model);

		if (e)
		{
			//*error = std::move(e);
			*foundOne = true;
			return;
		}

		for (auto& eq : filtered)
			newEqus.emplace_back(std::move(eq));

		*foundOne = true;
		*equations = std::move(newEqus);
	}

	private:
	bool* foundOne;
	EquationsVector* equations;
	const Model* model;
	const VVarDependencyGraph* graph;
	mlir::LogicalResult* error;
};

static mlir::LogicalResult fuseEquations(
		EquationsVector& equations, const Model& sourceModel, size_t maxIterations)
{
	bool atLeastOneCollapse = false;
	size_t currIterations = 0;

	do
	{
		atLeastOneCollapse = false;
		VVarDependencyGraph vectorGraph(sourceModel, equations);
		mlir::LogicalResult e = mlir::success();

		tiernan_all_cycles(
				vectorGraph.getImpl(),
				CycleFuser(atLeastOneCollapse, equations, sourceModel, vectorGraph, &e));

		if (failed(e))
			return e;

		if (++currIterations == maxIterations)
			return mlir::failure();

	} while (atLeastOneCollapse);

	return mlir::success();
}

static mlir::LogicalResult fuseScc(
		const modelica::Scc<VVarDependencyGraph>& SCC,
		const VVarDependencyGraph& vectorGraph,
		EquationsVector& out,
		size_t maxIterations)
{
	out.reserve(SCC.size());

	for (const auto& eq : SCC.range(vectorGraph))
		out.push_back(eq.getEquation());

	if (auto res = fuseEquations(out, vectorGraph.getModel(), maxIterations); failed(res))
		return res;

	return mlir::success();
}

mlir::LogicalResult modelica::codegen::model::solveSCC(Model& model, size_t maxIterations)
{
	VVarDependencyGraph vectorGraph(model);
	SccLookup SCCs(vectorGraph);

	llvm::SmallVector<EquationsVector, 3> possibleEquations(SCCs.count());

	for (size_t i = 0, e = SCCs.count(); i < e; ++i)
	{
		if (failed(fuseScc(SCCs[i], vectorGraph, possibleEquations[i], maxIterations)))
			return mlir::failure();
	}

	Model result(model.getOp(), model.getVariables(), {});

	for (auto& equationsList : possibleEquations)
		for (auto& equation : equationsList)
			result.addEquation(equation);

	return mlir::success();
}
