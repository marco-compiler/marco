#include <algorithm>
#include <boost/graph/lookup_edge.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <modelica/mlirlowerer/passes/matching/LinSolver.h>
#include <modelica/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <modelica/mlirlowerer/passes/matching/SCCLookup.h>
#include <modelica/mlirlowerer/passes/matching/VVarDependencyGraph.h>
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

static DendenciesVector cycleToDependencyVector(
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
	auto dep = cycleToDependencyVector(c, graph);
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

	// For each equation in the cycle
	for (auto i : modelica::irange(cycle.size()))
	{
		auto original = g[boost::source(c[i], g.getImpl())].getEquation();

		// copy the equation
		auto toFuseEq = original.clone();

		// set induction to those that generate the circular dependency
		assert(toFuseEq.getInductions().contains(vecSet[i]));
		toFuseEq.setInductionVars(vecSet[i]);

		if (auto res = toFuseEq.explicitate(); failed(res))
			return res;

		// add it to the list of filtered with normalized body
		if (auto res = toFuseEq.normalize(); failed(res))
			return res;

		filtered.emplace_back(toFuseEq);

		// then for all other index set that
		// are not in the circular set
		auto nonUsed = remove(original.getInductions(), vecSet[i]);

		for (auto set : nonUsed)
		{
			// add the equation to the untouched set
			untouched.emplace_back(original.clone());
			// and set the inductions to the ones  that have no circular dependencies
			untouched.back().setInductionVars(set);
		}

		original.getOp()->erase();
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
	CycleFuser(mlir::OpBuilder& builder,
						 std::shared_ptr<bool> f,
						 EquationsVector& equations,
						 const VVarDependencyGraph& graph,
						 mlir::LogicalResult* status)
			: builder(&builder),
				foundOne(std::move(f)),
				equations(&equations),
				graph(&graph),
				status(status)
	{
	}

	template<typename Graph>
	void cycle(const std::vector<VVarDependencyGraph::VertexDesc>& cycle, const Graph& g)
	{
		if (*foundOne)
			return;

		EquationsVector newEquations;
		EquationsVector filtered;

		if (auto err = extractEquationWithDependencies(*equations, filtered, newEquations, cycle, *graph); failed(err))
		{
			*status = err;
			*foundOne = true;
			return;
		}

		if (auto err = linearySolve(*builder, filtered); failed(err))
		{
			*status = err;
			*foundOne = true;
			return;
		}

		for (auto& eq : filtered)
			newEquations.emplace_back(eq);

		*foundOne = true;
		*equations = std::move(newEquations);
	}

	private:
	mlir::OpBuilder* builder;
	std::shared_ptr<bool> foundOne;
	EquationsVector* equations;
	const VVarDependencyGraph* graph;
	mlir::LogicalResult* status;
};

static mlir::LogicalResult fuseEquations(mlir::OpBuilder& builder, EquationsVector& equations, const Model& model, size_t maxIterations)
{
	auto atLeastOneCollapse = std::make_shared<bool>(false);
	size_t currentIteration = 0;

	do
	{
		*atLeastOneCollapse = false;
		VVarDependencyGraph vectorGraph(model, equations);
		mlir::LogicalResult status = mlir::success();

		tiernan_all_cycles(
				vectorGraph.getImpl(),
				CycleFuser(builder, atLeastOneCollapse, equations, vectorGraph, &status));

		if (failed(status))
			return status;

		if (++currentIteration == maxIterations)
			return mlir::failure();

	} while (*atLeastOneCollapse);

	return mlir::success();
}

static mlir::LogicalResult fuseScc(
		mlir::OpBuilder& builder,
		const SCC<VVarDependencyGraph>& SCC,
		const VVarDependencyGraph& vectorGraph,
		EquationsVector& out,
		size_t maxIterations)
{
	out.reserve(SCC.size());

	for (const auto& eq : SCC.range(vectorGraph))
		out.push_back(eq.getEquation());

	if (auto res = fuseEquations(builder, out, vectorGraph.getModel(), maxIterations); failed(res))
		return res;

	return mlir::success();
}

namespace modelica::codegen::model
{
	mlir::LogicalResult solveSCCs(mlir::OpBuilder& builder, Model& model, size_t maxIterations)
	{
		VVarDependencyGraph vectorGraph(model);
		SCCLookup SCCs(vectorGraph);

		llvm::SmallVector<EquationsVector, 3> possibleEquations(SCCs.count());

		for (size_t i = 0, e = SCCs.count(); i < e; ++i)
		{
			if (failed(fuseScc(builder, SCCs[i], vectorGraph, possibleEquations[i], maxIterations)))
				return mlir::failure();
		}

		llvm::SmallVector<Equation, 3> equations;

		for (auto& equationsList : possibleEquations)
			for (auto& equation : equationsList)
				equations.push_back(equation);

		model = Model(model.getOp(), model.getVariables(), equations);
		return mlir::success();
	}
}
