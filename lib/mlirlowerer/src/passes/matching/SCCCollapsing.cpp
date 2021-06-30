#include <algorithm>
#include <boost/graph/lookup_edge.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <marco/mlirlowerer/passes/matching/LinSolver.h>
#include <marco/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <marco/mlirlowerer/passes/matching/SCCLookup.h>
#include <marco/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <marco/utils/Interval.hpp>
#include <marco/utils/IRange.hpp>
#include <numeric>
#include <vector>

void boost::throw_exception(std::exception const & e) {
	llvm::errs() << e.what();
	abort();
}

#if BOOST_VERSION >= 107300 /* source_location only exists in boost >= 1.73 */

void boost::throw_exception(const std::exception& e, const boost::source_location& loc)
{
	llvm::errs() << e.what();
	abort();
}

#endif


namespace marco::codegen::model
{
	template<typename Graph>
	void renumber_vertex_indices(const Graph& graph)
	{
		assert(
				false && "YOU CANNOT USE THIS ALGORITHM BECAUSE RENUMBER_VERTEX_INDICES "
								 "IS NOT SUPPORTED");
	}
}

using namespace marco::codegen::model;

using EquationsVector = llvm::SmallVector<Equation, 3>;

using EdDescVector = llvm::SmallVector<VVarDependencyGraph::EdgeDesc, 3>;
using DependenciesVector = llvm::SmallVector<VectorAccess, 3>;
using IndexSetVector = llvm::SmallVector<marco::MultiDimInterval, 3>;

static EdDescVector cycleToEdgeVec(
		std::vector<VVarDependencyGraph::VertexDesc> c, const VVarDependencyGraph& graph)
{
	EdDescVector v;
	for (size_t a : marco::irange(c.size()))
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
	VectorAccess fin = std::accumulate(
			dep.begin() + 1, dep.end(), dep[0], [](const auto& l, const auto& r) {
				return l * r;
			});

	return fin.isIdentity();
}

static marco::MultiDimInterval cyclicDependentSet(
		const EdDescVector& c,
		const VVarDependencyGraph& graph,
		const DependenciesVector& dep)
{
	const IndexesOfEquation& firstEq = graph[source(c[0], graph.getImpl())];
	marco::MultiDimInterval set = firstEq.getInterval();
	for (size_t i : marco::irange(c.size()))
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
	modelica::MultiDimInterval cyclicSet = cyclicDependentSet(c, graph, dep);
	IndexSetVector v({ cyclicSet });

	for (size_t i : marco::irange(c.size() - 1))
		v.emplace_back(dep[i].map(v.back()));

	assert(dep.size() == c.size() && v.size() == c.size());

	if (!cycleHasIndentityDependency(c, graph, dep))
		return v;

	for (size_t i : marco::irange(v.size()))
	{
		const IndexesOfEquation& eq = graph[source(c[i], graph.getImpl())];
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
	EdDescVector c = cycleToEdgeVec(cycle, g);
	IndexSetVector vecSet = cyclicDependentSets(c, g);

	if (vecSet[0].empty())
		return mlir::failure();

	// For each equation in the cycle
	for (size_t i : marco::irange(cycle.size()))
	{
		Equation original = g[boost::source(c[i], g.getImpl())].getEquation();

		// copy the equation
		Equation toFuseEq = original.clone();

		// set induction to those that generate the circular dependency
		assert(toFuseEq.getInductions().contains(vecSet[i]));

		toFuseEq.setInductions(vecSet[i]);

		if (auto res = toFuseEq.explicitate(); failed(res))
			return res;

		// add it to the list of filtered with normalized body it there is no loop
		if (auto res = toFuseEq.normalize(); failed(res))
			return res;

		filtered.emplace_back(toFuseEq);

		// then for all other index set that
		// are not in the circular set
		modelica::IndexSet nonUsed = remove(original.getInductions(), vecSet[i]);

		for (modelica::MultiDimInterval set : nonUsed)
		{
			// add the equation to the untouched set
			untouched.emplace_back(original.clone());
			// and set the inductions to the ones  that have no circular dependencies
			untouched.back().setInductions(set);
		}

		original.getOp()->erase();
	}

	// for all equations that were not in the circular set, add it to the
	// untouched set.
	for (size_t i : marco::irange(source.size()))
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

		if (!canSolveSystem(*equations, graph->getModel()))
		{
			*status = mlir::failure();
			*foundOne = true;
			return;
		}

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

		for (Equation& eq : filtered)
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

	for (const IndexesOfEquation& eq : SCC.range(vectorGraph))
		out.push_back(eq.getEquation());

	if (auto res = fuseEquations(builder, out, vectorGraph.getModel(), maxIterations); failed(res))
		return res;

	return mlir::success();
}

namespace marco::codegen::model
{
	mlir::LogicalResult solveSCCs(Model& model, size_t maxIterations)
	{
		mlir::OpBuilder builder(model.getOp());
		VVarDependencyGraph vectorGraph(model);
		SCCLookup SCCs(vectorGraph);

		llvm::SmallVector<EquationsVector, 3> possibleEquations(SCCs.count());
		llvm::SmallVector<BltBlock, 3> algebraicLoops;

		for (size_t i : irange(SCCs.count()))
		{
			if (failed(fuseScc(builder, SCCs[i], vectorGraph, possibleEquations[i], maxIterations)))
			{
				// If the Scc Collapsing algorithm fails, it means that we have
				// an Algebraic Loop, which must be handled by a solver afterwards.
				llvm::SmallVector<Equation, 3> bltEquations;
				for (Equation& eq : possibleEquations[i])
					bltEquations.push_back(eq);

				algebraicLoops.push_back(BltBlock(bltEquations));

				possibleEquations[i].clear();
			}
		}

		llvm::SmallVector<Equation, 3> equations;
		auto* terminator = model.getOp().body().front().getTerminator();

		for (EquationsVector& equationsList : possibleEquations)
		{
			for (Equation& equation : equationsList)
			{
				//equation.getOp()->moveBefore(terminator);
				equations.push_back(equation);
			}
		}

		model = Model(model.getOp(), model.getVariables(), equations, algebraicLoops);
		return mlir::success();
	}
}
