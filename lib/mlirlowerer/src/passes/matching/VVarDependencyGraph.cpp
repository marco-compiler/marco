#include <algorithm>
#include <boost/graph/strong_components.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/range/iterator_range_core.hpp>

#if BOOST_VERSION >= 107300
#include <boost/assert/source_location.hpp>
#endif

#include <exception>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <map>
#include <modelica/mlirlowerer/passes/matching/MatchedEquationLookup.h>
#include <modelica/mlirlowerer/passes/matching/SCCLookup.h>
#include <modelica/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/utils/IRange.hpp>
#include <modelica/utils/IndexSet.hpp>
#include <modelica/utils/Interval.hpp>
#include <sstream>
#include <utility>

using namespace modelica::codegen::model;
using namespace std;
using namespace llvm;
using namespace boost;

void VVarDependencyGraph::dump() const
{
	dump(llvm::outs());
}

void VVarDependencyGraph::dump(llvm::raw_ostream& os) const
{
	os << "digraph G {";

	for (auto vertex : make_iterator_range(vertices(graph)))
		os << vertex << "[label=\"" << vertex << "\"]";

	for (auto edge : make_iterator_range(edges(graph)))
	{
		auto from = source(edge);
		auto to = target(edge);

		os << from << "->" << to << "[label=\"";
		graph[edge].dump(os);
		os << "\"];\n";
	}

	os << "}";
}

void VVarDependencyGraph::populateEdge(IndexesOfEquation& equation,
																			 const AccessToVar& toVariable,
																			 EqToVert& eqToVert)
{
	const auto& variable = model.getVariable(toVariable.getVar());
	const auto usedIndexes = toVariable.getAccess().map(equation.getEquation().getInductions());

	for (const auto& var : lookUp.eqsDeterminingVar(variable))
	{
		const auto& setOfVar = var.getInterval();

		if (areDisjoint(usedIndexes, setOfVar))
			continue;

		add_edge(
				eqToVert[equation.getEquation()],
				eqToVert[var.getEquation()],
				toVariable.getAccess(),
				graph);
	}
}

void VVarDependencyGraph::populateEq(IndexesOfEquation& equation,
																		 EqToVert& eqToVert)
{
	ReferenceMatcher rightHandMatcher;
	rightHandMatcher.visit(equation.getEquation(), true);

	for (const auto& toVariable : rightHandMatcher)
	{
		assert(VectorAccess::isCanonical(toVariable.getExp()));
		auto toAccess = AccessToVar::fromExp(toVariable.getExp());
		populateEdge(equation, toAccess, eqToVert);
	}
}

void VVarDependencyGraph::create()
{
	EqToVert eqToVert;

	for (const auto& eq : lookUp)
		eqToVert[eq.getEquation()] = add_vertex(&eq, graph);

	for (auto& eq : lookUp)
		populateEq(eq, eqToVert);
}

VVarDependencyGraph::VVarDependencyGraph(const Model& model, ArrayRef<Equation> equations)
		: model(model), lookUp(model, equations)
{
	create();
}

VVarDependencyGraph::VVarDependencyGraph(const Model& model): model(model), lookUp(model)
{
	create();
}
