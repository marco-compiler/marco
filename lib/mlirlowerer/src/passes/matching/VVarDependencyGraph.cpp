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
#include <marco/mlirlowerer/passes/matching/MatchedEquationLookup.h>
#include <marco/mlirlowerer/passes/matching/SCCLookup.h>
#include <marco/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <marco/utils/IRange.hpp>
#include <marco/utils/IndexSet.hpp>
#include <marco/utils/Interval.hpp>
#include <sstream>
#include <utility>

using namespace marco::codegen::model;
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

void VVarDependencyGraph::populateEdge(
		const IndexesOfEquation* content,
		const AccessToVar& toVariable,
		EqToVert& eqToVert)
{
	const Variable& variable = model.getVariable(toVariable.getVar());
	
	for (const Equation& equation : content->getEquations())
	{
		const MultiDimInterval usedIndexes = toVariable.getAccess().map(equation.getInductions());

		for (const IndexesOfEquation* var : lookUp.eqsDeterminingVar(variable))
		{
			for (const MultiDimInterval& setOfVar : var->getIntervals())
			{
				if (areDisjoint(usedIndexes, setOfVar))
					continue;

				add_edge(
						eqToVert[&content->getContent()],
						eqToVert[&var->getContent()],
						toVariable.getAccess(),
						graph);
			}
		}
	}
}

void VVarDependencyGraph::populateEq(const IndexesOfEquation* content, EqToVert& eqToVert)
{
	ReferenceMatcher rightHandMatcher;

	// Search for dependencies among the non-matched variables
	rightHandMatcher.visit(content->getContent(), true);

	for (const ExpressionPath& toVariable : rightHandMatcher)
	{
		assert(VectorAccess::isCanonical(toVariable.getExpression()));
		AccessToVar toAccess = AccessToVar::fromExp(toVariable.getExpression());
		
		// Add an edge for each dependency
		populateEdge(content, toAccess, eqToVert);
	}
}

void VVarDependencyGraph::create()
{
	EqToVert eqToVert;

	// Add all nodes
	for (const IndexesOfEquation* content : lookUp)
	{
		// Do not add duplicate IndexesOfEquation to the map.
		// This could be caused by multiple equations in a BltBlock.
		if (eqToVert.find(&content->getContent()) == eqToVert.end())
			eqToVert[&content->getContent()] = add_vertex(content, graph);
	}

	// Add all edges
	for (const IndexesOfEquation* content : lookUp)
		populateEq(content, eqToVert);
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
