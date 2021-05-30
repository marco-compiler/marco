#include <functional>
#include <iterator>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/mlirlowerer/passes/matching/Edge.h>
#include <modelica/mlirlowerer/passes/matching/Flow.h>
#include <modelica/mlirlowerer/passes/matching/Matching.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <modelica/utils/IRange.hpp>
#include <string>
#include <type_traits>

using namespace modelica::codegen::model;

MatchingGraph::MatchingGraph(const Model& model) : model(model)
{
	for (const auto& equation : model)
		addEquation(equation);
}

const Model& MatchingGraph::getModel() const
{
	return model;
}

size_t MatchingGraph::variableCount() const
{
	return model.getVariables().size();
}

size_t MatchingGraph::equationCount() const
{
	return model.getEquations().size();
}

Edge& MatchingGraph::operator[](EdgeDesc desc)
{
	return graph[desc];
}

const Edge&MatchingGraph:: operator[](EdgeDesc desc) const
{
	return graph[desc];
}

MatchingGraph::edge_iterator MatchingGraph::begin()
{
	return edge_iterator(*this, boost::edges(graph).first);
}

MatchingGraph::const_edge_iterator MatchingGraph::begin() const
{
	return const_edge_iterator(*this, boost::edges(graph).first);
}

MatchingGraph::edge_iterator MatchingGraph::end()
{
	return edge_iterator(*this, boost::edges(graph).second);
}

MatchingGraph::const_edge_iterator MatchingGraph::end() const
{
	return const_edge_iterator(*this, boost::edges(graph).second);
}

mlir::LogicalResult MatchingGraph::match(unsigned int maxIterations)
{
	for (unsigned int i = 0; i < maxIterations; ++i)
	{
		AugmentingPath path(*this);

		if (!path.valid())
			return mlir::success();

		path.apply();
	}

	return mlir::failure();
}

size_t MatchingGraph::matchedCount() const
{
	size_t count = 0;

	for (const auto& edge : *this)
		count += edge.getSet().size();

	return count;
}

size_t MatchingGraph::edgesCount() const
{
	auto [b, e] = boost::edges(graph);
	return std::distance(b, e);
}

size_t MatchingGraph::matchedEdgesCount() const
{
	return count_if(
			begin(), end(), [](const Edge& edge) { return !edge.empty(); });
}

llvm::iterator_range<MatchingGraph::out_iterator> MatchingGraph::arcsOf(const Equation& equation)
{
	const auto iter = equationLookUp.find(equation);
	assert(equationLookUp.end() != iter);
	auto [begin, end] = boost::out_edges(iter->second, graph);
	return llvm::make_range(out_iterator(*this, begin), out_iterator(*this, end));
}

llvm::iterator_range<MatchingGraph::const_out_iterator> MatchingGraph::arcsOf(const Equation& equation) const
{
	const auto iter = equationLookUp.find(equation);

	llvm::errs() << "---------DUMP\n";
	for (const auto& eq : equationLookUp)
		eq.first.getOp()->dump();

	assert(equationLookUp.end() != iter);
	auto [begin, end] = boost::out_edges(iter->second, graph);
	return llvm::make_range(const_out_iterator(*this, begin), const_out_iterator(*this, end));
}

llvm::iterator_range<MatchingGraph::out_iterator> MatchingGraph::arcsOf(const Variable& var)
{
	const auto iter = variableLookUp.find(var);
	assert(variableLookUp.end() != iter);
	auto [begin, end] = boost::out_edges(iter->second, graph);
	return llvm::make_range(out_iterator(*this, begin), out_iterator(*this, end));
}

llvm::iterator_range<MatchingGraph::const_out_iterator> MatchingGraph::arcsOf(const Variable& var) const
{
	const auto iter = variableLookUp.find(var);
	assert(variableLookUp.end() != iter);
	auto [begin, end] = boost::out_edges(iter->second, graph);
	return llvm::make_range(const_out_iterator(*this, begin), const_out_iterator(*this, end));
}

modelica::IndexSet MatchingGraph::getUnmatchedSet(const Variable& variable) const
{
	auto set = variable.toIndexSet();
	set.remove(getMatchedSet(variable));
	return set;
}

modelica::IndexSet MatchingGraph::getUnmatchedSet(const Equation& equation) const
{
	IndexSet set(equation.getInductions());
	set.remove(getMatchedSet(equation));
	return set;
}

modelica::IndexSet MatchingGraph::getMatchedSet(const Variable& variable) const
{
	IndexSet matched;

	for (const Edge& edge : arcsOf(variable))
		matched.unite(edge.getVectorAccess().map(edge.getSet()));

	return matched;
}

modelica::IndexSet MatchingGraph::getMatchedSet(const Equation& eq) const
{
	IndexSet matched;
	for (const Edge& edge : arcsOf(eq))
		matched.unite(edge.getSet());

	return matched;
}

MatchingGraph::VertexDesc MatchingGraph::getDesc(const Equation& eq)
{
	if (equationLookUp.find(eq) != equationLookUp.end())
		return equationLookUp[eq];

	auto dec = boost::add_vertex(graph);
	equationLookUp[eq] = dec;
	return dec;
}

MatchingGraph::VertexDesc MatchingGraph::getDesc(const Variable& var)
{
	if (variableLookUp.find(var) != variableLookUp.end())
		return variableLookUp[var];

	auto dec = boost::add_vertex(graph);
	variableLookUp[var] = dec;
	return dec;
}

void MatchingGraph::addEquation(Equation eq)
{
	ReferenceMatcher matcher(eq);

	for (size_t useIndex : irange(matcher.size()))
		emplaceEdge(eq, std::move(matcher[useIndex]), useIndex);
}

void MatchingGraph::emplaceEdge(Equation eq, ExpressionPath path, size_t useIndex)
{
	eq.getOp().dump();

	if (!VectorAccess::isCanonical(path.getExpression()))
		return;

	auto access = AccessToVar::fromExp(path.getExpression());
	const auto& var = model.getVariable(access.getVar());

	if (var.isState() || var.isConstant())
		return;

	auto dim1 = access.getAccess().mappableDimensions();
	auto dim2 = eq.dimensions();
	if (access.getAccess().mappableDimensions() < eq.dimensions())
		return;

	auto eqDesc = getDesc(eq);
	auto varDesc = getDesc(var);

	Edge e(eq, var, std::move(access.getAccess()), std::move(path), useIndex);
	boost::add_edge(eqDesc, varDesc, std::move(e), graph);
}

mlir::LogicalResult modelica::codegen::model::match(Model& model, size_t maxIterations)
{
	for (const auto& var : model.getVariables())
		var->getReference().dump();

	if (model.equationsCount() != model.nonStateNonConstCount())
		return model.getOp()->emitError("Equations amount (" + std::to_string(model.equationsCount()) + ") doesn't match the non state + non const variables amount (" + std::to_string(model.nonStateNonConstCount()) + ")");

	MatchingGraph graph(model);

	if (failed(graph.match(maxIterations)))
		return model.getOp()->emitError("Max iterations amount has been reached");

	if (graph.matchedCount() != model.equationsCount())
		return model.getOp()->emitError("Not all the equations have been matched");

	llvm::SmallVector<Equation, 3> equations;

	for (auto& edge : graph)
	{
		if (edge.empty())
			continue;

		for (const auto& inductionVars : edge.getSet())
		{
			// It is sufficient to make a copy of the equation descriptor, as we
			// don't need to clone the whole equation IR.
			auto equation = edge.getEquation();

			equation.setInductions(inductionVars);
			equation.setMatchedExp(edge.getPath().getEquationPath());
			equations.push_back(equation);
		}
	}

	Model result(model.getOp(), model.getVariables(), equations);

	model = result;
	return mlir::success();
}
