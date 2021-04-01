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

void MatchingGraph::match(int iterations)
{
	for (auto _ : irange(0, iterations))
	{
		AugmentingPath path(*this);
		if (!path.valid())
			return;

		path.apply();
	}
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
	const auto iter = equationLookUp.find(&equation);
	assert(equationLookUp.end() != iter);
	auto [begin, end] = boost::out_edges(iter->second, graph);
	return llvm::make_range(out_iterator(*this, begin), out_iterator(*this, end));
}

llvm::iterator_range<MatchingGraph::const_out_iterator> MatchingGraph::arcsOf(const Equation& equation) const
{
	const auto iter = equationLookUp.find(&equation);
	assert(equationLookUp.end() != iter);
	auto [begin, end] = boost::out_edges(iter->second, graph);
	return llvm::make_range(const_out_iterator(*this, begin), const_out_iterator(*this, end));
}

llvm::iterator_range<MatchingGraph::out_iterator> MatchingGraph::arcsOf(const Variable& var)
{
	const auto iter = variableLookUp.find(&var);
	assert(variableLookUp.end() != iter);
	auto [begin, end] = boost::out_edges(iter->second, graph);
	return llvm::make_range(out_iterator(*this, begin), out_iterator(*this, end));
}

llvm::iterator_range<MatchingGraph::const_out_iterator> MatchingGraph::arcsOf(const Variable& var) const
{
	const auto iter = variableLookUp.find(&var);
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
	if (equationLookUp.find(&eq) != equationLookUp.end())
		return equationLookUp[&eq];

	auto dec = boost::add_vertex(graph);
	equationLookUp[&eq] = dec;
	return dec;
}

MatchingGraph::VertexDesc MatchingGraph::getDesc(const Variable& var)
{
	if (variableLookUp.find(&var) != variableLookUp.end())
		return variableLookUp[&var];

	auto dec = boost::add_vertex(graph);
	variableLookUp[&var] = dec;
	return dec;
}

void MatchingGraph::addEquation(const Equation& eq)
{
	ReferenceMatcher matcher(eq);

	for (size_t useIndex : irange(matcher.size()))
		emplaceEdge(eq, std::move(matcher[useIndex]), useIndex);
}

void MatchingGraph::emplaceEdge(const Equation& eq, ExpressionPath path, size_t useIndex)
{
	if (!VectorAccess::isCanonical(path.getExp()))
		return;

	auto access = AccessToVar::fromExp(path.getExp());
	const auto& var = model.getVariable(access.getVar());

	if (var.isState() || var.isConstant())
		return;

	if (access.getAccess().mappableDimensions() < eq.dimensions())
		return;

	auto eqDesc = getDesc(eq);
	auto varDesc = getDesc(var);

	Edge e(eq, var, std::move(access.getAccess()), std::move(path), useIndex);
	boost::add_edge(eqDesc, varDesc, std::move(e), graph);
}

static llvm::Error insertEq(Edge& edge, const modelica::MultiDimInterval& inductionVars, Model& outModel)
{
	const auto& eq = edge.getEquation();
	const auto& templ = eq.getTemplate();
	auto newName = templ->getName() + "m" + std::to_string(outModel.getTemplates().size());
	outModel.addEquation(eq.clone(std::move(newName)));
	auto& justInserted = outModel.getEquations().back();
	justInserted->setInductionVars(inductionVars);
	justInserted->setMatchedExp(edge.getPath().getEqPath());
	return llvm::Error::success();
}

static llvm::Error insertAllEq(Edge& edge, Model& outModel)
{
	for (const auto& inductionVars : edge.getSet())
	{
		auto error = insertEq(edge, inductionVars, outModel);
		if (error)
			return error;
	}
	return llvm::Error::success();
}

static llvm::Expected<Model> explicitateModel(Model& model, MatchingGraph& graph)
{
	Model toReturn(model.getVariables(), {});

	for (auto& edge : graph)
	{
		if (edge.empty())
			continue;

		auto error = insertAllEq(edge, toReturn);
		if (error)
			return std::move(error);
	}

	return toReturn;
}

llvm::Expected<Model> modelica::codegen::model::match(Model entryModel, size_t maxIterations)
{
	assert(entryModel.equationsCount() == entryModel.nonStateNonConstCount());

	MatchingGraph graph(entryModel);
	graph.match(maxIterations);

	assert(graph.matchedCount() == entryModel.equationsCount());
	return explicitateModel(entryModel, graph);
}
