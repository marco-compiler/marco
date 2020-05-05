#include "modelica/matching/Matching.hpp"

#include <functional>
#include <iterator>
#include <string>
#include <type_traits>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/Edge.hpp"
#include "modelica/matching/Flow.hpp"
#include "modelica/matching/MatchingErrors.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModMatchers.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IRange.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

void MatchingGraph::addEquation(const ModEquation& eq)
{
	ReferenceMatcher matcher(eq);
	for (size_t useIndex : irange(matcher.size()))
		emplaceEdge(eq, move(matcher[useIndex]), useIndex);
}

void MatchingGraph::emplaceEdge(
		const ModEquation& eq, ModExpPath path, size_t useIndex)
{
	if (!VectorAccess::isCanonical(path.getExp()))
		return;

	auto access = AccessToVar::fromExp(path.getExp());
	const auto& var = model.getVar(access.getVarName());

	if (var.isState() || var.isConstant())
		return;

	if (access.getAccess().mappableDimensions() < eq.dimensions())
		return;

	auto eqDesc = getDesc(eq);
	auto varDesc = getDesc(var);

	Edge e(eq, var, move(access.getAccess()), move(path), useIndex);
	boost::add_edge(eqDesc, varDesc, std::move(e), graph);
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

size_t MatchingGraph::matchedEdgesCount() const
{
	return count_if(
			begin(), end(), [](const Edge& edge) { return !edge.empty(); });
}

static void dumpArcs(
		raw_ostream& OS,
		bool displayEmptyEdges,
		bool displayMappings,
		bool showMatchedCount,
		const Edge& edge,
		size_t equationIndex)
{
	if (!displayEmptyEdges && edge.empty())
		return;

	OS << "Eq_" << equationIndex << " -> " << edge.getVariable().getName();
	OS << "[";
	if (displayMappings)
	{
		OS << "headlabel=\"";
		edge.getInvertedAccess().dump(OS);
		OS << '\"';
	}

	OS << " label=\"";
	if (showMatchedCount)
		OS << edge.getSet().size();
	else
		edge.getSet().dump(OS);
	OS << "\"";

	if (displayMappings)
	{
		OS << " taillabel=\"";
		edge.getVectorAccess().dump(OS);
		OS << "\"";
	}
	OS << "];\n";
}

void MatchingGraph::dumpGraph(
		raw_ostream& OS,
		bool displayEmptyEdges,
		bool displayMappings,
		bool displayOnlyMatchedCount,
		bool closeGraph) const
{
	OS << "digraph {\n";

	size_t equationIndex = 0;
	for (const ModEquation& eq : model)
	{
		for (const Edge& edge : arcsOf(eq))
			dumpArcs(
					OS,
					displayEmptyEdges,
					displayMappings,
					displayOnlyMatchedCount,
					edge,
					equationIndex);
		equationIndex++;
	}

	equationIndex = 0;
	for (const ModEquation& eq : model)
	{
		OS << "Eq_" << equationIndex
			 << "[color=\"blue\" label=\"EQ: " << equationIndex << '\n';
		if (displayOnlyMatchedCount)
			OS << "matched: " << getMatchedSet(eq).size() << "/"
				 << eq.getInductions().size();
		else
			eq.dumpInductions(OS);
		OS << "\"];\n";

		equationIndex++;
	}

	for (const auto& var : variableLookUp)
	{
		OS << var.first->getName() << "[color=\"red\"";

		OS << " label=\"";
		OS << var.first->getName();
		OS << "\nmatched: ";
		OS << getMatchedSet(*var.first).size() << "/"
			 << var.first->toIndexSet().size();
		OS << "\"";

		OS << "];\n";
	}

	if (closeGraph)
		OS << "}\n";
}

void MatchingGraph::dump(llvm::raw_ostream& OS) const
{
	for (const auto& edge : *this)
		edge.dump(OS);
}

static Error insertEq(
		Edge& edge, const MultiDimInterval& inductionVars, EntryModel& outModel)
{
	const auto& eq = edge.getEquation();
	const auto& templ = eq.getTemplate();
	auto newName =
			templ->getName() + "m" + std::to_string(outModel.getTemplates().size());
	outModel.addEquation(eq.clone(std::move(newName)));
	auto& justInserted = outModel.getEquations().back();
	justInserted.setInductionVars(inductionVars);
	return justInserted.explicitate(edge.getPath());
}

static Error insertAllEq(Edge& edge, EntryModel& outModel)
{
	for (const auto& inductionVars : edge.getSet())
	{
		auto error = insertEq(edge, inductionVars, outModel);
		if (error)
			return error;
	}
	return Error::success();
}

static Expected<EntryModel> explicitateModel(
		EntryModel& model, MatchingGraph& graph)
{
	EntryModel toReturn({}, move(model.getVars()));

	for (auto& edge : graph)
	{
		if (edge.empty())
			continue;

		auto error = insertAllEq(edge, toReturn);
		if (error)
			return move(error);
	}

	return toReturn;
}

Expected<EntryModel> modelica::match(
		EntryModel entryModel, size_t maxIterations)
{
	if (entryModel.equationsCount() != entryModel.nonStateNonConstCount())
		return make_error<EquationAndStateMissmatch>(move(entryModel));
	MatchingGraph graph(entryModel);
	graph.match(maxIterations);

	if (graph.matchedCount() != entryModel.equationsCount())
		return make_error<FailedMatching>(move(entryModel), graph.matchedCount());

	return explicitateModel(entryModel, graph);
}

size_t MatchingGraph::indexOfEquation(const ModEquation& eq) const
{
	auto edgeIterator = find_if(
			model.getEquations(), [&](const auto& edge) { return &edge == &eq; });
	return distance(model.getEquations().begin(), edgeIterator);
}
