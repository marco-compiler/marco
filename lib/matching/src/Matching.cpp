#include "modelica/matching/Matching.hpp"

#include <functional>
#include <type_traits>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/Edge.hpp"
#include "modelica/matching/Flow.hpp"
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
	{
		const auto& use = matcher[useIndex];
		if (!VectorAccess::isCanonical(use.getExp()))
			continue;

		size_t edgeIndex = edges.size();
		edges.emplace_back(model, eq, use.getExp(), useIndex);
		equationLookUp.insert({ &eq, edgeIndex });
		auto var = &(edges.back().getVariable());
		variableLookUp.insert({ var, edgeIndex });
	}
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

void MatchingGraph::dumpGraph(raw_ostream& OS) const
{
	OS << "digraph {\n";

	int equationIndex = 1;
	for (const ModEquation& eq : model)
	{
		for (const Edge& edge : arcsOf(eq))
		{
			OS << "Eq_" << equationIndex << " -> " << edge.getVariable().getName();
			OS << "[label=\"";
			edge.getVectorAccess().dump(OS);
			OS << " ";
			edge.getSet().dump(OS);
			OS << " ";
			edge.getInvertedAccess().dump(OS);
			OS << "\"]"
				 << ";\n";
		}

		equationIndex++;
	}

	OS << "}\n";
}

void MatchingGraph::dump(llvm::raw_ostream& OS) const
{
	for (const auto& edge : *this)
		edge.dump(OS);
}
