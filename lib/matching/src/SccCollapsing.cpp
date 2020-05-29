#include "modelica/matching/SccCollapsing.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <vector>

#include "modelica/matching/SccLookup.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/Model.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

using EqVector = SmallVector<ModEquation, 3>;

static Error fuseInnermostScc(
		const Scc<VVarDependencyGraph>& scc,
		const VVarDependencyGraph& vectorGraph,
		EqVector& out)
{
}

static Error fuseEquations(EqVector& equs, const Model& sourceModel)
{
	while (true)
	{
		VVarDependencyGraph vectorGraph(sourceModel, equs);
		SccLookup sccs(vectorGraph);

		for (const auto& scc : sccs)
			if (auto error = fuseInnermostScc(scc, vectorGraph, equs); error)
				return error;
	}
}

static Error fuseScc(
		const Scc<VVarDependencyGraph>& scc,
		const VVarDependencyGraph& vectorGraph,
		EqVector& out)
{
	out.reserve(scc.size());

	for (const auto& eq : scc.range(vectorGraph))
		out.push_back(eq.getEquation());

	if (auto error = fuseEquations(out, vectorGraph.getModel()); error)
		return error;

	return Error::success();
}

Expected<Model> solveScc(Model&& model)
{
	VVarDependencyGraph vectorGraph(model);
	SccLookup sccs(vectorGraph);

	SmallVector<EqVector, 3> possibleEq(sccs.count());

	for (auto i : irange(sccs.count()))
		if (auto error = fuseScc(sccs[i], vectorGraph, possibleEq[i]); error)
			return move(error);

	Model outModel({}, move(model.getVars()));
	for (auto& eqList : possibleEq)
		for (auto& eq : eqList)
			outModel.addEquation(move(eq));

	return outModel;
}
