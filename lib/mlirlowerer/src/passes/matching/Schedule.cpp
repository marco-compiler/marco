#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <marco/mlirlowerer/passes/matching/SCCDependencyGraph.h>
#include <marco/mlirlowerer/passes/matching/SCCLookup.h>
#include <marco/mlirlowerer/passes/matching/SVarDependencyGraph.h>
#include <marco/mlirlowerer/passes/matching/Schedule.h>
#include <marco/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <marco/mlirlowerer/passes/model/BltBlock.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/utils/IndexSet.hpp>

using namespace marco::codegen::model;

static llvm::SmallVector<std::variant<Equation, BltBlock>, 3> collapseEquations(
		const SVarDependencyGraph& originalGraph)
{
	llvm::SmallVector<std::variant<Equation, BltBlock>, 3> out;

	marco::IndexSet currentSet;

	const auto onSched = [&](size_t node) {
		const auto& currentNode = originalGraph[node];
		currentSet.insert(currentNode.getIndexes());
	};

	const auto onGroupEnd = [&](size_t node, khanNextPreferred schedulingDirection) {
		assert(schedulingDirection != khanNextPreferred::cannotBeOptimized);

		const bool backward = schedulingDirection == khanNextPreferred::backwardPreferred;
		const auto& currentNode = originalGraph[node];
		const auto& content = currentNode.getCollapsedVertex();

		if (std::holds_alternative<Equation>(content))
		{
			const Equation& eq = std::get<Equation>(content);
			for (const marco::MultiDimInterval& set : currentSet)
			{
				Equation clone = eq.clone();
				clone.setForward(!backward);
				clone.setInductions(set);
				out.emplace_back(clone);
			}
			eq.getOp()->erase();
		}
		else
		{
			BltBlock bltBlock = std::get<BltBlock>(content);
			bltBlock.setForward(!backward);
			out.emplace_back(bltBlock);
		}

		currentSet = marco::IndexSet();
	};

	originalGraph.topoOrder(onSched, onGroupEnd);

	return out;
}

static bool isForward(const VectorAccess* access)
{
	return llvm::none_of(*access, [](const SingleDimensionAccess& varAccess) {
		return varAccess.isOffset() && varAccess.getOffset() >= 0;
	});
}

static bool isBackward(const VectorAccess* access)
{
	return llvm::none_of(*access, [](const SingleDimensionAccess& varAccess) {
		return varAccess.isOffset() && varAccess.getOffset() <= 0;
	});
}

static llvm::SmallVector<std::variant<Equation, BltBlock>, 3> trivialScheduling(
		const SCC<VVarDependencyGraph>& scc,
		const VVarDependencyGraph& originalGraph)
{
	if (scc.size() != 1)
		return {};

	llvm::SmallVector<const VectorAccess*, 3> internalEdges;

	auto content = originalGraph[scc[0]].getContent();

	// If it is an unsolvable algebraic loop, return it.
	if (std::holds_alternative<BltBlock>(content))
		return { content };

	for (const auto& edge : originalGraph.outEdges(scc[0]))
		if (originalGraph.target(edge) == scc[0])
			internalEdges.push_back(&originalGraph[edge]);

	if (llvm::all_of(internalEdges, isForward))
	{
		std::get<Equation>(content).setForward(true);
		return { content };
	}

	if (llvm::all_of(internalEdges, isBackward))
	{
		std::get<Equation>(content).setForward(false);
		return { content };
	}

	return {};
}

static llvm::SmallVector<std::variant<Equation, BltBlock>, 3> schedule(
		const SCC<VVarDependencyGraph>& scc,
		const VVarDependencyGraph& originalGraph)
{
	if (auto sched = trivialScheduling(scc, originalGraph); !sched.empty())
		return sched;

	// After a topological sort there should be no need of Khan's algorithm.
	assert(false && "Unreachable?");

	SVarDependencyGraph scalarGraph(originalGraph, scc);
	return collapseEquations(scalarGraph);
}

using ResultVector = llvm::SmallVector<llvm::SmallVector<std::variant<Equation, BltBlock>, 3>, 3>;
using SortedSCC = llvm::SmallVector<const SCC<VVarDependencyGraph>*, 3>;

mlir::LogicalResult marco::codegen::model::schedule(Model& model)
{
	VVarDependencyGraph vectorGraph(model);
	SCCDependencyGraph sccDependency(vectorGraph);

	SortedSCC sortedSCC = sccDependency.topologicalSort();
	ResultVector results(sortedSCC.size(), {});

	for (size_t i : irange(sortedSCC.size()))
		results[i] = ::schedule(*sortedSCC[i], vectorGraph);

	llvm::SmallVector<Equation, 3> equations;
	llvm::SmallVector<BltBlock, 3> bltBlocks;

	assert(model.getOp().body().getBlocks().size() == 1);
	mlir::Operation* op = model.getOp().body().front().getTerminator();

	for (const auto& res : results)
	{
		for (const auto& content : res)
		{
			if (std::holds_alternative<Equation>(content))
			{
				const Equation& equation = std::get<Equation>(content);
				equation.getOp()->moveBefore(op);
				equations.push_back(equation);
			}
			else
			{
				const BltBlock& bltBlock = std::get<BltBlock>(content);
				bltBlocks.push_back(bltBlock);
				for (const Equation& equation : bltBlock.getEquations())
					equation.getOp()->moveBefore(op);
			}
		}
	}

	Model result(model.getOp(), model.getVariables(), equations, bltBlocks);
	model = result;
	return mlir::success();
}
