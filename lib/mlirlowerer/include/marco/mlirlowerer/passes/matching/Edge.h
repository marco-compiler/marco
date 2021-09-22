#pragma once

#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Path.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <marco/utils/IndexSet.hpp>

namespace marco::codegen::model
{
	class Variable;

	class Edge
	{
		public:
		Edge(Equation equation, Variable variable, VectorAccess vectorAccess, ExpressionPath access, size_t index);

		[[nodiscard]] Equation getEquation() const;
		[[nodiscard]] Variable getVariable() const;

		[[nodiscard]] const VectorAccess& getVectorAccess() const;
		[[nodiscard]] const VectorAccess& getInvertedAccess() const;

		[[nodiscard]] IndexSet& getSet();
		[[nodiscard]] const IndexSet& getSet() const;

		[[nodiscard]] IndexSet map(const IndexSet& set) const;
		[[nodiscard]] IndexSet invertMap(const IndexSet& set) const;

		[[nodiscard]] bool empty() const;
		[[nodiscard]] size_t getIndex() const;
		[[nodiscard]] ExpressionPath& getPath();
		[[nodiscard]] const ExpressionPath& getPath() const;

		private:
		Equation equation;
		Variable variable;
		VectorAccess vectorAccess;
		VectorAccess invertedAccess;
		IndexSet set;
		size_t index;
		ExpressionPath pathToExp;
	};
}