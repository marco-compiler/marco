#pragma once

#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/utils/IndexSet.hpp>

namespace modelica::codegen::model
{
	class Equation;
	class Variable;

	class Edge
	{
		public:
		Edge(const Equation& eq, const Variable& var, VectorAccess vAccess, ExpressionPath access, size_t index);

		[[nodiscard]] const Equation& getEquation() const;
		[[nodiscard]] const Variable& getVariable() const;

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
		const Equation* equation;
		const Variable* variable;
		VectorAccess vectorAccess;
		VectorAccess invertedAccess;
		IndexSet set;
		size_t index;
		ExpressionPath pathToExp;
	};
}	 // namespace modelica
