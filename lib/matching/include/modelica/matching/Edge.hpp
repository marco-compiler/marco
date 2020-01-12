#pragma once
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/Model.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IndexSet.hpp"

namespace modelica
{
	class Edge
	{
		public:
		Edge(
				const ModEquation& eq,
				const ModVariable& var,
				size_t index,
				VectorAccess acc)
				: vectorAccess(std::move(acc)),
					invertedAccess(vectorAccess.invert()),
					equation(&eq),
					variable(&var),
					index(index)
		{
		}
		Edge(
				const Model& model,
				const ModEquation& eq,
				const ModExp& access,
				size_t index)
				: vectorAccess(VectorAccess::fromExp(access)),
					invertedAccess(vectorAccess.invert()),
					equation(&eq),
					variable(&(model.getVar(vectorAccess.getName()))),
					index(index)
		{
		}

		[[nodiscard]] const ModEquation& getEquation() const { return *equation; }
		[[nodiscard]] const ModVariable& getVariable() const { return *variable; }

		[[nodiscard]] IndexSet& getSet() { return set; }
		[[nodiscard]] const IndexSet& getSet() const { return set; }
		[[nodiscard]] const VectorAccess& getVectorAccess() const
		{
			return vectorAccess;
		}
		[[nodiscard]] const VectorAccess& getInvertedAccess() const
		{
			return invertedAccess;
		}

		[[nodiscard]] IndexSet map(const IndexSet& set) const
		{
			return vectorAccess.map(set);
		}

		[[nodiscard]] IndexSet invertMap(const IndexSet& set) const
		{
			return invertedAccess.map(set);
		}
		[[nodiscard]] size_t getIndex() const { return index; }
		[[nodiscard]] bool empty() const { return set.empty(); }
		void dump(llvm::raw_ostream& OS) const;
		[[nodiscard]] std::string toString() const;

		private:
		VectorAccess vectorAccess;
		VectorAccess invertedAccess;
		const ModEquation* equation;
		IndexSet set;
		const ModVariable* variable;
		size_t index;
	};
}	 // namespace modelica
