#pragma once
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/ModExpPath.hpp"
#include "modelica/model/ModVariable.hpp"
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
				VectorAccess vAccess,
				ModExpPath access,
				size_t index)
				: vectorAccess(std::move(vAccess)),
					equation(&eq),
					variable(&var),
					pathToExp(std::move(access)),
					index(index)
		{
			if (eq.isForEquation())
				invertedAccess = vectorAccess.invert();
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
		[[nodiscard]] const ModExpPath& getPath() const { return pathToExp; }
		[[nodiscard]] ModExpPath& getPath() { return pathToExp; }

		private:
		VectorAccess vectorAccess;
		const ModEquation* equation;
		IndexSet set;
		const ModVariable* variable;
		ModExpPath pathToExp;
		size_t index;
		VectorAccess invertedAccess;
	};
}	 // namespace modelica
