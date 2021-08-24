#pragma once
#include "llvm/Support/raw_ostream.h"
#include "marco/model/ModExpPath.hpp"
#include "marco/model/ModVariable.hpp"
#include "marco/model/Model.hpp"
#include "marco/model/VectorAccess.hpp"
#include "marco/utils/IndexSet.hpp"

namespace marco
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
}	 // namespace marco
