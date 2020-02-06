#pragma once

#include "llvm/ADT/SmallVector.h"
#include "modelica/model/ModEquation.hpp"
namespace modelica
{
	class ModExpPath
	{
		public:
		ModExpPath(
				const ModEquation& eq,
				llvm::SmallVector<unsigned short, 3> path,
				bool left)
				: path(std::move(path)), equation(&eq), left(left)
		{
		}

		[[nodiscard]] const ModEquation& getEquation() const { return *equation; }
		[[nodiscard]] bool isOnEquationLeftHand() const { return left; }
		[[nodiscard]] size_t depth() const { return path.size(); }

		[[nodiscard]] auto begin() const { return path.begin(); }
		[[nodiscard]] auto end() const { return path.begin(); }

		private:
		llvm::SmallVector<unsigned short, 3> path;
		const ModEquation* equation;
		bool left;
	};
}	 // namespace modelica
