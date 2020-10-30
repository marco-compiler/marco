#pragma once

#include <cstddef>
#include <llvm/Support/raw_ostream.h>

#include "llvm/ADT/SmallVector.h"
#include "modelica/model/ModExp.hpp"
namespace modelica
{
	class EquationPath
	{
		public:
		EquationPath(llvm::SmallVector<size_t, 3> path, bool left)
				: path(std::move(path)), left(left)
		{
		}

		[[nodiscard]] bool isOnEquationLeftHand() const { return left; }
		[[nodiscard]] size_t depth() const { return path.size(); }

		[[nodiscard]] auto begin() const { return path.begin(); }
		[[nodiscard]] auto end() const { return path.end(); }

		[[nodiscard]] ModExp& reach(ModExp& exp) const
		{
			ModExp* e = &exp;
			for (auto i : path)
				e = &e->getChild(i);

			return *e;
		}
		[[nodiscard]] const ModExp& reach(const ModExp& exp) const
		{
			const ModExp* e = &exp;
			for (auto i : path)
				e = &e->getChild(i);

			return *e;
		}

		void print(llvm::raw_ostream& OS) const
		{
			OS << "[";
			OS << (left ? "0" : "1");
			for (auto i : path)
			{
				OS << ",";
				OS << i;
			}
			OS << "]";
		}
		void dump(llvm::raw_ostream& OS) const { print(llvm::outs()); }

		private:
		llvm::SmallVector<size_t, 3> path;
		bool left;
	};

	class ModExpPath
	{
		public:
		ModExpPath(const ModExp& exp, llvm::SmallVector<size_t, 3> path, bool left)
				: path(std::move(path), left), exp(&exp)
		{
		}

		ModExpPath(const ModExp& exp, EquationPath path)
				: path(std::move(path)), exp(&exp)
		{
		}

		[[nodiscard]] const ModExp& getExp() const { return *exp; }
		[[nodiscard]] bool isOnEquationLeftHand() const
		{
			return path.isOnEquationLeftHand();
		}
		[[nodiscard]] size_t depth() const { return path.depth(); }

		[[nodiscard]] auto begin() const { return path.begin(); }
		[[nodiscard]] auto end() const { return path.end(); }
		[[nodiscard]] ModExp& reach(ModExp& exp) const { return path.reach(exp); }
		[[nodiscard]] const ModExp& reach(const ModExp& exp) const
		{
			return path.reach(exp);
		}
		[[nodiscard]] const EquationPath& getEqPath() const { return path; }

		private:
		EquationPath path;
		const ModExp* exp;
	};
}	 // namespace modelica
