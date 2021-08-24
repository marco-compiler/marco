#pragma once

#include "ModExp.hpp"
#include "llvm/Support/raw_ostream.h"

namespace marco
{
	class ModEqTemplate
	{
		public:
		ModEqTemplate(ModExp left, ModExp right, std::string tempName)
				: leftHand(std::move(left)),
					rightHand(std::move(right)),
					templateName(std::move(tempName))
		{
		}

		[[nodiscard]] const ModExp& getLeft() const { return leftHand; }
		[[nodiscard]] const ModExp& getRight() const { return rightHand; }
		[[nodiscard]] ModExp& getLeft() { return leftHand; }
		[[nodiscard]] ModExp& getRight() { return rightHand; }
		void dump(bool dumpName = true, llvm::raw_ostream& OS = llvm::outs()) const
		{
			if (dumpName)
				OS << templateName << " ";
			leftHand.dump(OS);
			OS << " = ";
			rightHand.dump(OS);
		}

		[[nodiscard]] const std::string& getName() const { return templateName; }
		void setName(std::string name) { templateName = name; }

		void swapLeftRight() { std::swap(leftHand, rightHand); }

		private:
		ModExp leftHand;
		ModExp rightHand;
		std::string templateName;
	};
}	 // namespace marco
