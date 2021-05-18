#pragma once

#include <variant>

namespace modelica::frontend
{
	class DerFunction;
	class StandardFunction;
	class Induction;
	class Member;
	class Model;
	class Package;
	class Record;

	class Symbol
	{
		public:
		Symbol();
		explicit Symbol(DerFunction& function);
		explicit Symbol(StandardFunction& function);
		explicit Symbol(Model& model);
		explicit Symbol(Package& package);
		explicit Symbol(Record& record);
		explicit Symbol(Member& member);
		explicit Symbol(Induction& induction);

		template<typename T>
		[[nodiscard]] bool isa() const
		{
			return std::holds_alternative<T*>(content);
		}

		template<typename T>
		[[nodiscard]] const T* get() const
		{
			assert(isa<T>());
			return std::get<T*>(content);
		}

		private:
		std::variant<DerFunction*, StandardFunction*, Model*, Package*, Record*, Member*, Induction*> content;
	};
}
