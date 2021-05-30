#pragma once

#include <modelica/utils/SourcePosition.h>
#include <variant>

namespace modelica::frontend
{
	class Class;
	class Induction;
	class Member;

	class Symbol
	{
		public:
		Symbol();
		explicit Symbol(Class& cls);
		explicit Symbol(Member& member);
		explicit Symbol(Induction& induction);

		template<typename T>
		[[nodiscard]] bool isa() const
		{
			return std::holds_alternative<T*>(content);
		}

		template<typename T>
		[[nodiscard]] T* get()
		{
			assert(isa<T>());
			return std::get<T*>(content);
		}

		template<typename T>
		[[nodiscard]] const T* get() const
		{
			assert(isa<T>());
			return std::get<T*>(content);
		}

		template<typename T>
		[[nodiscard]] T* dyn_get()
		{
			if (!isa<T>())
				return nullptr;

			return std::get<T*>(content);
		}

		template<typename T>
		[[nodiscard]] const T* dyn_get() const
		{
			if (!isa<T>())
				return nullptr;

			return std::get<T*>(content);
		}

		template<typename Visitor>
		auto visit(Visitor&& visitor)
		{
			return std::visit(visitor, content);
		}

		template<typename Visitor>
		auto visit(Visitor&& visitor) const
		{
			return std::visit(visitor, content);
		}

		private:
		std::variant<Class*, Member*, Induction*> content;
	};
}
