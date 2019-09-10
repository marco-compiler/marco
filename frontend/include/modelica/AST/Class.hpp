#pragma once
#include "modelica/AST/Equation.hpp"
#include "modelica/AST/Expr.hpp"
#include "modelica/AST/Statement.hpp"

namespace modelica
{
	/**
	 * page 270, type-specifier rule.
	 */
	using TypeSpecifier = std::pair<std::vector<std::string>, bool>;

	class Declaration
	{
		public:
		enum DeclarationKind
		{
			Decl,
			CompositeDecl,
			ClassModification,
			Composition,
			ExtendClause,
			EnumerationLiteral,
			Element,
			ClassDecl,
			EnumerationClassDecl,
			LongClassDecl,
			ShortClassDecl,
			DerClassDecl,
			LastClassDecl,
			CompositionSection,
			OverridingClassModification,
			ImportClause,
			ComponentDeclaration,
			Annotation,
			ComponentClause,
			Redeclaration,
			ReplecableModification,
			ConstrainingClause,
			ElementModification,
			LastCompositeDecl,
			ExprCompositeDecl,
			ConditionAttribute,
			ExternalFunctionCall,
			SimpleModification,
			ArraySubscriptionDecl,
			LastExprCompositeDecl,
			EqCompositeDecl,
			EquationSection,
			LastEqCompositeDecl,
			StatementCompositeDecl,
			AlgorithmSection,
			LastStatementCompositeDecl,
			ElementList,
			LastDecl
		};

		[[nodiscard]] DeclarationKind getKind() const { return kind; }
		Declaration(SourceRange location, DeclarationKind kind)
				: location(std::move(location)), kind(kind)
		{
		}
		virtual ~Declaration() = default;
		void setComment(std::string com) { comment = std::move(com); }
		[[nodiscard]] const std::string& getComment() const { return comment; }

		private:
		SourceRange location;
		std::string comment;
		DeclarationKind kind;
	};

	using UniqueDecl = std::unique_ptr<Declaration>;
	/**
	 * This is the template used by every ast leaf member as an alias
	 * to implement classof, which is used by llvm::cast
	 */
	template<Declaration::DeclarationKind kind>
	constexpr bool leafClassOf(const Declaration* e)
	{
		return e->getKind() == kind;
	}

	/**
	 * This is the template used by every ast non leaf member as an alias
	 * to implement classof, which is used by llvm::cast
	 */
	template<
			Declaration::DeclarationKind kind,
			Declaration::DeclarationKind lastKind>
	constexpr bool nonLeafClassOf(const Declaration* e)
	{
		return e->getKind() >= kind && e->getKind() < lastKind;
	}

	template<
			typename Children,
			Declaration::DeclarationKind firstKind,
			Declaration::DeclarationKind lastKind>
	class NonLeafDeclaration: public Declaration
	{
		public:
		using Iterator = typename vectorUnique<Children>::iterator;
		using ConstIterator = typename vectorUnique<Children>::const_iterator;
		NonLeafDeclaration(
				SourceRange range,
				DeclarationKind kin = firstKind,
				vectorUnique<Children> childs = {})
				: Declaration(range, kin), childs(std::move(childs))
		{
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~NonLeafDeclaration() override = default;
		static constexpr auto classof = nonLeafClassOf<firstKind, lastKind>;

		[[nodiscard]] int size() const { return childs.size(); }
		[[nodiscard]] Iterator begin() { return childs.begin(); }
		[[nodiscard]] Iterator end() { return childs.end(); }
		[[nodiscard]] ConstIterator cbegin() const { return childs.cbegin(); }
		[[nodiscard]] ConstIterator cend() const { return childs.cend(); }
		[[nodiscard]] Children* at(int index) { return childs.at(index).get(); }
		[[nodiscard]] const Children* at(int index) const
		{
			return childs.at(index).get();
		}
		[[nodiscard]] vectorUnique<Children> takeVector()
		{
			return std::move(childs);
		}
		[[nodiscard]] llvm::iterator_range<Iterator> children()
		{
			return llvm::make_range(begin(), end());
		}
		[[nodiscard]] llvm::iterator_range<ConstIterator> children() const
		{
			return llvm::make_range(cbegin(), cend());
		}
		void removeNulls()
		{
			childs.erase(std::remove(begin(), end(), nullptr), end());
		}

		protected:
		[[nodiscard]] vectorUnique<Children>& getVector() { return childs; }
		[[nodiscard]] const vectorUnique<Children>& getVector() const
		{
			return childs;
		}

		private:
		vectorUnique<Children> childs;
	};

	/**
	 * page 267, declaration rule. string is ident, unique expr is array
	 * subscript, unique decl is modification.
	 */
	using DeclarationName = std::tuple<std::string, UniqueDecl, UniqueDecl>;

	using CompositeDecl = NonLeafDeclaration<
			Declaration,
			Declaration::CompositeDecl,
			Declaration::LastCompositeDecl>;

	using ExprCompositeDecl = NonLeafDeclaration<
			Expr,
			Declaration::ExprCompositeDecl,
			Declaration::LastExprCompositeDecl>;

	using EqCompositeDecl = NonLeafDeclaration<
			Equation,
			Declaration::EqCompositeDecl,
			Declaration::LastEqCompositeDecl>;

	using StatementCompositeDecl = NonLeafDeclaration<
			Statement,
			Declaration::StatementCompositeDecl,
			Declaration::LastStatementCompositeDecl>;

	class ClassDecl: public CompositeDecl
	{
		public:
		enum class SubType
		{
			Class,
			Model,
			Operator,
			Record,
			OperatorRecord,
			Block,
			Connector,
			ExpandableConnector,
			Type,
			Package,
			Function,
			OperatorFunction
		};

		static constexpr auto classof = nonLeafClassOf<
				DeclarationKind::ClassDecl,
				DeclarationKind::LastClassDecl>;

		ClassDecl(
				SourceRange range,
				DeclarationKind kind = DeclarationKind::ClassDecl,
				vectorUnique<Declaration> childs = {})
				: NonLeafDeclaration(range, kind, move(childs))
		{
		}
		~ClassDecl() override = default;

		void setType(SubType type) { subtype = type; }
		void setPartial(bool part) { partial = part; }
		void setEncapsulated(bool enc) { encapsulated = enc; }
		void setPure(bool p) { pure = p; }
		void setName(std::string newName) { name = std::move(newName); }
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}

		[[nodiscard]] bool isEncapsulated() const { return encapsulated; }
		[[nodiscard]] bool isPure() const { return pure; }
		[[nodiscard]] bool isPartial() const { return partial; }
		[[nodiscard]] SubType subType() const { return subtype; }

		private:
		bool partial{ false };
		bool encapsulated{ false };
		bool pure{ true };
		std::string name;
		SubType subtype{ SubType::Class };
	};

	class SimpleModification: public ExprCompositeDecl
	{
		public:
		SimpleModification(SourceRange range, UniqueExpr expr)
				: NonLeafDeclaration(range, DeclarationKind::SimpleModification)
		{
			getVector().emplace_back(std::move(expr));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~SimpleModification() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::SimpleModification>;

		[[nodiscard]] const Expr* getExpression() const
		{
			return getVector()[0].get();
		}
	};

	class Annotation: public CompositeDecl
	{
		public:
		Annotation(SourceRange range, UniqueDecl decl)
				: NonLeafDeclaration(range, DeclarationKind::Annotation)
		{
			getVector().emplace_back(std::move(decl));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~Annotation() override = default;
		static constexpr auto classof = leafClassOf<DeclarationKind::Annotation>;

		[[nodiscard]] const Declaration* getModification() const
		{
			return getVector()[0].get();
		}
	};

	class OverridingClassModification: public CompositeDecl
	{
		public:
		OverridingClassModification(
				SourceRange range, UniqueDecl classMod, UniqueDecl simpleMd)
				: NonLeafDeclaration(
							range, DeclarationKind::OverridingClassModification)
		{
			getVector().emplace_back(std::move(classMod));
			getVector().emplace_back(std::move(simpleMd));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~OverridingClassModification() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::OverridingClassModification>;

		[[nodiscard]] const Declaration* getClassModification() const
		{
			return getVector()[0].get();
		}
		[[nodiscard]] const Declaration* getSimpleModification() const
		{
			return getVector()[1].get();
		}
	};
	class CompositionSection: public CompositeDecl
	{
		public:
		CompositionSection(SourceRange range, vectorUnique<Declaration> decls)
				: NonLeafDeclaration(
							range, DeclarationKind::CompositionSection, std::move(decls))
		{
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~CompositionSection() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::CompositionSection>;
	};

	class ClassModification: public CompositeDecl
	{
		public:
		ClassModification(SourceRange range, vectorUnique<Declaration> decls)
				: NonLeafDeclaration(
							range, DeclarationKind::ClassModification, std::move(decls))
		{
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ClassModification() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::ClassModification>;
	};
	class ComponentDeclaration: public CompositeDecl
	{
		public:
		ComponentDeclaration(
				SourceRange range,
				std::string ident,
				UniqueDecl arraySubSubscript = nullptr,
				UniqueDecl subModification = nullptr,
				UniqueDecl annotation = nullptr,
				UniqueDecl conditionAttribute = nullptr)
				: NonLeafDeclaration(range, DeclarationKind::ComponentDeclaration),
					ident(move(ident))
		{
			if (arraySubSubscript != nullptr)
				getVector().push_back(std::move(arraySubSubscript));
			if (subModification != nullptr)
				getVector().push_back(std::move(subModification));
			if (annotation != nullptr)
				getVector().push_back(std::move(annotation));
			if (conditionAttribute != nullptr)
				getVector().push_back(std::move(conditionAttribute));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ComponentDeclaration() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::ComponentDeclaration>;

		private:
		std::string ident;
	};

	class ComponentClause: public CompositeDecl
	{
		public:
		enum class FlowStream
		{
			flow,
			stream,
			none
		};
		enum class IO
		{
			input,
			output,
			none
		};
		enum class Type
		{
			discrete,
			parameter,
			consant,
			none
		};

		class Prefix
		{
			public:
			Prefix(FlowStream fl, IO io, Type typ): flowstream(fl), io(io), type(typ)
			{
			}
			[[nodiscard]] FlowStream getFlowStream() const { return flowstream; }
			[[nodiscard]] IO getIOType() const { return io; }
			[[nodiscard]] Type getType() const { return type; }
			[[nodiscard]] bool noneSet() const
			{
				return flowstream == FlowStream::none && io == IO::none &&
							 type == Type::none;
			}

			private:
			FlowStream flowstream;
			IO io;
			Type type;
		};

		ComponentClause(
				SourceRange range,
				Prefix prefix,
				bool gLookUp,
				std::vector<std::string> nm,
				std::vector<UniqueDecl> componendDecl,
				UniqueDecl arraySubscript = nullptr)
				: NonLeafDeclaration(
							range,
							DeclarationKind::ComponentClause,
							std::move(componendDecl)),
					prefix(prefix),
					globalLookUp(gLookUp),
					name(std::move(nm))
		{
			if (arraySubscript != nullptr)
				getVector().push_back(std::move(arraySubscript));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ComponentClause() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::ComponentClause>;

		[[nodiscard]] bool hasGlobalLookup() const { return globalLookUp; }
		[[nodiscard]] const Prefix& getPrefix() const { return prefix; }
		[[nodiscard]] const std::vector<std::string>& getNme() const
		{
			return name;
		}
		[[nodiscard]] const Declaration* getComponent(int index) const
		{
			return getVector().at(index).get();
		}

		private:
		Prefix prefix;
		bool globalLookUp;
		std::vector<std::string> name;
	};

	class ElementModification: public CompositeDecl
	{
		public:
		ElementModification(
				SourceRange range,
				UniqueDecl modification,
				std::vector<std::string> nm,
				bool eachModification,
				bool finalModification)
				: NonLeafDeclaration(range, DeclarationKind::ElementModification),
					name(std::move(nm)),
					each(eachModification),
					finl(finalModification)
		{
			getVector().push_back(std::move(modification));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ElementModification() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::ElementModification>;

		[[nodiscard]] bool hasEach() const { return each; }
		[[nodiscard]] bool isFinal() const { return finl; }
		[[nodiscard]] const std::vector<std::string>& getName() const
		{
			return name;
		}

		private:
		std::vector<std::string> name;
		bool each;
		bool finl;
	};

	class ReplecableModification: public CompositeDecl
	{
		public:
		ReplecableModification(
				SourceRange range,
				UniqueDecl child,
				bool each,
				bool fnl,
				UniqueDecl constrainingClause = nullptr,
				UniqueDecl annotation = nullptr)
				: NonLeafDeclaration(range, DeclarationKind::ReplecableModification),
					each(each),
					finl(fnl)
		{
			getVector().push_back(std::move(child));
			if (constrainingClause != nullptr)
				getVector().push_back(std::move(constrainingClause));
			if (annotation != nullptr)
				getVector().push_back(std::move(annotation));
		}

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ReplecableModification() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::ReplecableModification>;

		[[nodiscard]] bool hasConstraingClause() const { return size() >= 2; }

		[[nodiscard]] bool isFinal() const { return finl; }
		[[nodiscard]] bool hasEach() const { return each; }

		private:
		bool each;
		bool finl;
	};

	class Element: public CompositeDecl
	{
		public:
		Element(SourceRange range, UniqueDecl child, bool inner, bool outer)
				: NonLeafDeclaration(range, DeclarationKind::Element),
					inner(inner),
					outer(outer)
		{
			getVector().push_back(std::move(child));
		}

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~Element() override = default;
		static constexpr auto classof = leafClassOf<DeclarationKind::Element>;

		[[nodiscard]] bool hasConstraingClause() const { return size() >= 2; }
		[[nodiscard]] bool isInner() const { return inner; }
		[[nodiscard]] bool isOuter() const { return outer; }

		private:
		bool inner;
		bool outer;
	};

	class Redeclaration: public CompositeDecl
	{
		public:
		Redeclaration(SourceRange range, UniqueDecl child, bool each, bool fnl)
				: NonLeafDeclaration(range, DeclarationKind::Redeclaration),
					fnl(fnl),
					each(each)
		{
			getVector().push_back(std::move(child));
		}

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~Redeclaration() override = default;
		static constexpr auto classof = leafClassOf<DeclarationKind::Redeclaration>;

		[[nodiscard]] bool hasConstraingClause() const { return size() >= 2; }
		[[nodiscard]] bool isFinal() const { return fnl; }
		[[nodiscard]] bool hasEach() const { return each; }

		private:
		bool fnl;
		bool each;
	};

	class ConstrainingClause: public CompositeDecl
	{
		public:
		ConstrainingClause(
				SourceRange range, TypeSpecifier spec, UniqueDecl mod = nullptr)
				: NonLeafDeclaration(range, DeclarationKind::ConstrainingClause),
					specifier(move(spec))
		{
			if (mod != nullptr)
				getVector().push_back(std::move(mod));
		}

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ConstrainingClause() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::ConstrainingClause>;

		[[nodiscard]] bool hasModification() const { return size() >= 1; }

		private:
		TypeSpecifier specifier;
	};

	class EnumerationLiteral: public CompositeDecl
	{
		public:
		EnumerationLiteral(
				SourceRange range,
				std::string enumerationName,
				UniqueDecl annotation = nullptr)
				: NonLeafDeclaration(range, DeclarationKind::EnumerationLiteral),
					name(move(enumerationName))
		{
			if (annotation != nullptr)
				getVector().push_back(std::move(annotation));
		}

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		[[nodiscard]] const Declaration* getAnnotation() const
		{
			return getVector()[0].get();
		}
		~EnumerationLiteral() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::EnumerationLiteral>;

		[[nodiscard]] const std::string& getName() const { return name; }

		private:
		std::string name;
	};

	class ExtendClause: public CompositeDecl
	{
		public:
		ExtendClause(
				SourceRange range,
				TypeSpecifier spec,
				UniqueDecl mod = nullptr,
				UniqueDecl annotation = nullptr)
				: NonLeafDeclaration(range, DeclarationKind::ExtendClause),
					specifier(move(spec))
		{
			if (mod != nullptr)
				getVector().push_back(std::move(mod));
			if (annotation != nullptr)
				getVector().push_back(std::move(annotation));
		}

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ExtendClause() override = default;
		static constexpr auto classof = leafClassOf<DeclarationKind::ExtendClause>;

		[[nodiscard]] bool hasModification() const { return size() >= 1; }

		private:
		TypeSpecifier specifier;
	};

	class Composition: public CompositeDecl
	{
		public:
		Composition(
				SourceRange range,
				UniqueDecl privateDecl,
				UniqueDecl publicDecl,
				UniqueDecl protectedDecl,
				vectorUnique<Declaration> equationSection,
				vectorUnique<Declaration> algSection,
				UniqueDecl externalFunctionCall = nullptr,
				UniqueDecl externalCallAnn = nullptr,
				UniqueDecl annotation = nullptr,
				std::string externalLanguageSpec = "")
				: NonLeafDeclaration(range, DeclarationKind::Composition),
					languageSpec(move(externalLanguageSpec))
		{
			getVector().push_back(move(privateDecl));
			getVector().push_back(move(publicDecl));
			getVector().push_back(move(protectedDecl));
			getVector().push_back(move(externalFunctionCall));
			getVector().push_back(move(externalCallAnn));
			getVector().push_back(move(annotation));
			for (auto& p : equationSection)
				getVector().push_back(move(p));
			for (auto& p : algSection)
				getVector().push_back(move(p));
		}
		[[nodiscard]] const Declaration* getPrivateSection() const
		{
			return getVector()[0].get();
		}
		[[nodiscard]] const Declaration* getPublicSection() const
		{
			return getVector()[1].get();
		}
		[[nodiscard]] const Declaration* getProtectedSection() const
		{
			return getVector()[2].get();
		}
		[[nodiscard]] const Declaration* getExternalFunctionCall() const
		{
			return getVector()[3].get();
		}
		[[nodiscard]] const Declaration* getExternalCallAnnotation() const
		{
			return getVector()[4].get();
		}
		[[nodiscard]] const Declaration* getAnnotation() const
		{
			return getVector()[5].get();
		}
		[[nodiscard]] const std::string& getLanguageSpec() const
		{
			return languageSpec;
		}

		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~Composition() override = default;
		static constexpr auto classof = leafClassOf<DeclarationKind::Composition>;

		private:
		std::string languageSpec;
	};

	class ArraySubscriptionDecl: public ExprCompositeDecl
	{
		public:
		ArraySubscriptionDecl(SourceRange range, UniqueExpr expr)
				: NonLeafDeclaration(range, DeclarationKind::ArraySubscriptionDecl)
		{
			getVector().emplace_back(std::move(expr));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ArraySubscriptionDecl() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::ArraySubscriptionDecl>;

		[[nodiscard]] const Expr* getArraySubscript() const
		{
			return getVector()[0].get();
		}
	};

	class ExternalFunctionCall: public ExprCompositeDecl
	{
		public:
		ExternalFunctionCall(
				SourceRange range,
				std::string name,
				UniqueExpr args,
				UniqueExpr componentRef = nullptr)
				: NonLeafDeclaration(range, DeclarationKind::ExternalFunctionCall),
					name(std::move(name))
		{
			getVector().push_back(move(args));
			if (componentRef != nullptr)
				getVector().push_back(move(componentRef));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ExternalFunctionCall() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::ExternalFunctionCall>;

		[[nodiscard]] const Expr* getArguments() const
		{
			return getVector()[0].get();
		}

		[[nodiscard]] const std::string& getName() const { return name; }

		private:
		std::string name;
	};

	class ConditionAttribute: public ExprCompositeDecl
	{
		public:
		ConditionAttribute(SourceRange range, UniqueExpr expr)
				: NonLeafDeclaration(range, DeclarationKind::ConditionAttribute)
		{
			getVector().emplace_back(std::move(expr));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ConditionAttribute() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::ConditionAttribute>;

		[[nodiscard]] const Expr* getExpr() const { return getVector()[0].get(); }
	};

	class EquationSection: public EqCompositeDecl
	{
		public:
		EquationSection(
				SourceRange range, vectorUnique<Equation> equs, bool initial = false)
				: NonLeafDeclaration(
							range, DeclarationKind::EquationSection, move(equs)),
					initial(initial)
		{
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~EquationSection() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::EquationSection>;

		[[nodiscard]] const Equation* getEquation(int index) const
		{
			return getVector()[index].get();
		}
		[[nodiscard]] bool isInitial() const { return initial; }
		void setInitial(bool init) { initial = init; }

		private:
		bool initial;
	};

	class AlgorithmSection: public StatementCompositeDecl
	{
		public:
		AlgorithmSection(
				SourceRange range, vectorUnique<Statement> equs, bool initial = false)
				: NonLeafDeclaration(
							range, DeclarationKind::AlgorithmSection, move(equs)),
					initial(initial)
		{
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~AlgorithmSection() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::AlgorithmSection>;

		[[nodiscard]] const Statement* getEquation(int index) const
		{
			return getVector()[index].get();
		}
		[[nodiscard]] bool isInitial() const { return initial; }
		void setInitial(bool init) { initial = init; }

		private:
		bool initial;
	};

	class DerClassDecl: public ClassDecl
	{
		public:
		DerClassDecl(
				SourceRange range,
				std::vector<std::string> idents,
				TypeSpecifier spec,
				UniqueDecl commnt)
				: ClassDecl(range, DeclarationKind::DerClassDecl),
					idents(move(idents)),
					typeSpec(move(spec))
		{
			getVector().push_back(move(commnt));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~DerClassDecl() override = default;
		static constexpr auto classof = leafClassOf<DeclarationKind::DerClassDecl>;
		[[nodiscard]] const TypeSpecifier& getTypeSpecifier() const
		{
			return typeSpec;
		}
		[[nodiscard]] const std::vector<std::string>& getIdents() const
		{
			return idents;
		}

		private:
		std::vector<std::string> idents;
		TypeSpecifier typeSpec;
	};

	class EnumerationClass: public ClassDecl
	{
		public:
		EnumerationClass(
				SourceRange range,
				bool colons,
				vectorUnique<Declaration> enums,
				UniqueDecl annotation = nullptr)
				: ClassDecl(range, DeclarationKind::EnumerationClassDecl, move(enums)),
					colons(colons)
		{
			getVector().push_back(move(annotation));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~EnumerationClass() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::EnumerationClassDecl>;

		[[nodiscard]] bool hasColons() const { return colons; }
		[[nodiscard]] int enumsCount() const { return getVector().size() - 1; }
		[[nodiscard]] const Declaration* getEnumLiteratl(int index) const
		{
			return getVector().at(index).get();
		}

		private:
		bool colons;
	};

	class LongClassDecl: public ClassDecl
	{
		public:
		LongClassDecl(
				SourceRange range,
				UniqueDecl comp,
				UniqueDecl modification = nullptr,
				bool extends = false)
				: ClassDecl(range, DeclarationKind::LongClassDecl), extends(extends)
		{
			getVector().push_back(move(comp));
			getVector().push_back(move(modification));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~LongClassDecl() override = default;
		static constexpr auto classof = leafClassOf<DeclarationKind::LongClassDecl>;

		[[nodiscard]] const Declaration* getComposition() const
		{
			return getVector()[0].get();
		}
		[[nodiscard]] const Declaration* getModification() const
		{
			return getVector()[1].get();
		}

		[[nodiscard]] bool isExtend() const { return extends; }

		private:
		bool extends;
	};

	class ShortClassDecl: public ClassDecl
	{
		public:
		ShortClassDecl(
				SourceRange range,
				bool input,
				bool output,
				TypeSpecifier sp,
				UniqueDecl arraySub = nullptr,
				UniqueDecl modification = nullptr,
				UniqueDecl ann = nullptr)
				: ClassDecl(range, DeclarationKind::ShortClassDecl),
					input(input),
					output(output),
					typeSpec(std::move(sp))
		{
			getVector().push_back(std::move(arraySub));
			getVector().push_back(std::move(modification));
			getVector().push_back(std::move(ann));
		}
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		~ShortClassDecl() override = default;
		static constexpr auto classof =
				leafClassOf<DeclarationKind::ShortClassDecl>;

		[[nodiscard]] bool isInput() const { return input; }
		[[nodiscard]] bool isOutput() const { return output; }

		private:
		bool input;
		bool output;
		TypeSpecifier typeSpec;
	};

	class ElementList: public Declaration
	{
		public:
		ElementList(SourceRange range, vectorUnique<Declaration> declarations = {})
				: Declaration(range, DeclarationKind::ElementList),
					decls(std::move(declarations)),
					proected(false)
		{
		}
		~ElementList() override = default;
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		[[nodiscard]] bool isProtected() const { return proected; }
		[[nodiscard]] bool isPublic() const { return !proected; }
		static constexpr auto classof = leafClassOf<DeclarationKind::ElementList>;

		private:
		vectorUnique<Declaration> decls;
		bool proected;
	};

	class ImportClause: public CompositeDecl
	{
		public:
		ImportClause(
				SourceRange range,
				std::vector<std::string> baseName,
				std::string newName = "",
				UniqueDecl comment = nullptr,
				bool importAll = false,
				std::vector<std::string> toImportNames = {})
				: NonLeafDeclaration(range, DeclarationKind::ImportClause),
					newName(std::move(newName)),
					importAll(importAll),
					baseName(std::move(baseName)),
					toImportNames(std::move(toImportNames))

		{
			if (comment != nullptr)
				getVector().emplace_back(std::move(comment));
		}
		~ImportClause() override = default;
		[[nodiscard]] bool importAllNamespace() const { return importAll; }
		[[nodiscard]] llvm::Error isConsistent() const
		{
			return llvm::Error::success();
		}
		static constexpr auto classof = leafClassOf<DeclarationKind::ImportClause>;

		private:
		std::string newName;
		bool importAll;
		std::vector<std::string> baseName;
		std::vector<std::string> toImportNames;
	};
}	// namespace modelica
