#include <cstdio>
#include <marco/frontend/AST.h>
#include <marco/frontend/Errors.h>
#include <marco/frontend/passes/TypeCheckingPass.h>
#include <numeric>
#include <queue>
#include <stack>

using namespace marco;
using namespace marco::frontend;

static bool operator>=(Type x, Type y)
{
	assert(x.isa<BuiltInType>());
	assert(y.isa<BuiltInType>());

	if (y.get<BuiltInType>() == BuiltInType::Unknown)
		return true;

	if (x.get<BuiltInType>() == BuiltInType::Float)
		return true;

	if (x.get<BuiltInType>() == BuiltInType::Integer)
		return y.get<BuiltInType>() != BuiltInType::Float;

	if (x.get<BuiltInType>() == BuiltInType::Boolean)
		return y.get<BuiltInType>() == BuiltInType::Boolean;

	return false;
}

static BuiltInType getMostGenericBaseType(Type x, Type y)
{
	return x >= y ? x.get<BuiltInType>() : y.get<BuiltInType>();
}

namespace marco::frontend::typecheck::detail
{
	struct DerFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	struct AbsFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return Type(args[0]->getType().get<BuiltInType>());
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	struct AcosFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	struct AsinFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	struct AtanFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	struct Atan2Function : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
			ranks.push_back(0);
		}
	};

	struct CosFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	struct CoshFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	/**
	 * Returns a square matrix with the elements of a vector on the
	 * diagonal and all other elements set to zero.
	 */
	struct DiagonalFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			// 2D array as output
			return makeType<int>(-1, -1);
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			// 1D array as input
			ranks.push_back(1);
		}
	};

	struct ExpFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	/**
	 * Returns an integer identity matrix, with ones on the diagonal
	 * and zeros at the other places.
	 */
	struct IdentityFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			// 2D array as output
			return makeType<int>(-1, -1);
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			// Square matrix size
			ranks.push_back(0);
		}
	};

	/**
	 * Returns an array with equally spaced elements.
	 */
	struct LinspaceFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			// The result 1D array has a dynamic size, as it depends on the
			// input argument.
			return makeType<float>(-1);
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0); // x1
			ranks.push_back(0); // x2
			ranks.push_back(0); // n
		}
	};

	struct LogFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	struct Log10Function : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	/**
	 * Returns the greatest element of an array, or between two scalar values.
	 */
	struct MaxFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			if (args.size() == 1)
				return Type(args[0]->getType().get<BuiltInType>());

			if (args.size() == 2)
			{
				auto& xType = args[0]->getType();
				auto& yType = args[1]->getType();

				return xType >= yType ? xType : yType;
			}

			return llvm::None;
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			if (argsCount == 1)
			{
				// The array can have any rank
				ranks.push_back(-1);
			}
			else if (argsCount == 2)
			{
				ranks.push_back(0);
				ranks.push_back(0);
			}
		}
	};

	/**
	 * Returns the least element of an array, or between two scalar values.
	 */
	struct MinFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			if (args.size() == 1)
				return Type(args[0]->getType().get<BuiltInType>());

			if (args.size() == 2)
			{
				auto& xType = args[0]->getType();
				auto& yType = args[1]->getType();

				return xType >= yType ? xType : yType;
			}

			return llvm::None;
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			if (argsCount == 1)
			{
				// The array can have any rank
				ranks.push_back(-1);
			}
			else if (argsCount == 2)
			{
				ranks.push_back(0);
				ranks.push_back(0);
			}
		}
	};

	/**
	 * Returns the number of dimensions of an array.
	 */
	struct NdimsFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<int>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(-1);
		}
	};

	/**
	 * Return a n-D array with all elements equal to one.
	 */
	struct OnesFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			llvm::SmallVector<ArrayDimension, 2> dimensions(args.size(), -1);
			return Type(BuiltInType::Integer, dimensions);
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			// All the arguments are scalars
			for (size_t i = 0; i < argsCount; ++i)
				ranks.push_back(0);
		}
	};

	/**
	 * Returns the scalar product of all the elements of an array.
	 */
	struct ProductFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return Type(args[0]->getType().get<BuiltInType>());
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(-1);
		}
	};

	struct SignFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<int>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	struct SinFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	struct SinhFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	/**
	 * If there is a single array argument, then returns a 1D array containing
	 * the dimension sizes of A. If a second scalar argument is provided, only
	 * the size of that dimension is returned.
	 */
	struct SizeFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			if (args.size() == 1)
				return makeType<int>(args[0]->getType().getDimensions().size());

			if (args.size() == 2)
				return makeType<int>();

			return llvm::None;
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(-1);

			if (argsCount == 2)
				ranks.push_back(0);
		}
	};

	struct SqrtFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	/**
	 * Returns the scalar sum of all the elements of an array.
	 */
	struct SumFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return Type(args[0]->getType().get<BuiltInType>());
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(-1);
		}
	};

	/**
	 * Returns a matrix where the diagonal elements and the elements above the
	 * diagonal are identical to the corresponding elements of the source matrix,
	 * while the elements below the diagonal are set equal to the elements above
	 * the diagonal.
	 */
	struct SymmetricFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return args[0]->getType();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(2);
		}
	};

	struct TanFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	struct TanhFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			return makeType<float>();
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return true;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(0);
		}
	};

	/**
	 * Permutes a matrix.
	 * // TODO: should accept also arrays with rank > 2
	 */
	struct TransposeFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			auto type = args[0]->getType();
			llvm::SmallVector<ArrayDimension, 2> dimensions;

			dimensions.push_back(type[1].isDynamic() ? -1 : type[1].getNumericSize());
			dimensions.push_back(type[0].isDynamic() ? -1 : type[0].getNumericSize());

			type.setDimensions(dimensions);
			return type;
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			ranks.push_back(2);
		}
	};

	/**
	 * Return a n-D array with all elements equal to zero.
	 */
	struct ZerosFunction : public BuiltInFunction
	{
		[[nodiscard]] llvm::Optional<Type> resultType(
				llvm::ArrayRef<std::unique_ptr<Expression>> args) const override
		{
			llvm::SmallVector<ArrayDimension, 2> dimensions(args.size(), -1);
			return Type(BuiltInType::Integer, dimensions);
		}

		[[nodiscard]] bool canBeCalledElementWise() const override
		{
			return false;
		}

		void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const override
		{
			// All the arguments are scalars
			for (size_t i = 0; i < argsCount; ++i)
				ranks.push_back(0);
		}
	};
}

llvm::Error resolveDummyReferences(StandardFunction& function);
llvm::Error resolveDummyReferences(Model& model);

TypeChecker::TypeChecker()
{
	using namespace typecheck::detail;

	builtInFunctions["abs"] = std::make_unique<AbsFunction>();
	builtInFunctions["acos"] = std::make_unique<AcosFunction>();
	builtInFunctions["asin"] = std::make_unique<AsinFunction>();
	builtInFunctions["atan"] = std::make_unique<AtanFunction>();
	builtInFunctions["atan2"] = std::make_unique<Atan2Function>();
	builtInFunctions["cos"] = std::make_unique<CosFunction>();
	builtInFunctions["cosh"] = std::make_unique<CoshFunction>();
	builtInFunctions["der"] = std::make_unique<DerFunction>();
	builtInFunctions["diagonal"] = std::make_unique<DiagonalFunction>();
	builtInFunctions["exp"] = std::make_unique<ExpFunction>();
	builtInFunctions["identity"] = std::make_unique<IdentityFunction>();
	builtInFunctions["linspace"] = std::make_unique<LinspaceFunction>();
	builtInFunctions["log"] = std::make_unique<LogFunction>();
	builtInFunctions["log10"] = std::make_unique<Log10Function>();
	builtInFunctions["max"] = std::make_unique<MaxFunction>();
	builtInFunctions["min"] = std::make_unique<MinFunction>();
	builtInFunctions["ndims"] = std::make_unique<NdimsFunction>();
	builtInFunctions["ones"] = std::make_unique<OnesFunction>();
	builtInFunctions["product"] = std::make_unique<ProductFunction>();
	builtInFunctions["sign"] = std::make_unique<SignFunction>();
	builtInFunctions["sin"] = std::make_unique<SinFunction>();
	builtInFunctions["sinh"] = std::make_unique<SinhFunction>();
	builtInFunctions["size"] = std::make_unique<SizeFunction>();
	builtInFunctions["sqrt"] = std::make_unique<SqrtFunction>();
	builtInFunctions["sum"] = std::make_unique<SumFunction>();
	builtInFunctions["symmetric"] = std::make_unique<SymmetricFunction>();
	builtInFunctions["tan"] = std::make_unique<TanFunction>();
	builtInFunctions["tanh"] = std::make_unique<TanhFunction>();
	builtInFunctions["transpose"] = std::make_unique<TransposeFunction>();
	builtInFunctions["zeros"] = std::make_unique<ZerosFunction>();
}

template<>
llvm::Error TypeChecker::run<Class>(Class& cls)
{
	return cls.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(cls);
	});
}

llvm::Error TypeChecker::run(llvm::ArrayRef<std::unique_ptr<Class>> classes)
{
	for (const auto& cls : classes)
		if (auto error = run<Class>(*cls); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<PartialDerFunction>(Class& cls)
{
	SymbolTable::ScopeTy varScope(symbolTable);
	auto* derFunction = cls.get<PartialDerFunction>();

	// Populate the symbol table
	symbolTable.insert(derFunction->getName(), Symbol(cls));

	if (auto* derivedFunction = derFunction->getDerivedFunction(); !derivedFunction->isa<ReferenceAccess>())
		return llvm::make_error<BadSemantic>(derivedFunction->getLocation(), "the derived function must be a reference");

	Class* baseFunction = &cls;

	while (!baseFunction->isa<StandardFunction>())
	{
		auto symbol = symbolTable.lookup(
				derFunction->getDerivedFunction()->get<ReferenceAccess>()->getName());

		if (symbol.isa<Class>())
			baseFunction = symbol.get<Class>();
		else
			return llvm::make_error<BadSemantic>(
					derFunction->getLocation(),
					"the derived function name must refer to a function");

		if (!cls.isa<StandardFunction>() && !cls.isa<PartialDerFunction>())
			return llvm::make_error<BadSemantic>(
					derFunction->getLocation(),
					"the derived function name must refer to a function");
	}

	auto* standardFunction = baseFunction->get<StandardFunction>();
	auto members = standardFunction->getMembers();
	llvm::SmallVector<size_t, 3> independentVariablesIndexes;

	for (auto& independentVariable : derFunction->getIndependentVariables())
	{
		auto name = independentVariable->get<ReferenceAccess>()->getName();
		auto membersEnum = llvm::enumerate(members);

		auto member = std::find_if(membersEnum.begin(), membersEnum.end(),
																[&name](const auto& obj) {
																	return obj.value()->getName() == name;
																});

		if (member == membersEnum.end())
			return llvm::make_error<BadSemantic>(
					independentVariable->get<ReferenceAccess>()->getLocation(),
					"independent variable not found");

		auto type = (*member).value()->getType();

		if (!type.isa<float>())
			return llvm::make_error<BadSemantic>(
					independentVariable->getLocation(),
					"independent variables must have Real type");

		independentVariable->setType(std::move(type));
		independentVariablesIndexes.push_back((*member).index());
	}

	llvm::SmallVector<Type, 3> argsTypes;
	llvm::SmallVector<Type, 3> resultsTypes;

	for (const auto& arg : standardFunction->getArgs())
		argsTypes.push_back(arg->getType());

	for (const auto& result : standardFunction->getResults())
		resultsTypes.push_back(result->getType());

	derFunction->setArgsTypes(argsTypes);
	derFunction->setResultsTypes(resultsTypes);

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<StandardFunction>(Class& cls)
{
	SymbolTable::ScopeTy varScope(symbolTable);
	auto* function = cls.get<StandardFunction>();

	// Populate the symbol table
	symbolTable.insert(function->getName(), Symbol(cls));

	for (const auto& member : function->getMembers())
		symbolTable.insert(member->getName(), Symbol(*member));

	// Check members
	for (auto& member : function->getMembers())
	{
		if (auto error = run(*member); error)
			return error;

		// From Function reference:
		// "Each input formal parameter of the function must be prefixed by the
		// keyword input, and each result formal parameter by the keyword output.
		// All public variables are formal parameters."

		if (member->isPublic() && !member->isInput() && !member->isOutput())
			return llvm::make_error<BadSemantic>(
					member->getLocation(),
					"public members of functions must be input or output variables");

		// From Function reference:
		// "Input formal parameters are read-only after being bound to the actual
		// arguments or default values, i.e., they may not be assigned values in
		// the body of the function."

		if (member->isInput() && member->hasInitializer())
			return llvm::make_error<AssignmentToInputMember>(
					member->getInitializer()->getLocation(),
					function->getName());
	}

	auto algorithms = function->getAlgorithms();

	// From Function reference:
	// "A function can have at most one algorithm section or one external
	// function interface (not both), which, if present, is the body of the
	// function."

	if (algorithms.size() > 1)
		return llvm::make_error<MultipleAlgorithmsFunction>(
				function->getAlgorithms()[1]->getLocation(),
				function->getName());

	// For now, functions can't have an external implementation and thus must
	// have exactly one algorithm section. When external implementations will
	// be allowed, the algorithms amount may also be zero.
	assert(algorithms.size() == 1);

	if (auto error = run(*algorithms[0]); error)
		return error;

	if (auto error = resolveDummyReferences(*function); error)
		return error;

	for (const auto& statement : *algorithms[0])
	{
		for (const auto& assignment : *statement)
		{
			for (const auto& exp : *assignment.getDestinations()->get<Tuple>())
			{
				// From Function reference:
				// "Input formal parameters are read-only after being bound to the
				// actual arguments or default values, i.e., they may not be assigned
				// values in the body of the function."
				const auto* current = exp.get();

				while (current->isa<Operation>())
				{
					const auto* operation = current->get<Operation>();
					assert(operation->getOperationKind() == OperationKind::subscription);
					current = operation->getArg(0);
				}

				assert(current->isa<ReferenceAccess>());
				const auto* ref = current->get<ReferenceAccess>();

				if (!ref->isDummy())
				{
					const auto& name = ref->getName();

					if (symbolTable.count(name) == 0)
						return llvm::make_error<NotFound>(ref->getLocation(), name);

					const auto& member = symbolTable.lookup(name).get<Member>();

					if (member->isInput())
						return llvm::make_error<AssignmentToInputMember>(
								ref->getLocation(),
								function->getName());
				}
			}

			// From Function reference:
			// "A function cannot contain calls to the Modelica built-in operators
			// der, initial, terminal, sample, pre, edge, change, reinit, delay,
			// cardinality, inStream, actualStream, to the operators of the built-in
			// package Connections, and is not allowed to contain when-statements."

			std::stack<const Expression*> stack;
			stack.push(assignment.getExpression());

			while (!stack.empty())
			{
				const auto *expression = stack.top();
				stack.pop();

				if (expression->isa<ReferenceAccess>())
				{
					llvm::StringRef name = expression->get<ReferenceAccess>()->getName();

					if (name == "der" || name == "initial" || name == "terminal" ||
							name == "sample" || name == "pre" || name == "edge" ||
							name == "change" || name == "reinit" || name == "delay" ||
							name == "cardinality" || name == "inStream" ||
							name == "actualStream")
					{
						return llvm::make_error<BadSemantic>(
								expression->getLocation(),
								"'" + name.str() + "' is not allowed in procedural code");
					}

					// TODO: Connections built-in operators + when statement
				}
				else if (expression->isa<Operation>())
				{
					for (const auto& arg : *expression->get<Operation>())
						stack.push(arg.get());
				}
				else if (expression->isa<Call>())
				{
					const auto* call = expression->get<Call>();

					for (const auto& arg : *call)
						stack.push(arg.get());

					stack.push(call->getFunction());
				}
			}
		}
	}

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Model>(Class& cls)
{
	SymbolTable::ScopeTy varScope(symbolTable);
	auto* model = cls.get<Model>();

	// Populate the symbol table
	symbolTable.insert(model->getName(), Symbol(cls));

	for (auto& member : model->getMembers())
		symbolTable.insert(member->getName(), Symbol(*member));

	for (auto& m : model->getMembers())
		if (auto error = run(*m); error)
			return error;

	// Functions type checking must be done before the equations or algorithm
	// ones, because it establishes the result type of the functions that may
	// be invoked elsewhere.
	for (auto& innerClass : model->getInnerClasses())
		if (auto error = run<Class>(*innerClass); error)
			return error;

	for (auto& eq : model->getEquations())
		if (auto error = run(*eq); error)
			return error;

	for (auto& eq : model->getForEquations())
		if (auto error = run(*eq); error)
			return error;

	for (auto& algorithm : model->getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;

	if (auto error = resolveDummyReferences(*model); error)
		return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Package>(Class& cls)
{
	SymbolTable::ScopeTy varScope(symbolTable);
	auto* package = cls.get<Package>();

	// Populate the symbol table
	symbolTable.insert(package->getName(), Symbol(cls));

	for (auto& innerClass : *package)
		symbolTable.insert(innerClass->getName(), Symbol(*innerClass));

	for (auto& innerClass : *package)
		if (auto error = run<Class>(*innerClass); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Record>(Class& cls)
{
	SymbolTable::ScopeTy varScope(symbolTable);
	auto* record = cls.get<Record>();

	// Populate the symbol table
	symbolTable.insert(record->getName(), Symbol(cls));

	for (auto& member : *record)
		symbolTable.insert(member->getName(), Symbol(*member));

	for (auto& member : *record)
		if (auto error = run(*member); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Equation& equation)
{
	if (auto error = run<Expression>(*equation.getLhsExpression()); error)
		return error;

	if (auto error = run<Expression>(*equation.getRhsExpression()); error)
		return error;

	auto* lhs = equation.getLhsExpression();
	auto* rhs = equation.getRhsExpression();

	const auto& rhsType = rhs->getType();

	if (auto* lhsTuple = lhs->dyn_get<Tuple>())
	{
		if (!rhsType.isa<PackedType>() || lhsTuple->size() != rhsType.get<PackedType>().size())
			return llvm::make_error<IncompatibleType>(
					rhs->getLocation(),
					"number of results don't match with the destination tuple size");
	}

	if (auto* lhsTuple = lhs->dyn_get<Tuple>())
	{
		assert(rhs->getType().isa<PackedType>());
		auto& rhsTypes = rhs->getType().get<PackedType>();

		// Assign type to dummy variables.
		// The assignment can't be done earlier because the expression type would
		// have not been evaluated yet.

		for (size_t i = 0; i < lhsTuple->size(); ++i)
		{
			// If it's not a direct reference access, there's no way it can be a
			// dummy variable.

			if (!lhsTuple->getArg(i)->isa<ReferenceAccess>())
				continue;

			auto* ref = lhsTuple->getArg(i)->get<ReferenceAccess>();

			if (ref->isDummy())
			{
				assert(rhsTypes.size() >= i);
				lhsTuple->getArg(i)->setType(rhsTypes[i]);
			}
		}
	}

	// If the function call has more return values than the provided
	// destinations, then we need to add more dummy references.

	if (rhsType.isa<PackedType>())
	{
		const auto& rhsPackedType = rhsType.get<PackedType>();
		size_t returns = rhsPackedType.size();

		llvm::SmallVector<std::unique_ptr<Expression>, 3> newDestinations;
		llvm::SmallVector<Type, 3> destinationsTypes;

		if (auto* lhsTuple = lhs->dyn_get<Tuple>())
		{
			for (auto& destination : *lhsTuple)
			{
				destinationsTypes.push_back(destination->getType());
				newDestinations.push_back(std::move(destination));
			}
		}
		else
		{
			destinationsTypes.push_back(lhs->getType());
			newDestinations.push_back(lhs->clone());
		}

		for (size_t i = newDestinations.size(); i < returns; ++i)
		{
			destinationsTypes.push_back(rhsPackedType[i]);
			newDestinations.push_back(ReferenceAccess::dummy(equation.getLocation(), rhsPackedType[i]));
		}

		equation.setLhsExpression(
				Expression::tuple(lhs->getLocation(), Type(PackedType(destinationsTypes)), newDestinations));
	}

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(ForEquation& forEquation)
{
	SymbolTable::ScopeTy varScope(symbolTable);

	for (auto& ind : forEquation.getInductions())
	{
		symbolTable.insert(ind->getName(), Symbol(*ind));

		if (auto error = run<Expression>(*ind->getBegin()); error)
			return error;

		if (auto error = run<Expression>(*ind->getEnd()); error)
			return error;
	}

	if (auto error = run(*forEquation.getEquation()); error)
		return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Expression>(Expression& expression)
{
	return expression.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(expression);
	});
}

template<>
llvm::Error TypeChecker::run<Array>(Expression& expression)
{
	auto* array = expression.get<Array>();

	llvm::SmallVector<long, 3> sizes;

	auto resultType = makeType<bool>();

	for (auto& element : *array)
	{
		if (auto error = run<Expression>(*element); error)
			return error;

		auto& elementType = element->getType();
		assert(elementType.isNumeric());

		if (elementType >= resultType)
			resultType = elementType;

		unsigned int rank = elementType.dimensionsCount();

		if (!elementType.isScalar())
		{
			if (sizes.empty())
			{
				for (size_t i = 0; i < rank; ++i)
				{
					assert(!elementType[i].hasExpression());
					sizes.push_back(elementType[i].getNumericSize());
				}
			}
			else
			{
				assert(sizes.size() == rank);
			}
		}
	}

	llvm::SmallVector<ArrayDimension, 3> dimensions;
	dimensions.emplace_back(array->size());

	for (auto size : sizes)
		dimensions.emplace_back(size);

	resultType.setDimensions(dimensions);
	expression.setType(resultType);

	return llvm::Error::success();
}

static llvm::Expected<Type> getCallElementWiseResultType(
		llvm::ArrayRef<long> argsExpectedRanks, Call& call)
{
	llvm::SmallVector<ArrayDimension, 3> dimensions;

	for (const auto& arg : llvm::enumerate(call.getArgs()))
	{
		unsigned int argActualRank = arg.value()->getType().getRank();
		unsigned int argExpectedRank = argsExpectedRanks[arg.index()];

		if (arg.index() == 0)
		{
			// If this is the first argument, then it will determine the
			// rank and dimensions of the result array, although the dimensions
			// can be also specialized by the other arguments if initially unknown.

			for (size_t i = 0; i < argActualRank - argExpectedRank; ++i)
			{
				auto& dimension = arg.value()->getType()[arg.index()];
				dimensions.push_back(dimension.isDynamic() ? -1 : dimension.getNumericSize());
			}
		}
		else
		{
			// The rank difference must match with the one given by the first
			// argument, independently from the dimensions sizes.
			if (argActualRank != argExpectedRank + dimensions.size())
				return llvm::make_error<IncompatibleType>(
						arg.value()->getLocation(),
						"argument is incompatible with call vectorization (rank mismatch)");

			for (size_t i = 0; i < argActualRank - argExpectedRank; ++i)
			{
				auto& dimension = arg.value()->getType()[arg.index()];

				// If the dimension is dynamic, then no further checks or
				// specializations are possible.
				if (dimension.isDynamic())
					continue;

				// If the dimension determined by the first argument is fixed, then
				// also the dimension of the other arguments must match (when that's
				// fixed too).
				if (!dimensions[i].isDynamic() && dimensions[i] != dimension)
					return llvm::make_error<IncompatibleType>(
							arg.value()->getLocation(),
							"argument is incompatible with call vectorization (dimensions mismatch)");

				// If the dimension determined by the first argument is dynamic, then
				// set it to a required size.
				if (dimensions[i].isDynamic())
					dimensions[i] = dimension;
			}
		}
	}

	if (dimensions.empty())
		return call.getFunction()->getType();

	return call.getFunction()->getType().to(dimensions);
}

template<>
llvm::Error TypeChecker::run<Call>(Expression& expression)
{
	auto* call = expression.get<Call>();

	for (auto& arg : *call)
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto* function = call->getFunction();
	llvm::StringRef functionName = function->get<ReferenceAccess>()->getName();
	bool canBeCalledElementWise = true;
	llvm::SmallVector<long, 3> argsExpectedRanks;

	// If the function name refers to a built-in one, we also need to check
	// whether there is a user defined one that is shadowing it. If this is
	// not the case, then we can fallback to the real built-in function.

	if (symbolTable.count(functionName) == 0 &&
			builtInFunctions.count(functionName) != 0)
	{
		// Built-in function
		auto& builtInFunction = builtInFunctions[functionName];
		auto resultType = builtInFunction->resultType(call->getArgs());

		if (!resultType.hasValue())
			return llvm::make_error<BadSemantic>(call->getLocation(), "wrong number of arguments");

		function->setType(*resultType);
		canBeCalledElementWise = builtInFunction->canBeCalledElementWise();
		builtInFunction->getArgsExpectedRanks(call->argumentsCount(), argsExpectedRanks);
	}
	else
	{
		// User defined function

		if (auto error = run<Expression>(*function); error)
			return error;

		auto functionTypeResolver = [&](llvm::StringRef name) -> llvm::Optional<FunctionType> {
			if (symbolTable.count(name) == 0)
				return llvm::None;

			auto symbol = symbolTable.lookup(functionName);

			if (const auto* cls = symbol.dyn_get<Class>())
			{
				if (const auto* standardFunction = cls->dyn_get<StandardFunction>())
					return standardFunction->getType();

				if (const auto* partialDerFunction = cls->dyn_get<PartialDerFunction>())
					return partialDerFunction->getType();
			}

			return llvm::None;
		};

		auto functionType = functionTypeResolver(functionName);

		if (!functionType.hasValue())
			return llvm::make_error<NotFound>(function->getLocation(), functionName);

		for (const auto& type : functionType->getArgs())
			argsExpectedRanks.push_back(type.getRank());
	}

	if (canBeCalledElementWise)
	{
		auto type = getCallElementWiseResultType(argsExpectedRanks, *call);

		if (!type)
			return type.takeError();

		expression.setType(std::move(*type));
	}
	else
	{
		expression.setType(function->getType());
	}

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Constant>(Expression& expression)
{
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Operation>(Expression& expression)
{
	auto* operation = expression.get<Operation>();

	auto checkOperation = [&](Expression& expression, std::function<llvm::Error(TypeChecker&, Expression&)> checker) -> llvm::Error {
		if (auto error = checker(*this, expression); error)
			return error;

		return llvm::Error::success();
	};

	switch (operation->getOperationKind())
	{
		case OperationKind::add:
			return checkOperation(expression, &TypeChecker::checkAddOp);

		case OperationKind::different:
			return checkOperation(expression, &TypeChecker::checkDifferentOp);

		case OperationKind::divide:
			return checkOperation(expression, &TypeChecker::checkDivOp);

		case OperationKind::equal:
			return checkOperation(expression, &TypeChecker::checkEqualOp);

		case OperationKind::greater:
			return checkOperation(expression, &TypeChecker::checkGreaterOp);

		case OperationKind::greaterEqual:
			return checkOperation(expression, &TypeChecker::checkGreaterEqualOp);

		case OperationKind::ifelse:
			return checkOperation(expression, &TypeChecker::checkIfElseOp);

		case OperationKind::less:
			return checkOperation(expression, &TypeChecker::checkLessOp);

		case OperationKind::lessEqual:
			return checkOperation(expression, &TypeChecker::checkLessEqualOp);

		case OperationKind::land:
			return checkOperation(expression, &TypeChecker::checkLogicalAndOp);

		case OperationKind::lor:
			return checkOperation(expression, &TypeChecker::checkLogicalOrOp);

		case OperationKind::memberLookup:
			return checkOperation(expression, &TypeChecker::checkMemberLookupOp);

		case OperationKind::multiply:
			return checkOperation(expression, &TypeChecker::checkMulOp);

		case OperationKind::negate:
			return checkOperation(expression, &TypeChecker::checkNegateOp);

		case OperationKind::powerOf:
			return checkOperation(expression, &TypeChecker::checkPowerOfOp);

		case OperationKind::subscription:
			return checkOperation(expression, &TypeChecker::checkSubscriptionOp);

		case OperationKind::subtract:
			return checkOperation(expression, &TypeChecker::checkSubOp);
	}

	return llvm::Error::success();
}

static llvm::Optional<Type> builtInReferenceType(ReferenceAccess& reference)
{
	assert(!reference.isDummy());
	auto name = reference.getName();

	if (name == "time")
		return makeType<float>();

	return llvm::None;
}

template<>
llvm::Error TypeChecker::run<ReferenceAccess>(Expression& expression)
{
	auto* reference = expression.get<ReferenceAccess>();

	// If the referenced variable is a dummy one (meaning that it is created
	// to store a result value that will never be used), its type is still
	// unknown and will be determined according to the assigned value.

	if (reference->isDummy())
		return llvm::Error::success();

	auto name = reference->getName();

	if (symbolTable.count(name) == 0)
	{
		if (auto type = builtInReferenceType(*reference); type.hasValue())
		{
			expression.setType(type.getValue());
			return llvm::Error::success();
		}

		return llvm::make_error<NotFound>(reference->getLocation(), name);
	}

	auto symbol = symbolTable.lookup(name);

	auto symbolType = [](Symbol& symbol) -> Type {
		if (auto* cls = symbol.dyn_get<Class>(); cls != nullptr && cls->isa<StandardFunction>())
			return cls->get<StandardFunction>()->getType().packResults();

		if (auto* cls = symbol.dyn_get<Class>(); cls != nullptr && cls->isa<PartialDerFunction>())
		{
			auto types = cls->get<PartialDerFunction>()->getResultsTypes();

			if (types.size() == 1)
				return types[0];

			return Type(PackedType(types));
		}

		if (symbol.isa<Member>())
			return symbol.get<Member>()->getType();

		if (symbol.isa<Induction>())
			return makeType<int>();

		assert(false && "Unexpected symbol type");
		return Type::unknown();
	};

	expression.setType(symbolType(symbol));
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Tuple>(Expression& expression)
{
	auto* tuple = expression.get<Tuple>();
	llvm::SmallVector<Type, 3> types;

	for (auto& exp : *tuple)
	{
		if (auto error = run<Expression>(*exp); error)
			return error;

		types.push_back(exp->getType());
	}

	expression.setType(Type(PackedType(types)));
	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Member& member)
{
	for (auto& dimension : member.getType().getDimensions())
		if (dimension.hasExpression())
			if (auto error = run<Expression>(*dimension.getExpression()); error)
				return error;

	if (member.hasInitializer())
		if (auto error = run<Expression>(*member.getInitializer()); error)
			return error;

	if (member.hasStartOverload())
		if (auto error = run<Expression>(*member.getStartOverload()); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Statement>(Statement& statement)
{
	return statement.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(statement);
	});
}

template<>
llvm::Error TypeChecker::run<AssignmentStatement>(Statement& statement)
{
	auto* assignmentStatement = statement.get<AssignmentStatement>();

	auto* destinations = assignmentStatement->getDestinations();
	auto* destinationsTuple = destinations->get<Tuple>();

	for (auto& destination : *destinationsTuple)
	{
		if (auto error = run<Expression>(*destination); error)
			return error;

		// The destinations must be l-values.
		// The check can't be enforced at parsing time because the grammar
		// specifies the destinations as expressions.

		if (!destination->isLValue())
			return llvm::make_error<BadSemantic>(
					destination->getLocation(),
					"Destinations of statements must be l-values");
	}

	auto* expression = assignmentStatement->getExpression();

	if (auto error = run<Expression>(*expression); error)
		return error;

	if (destinationsTuple->size() > 1 && !expression->getType().isa<PackedType>())
		return llvm::make_error<IncompatibleType>(
				expression->getLocation(),
				"The expression must return at least " +
				std::to_string(destinationsTuple->size()) + "values");

	// Assign type to dummy variables.
	// The assignment can't be done earlier because the expression type would
	// have not been evaluated yet.

	for (size_t i = 0, e = destinationsTuple->size(); i < e; ++i)
	{
		// If it's not a direct reference access, there's no way it can be a
		// dummy variable.
		if (!destinationsTuple->getArg(i)->isa<ReferenceAccess>())
			continue;

		auto* reference = destinationsTuple->getArg(i)->get<ReferenceAccess>();

		if (reference->isDummy())
		{
			auto& expressionType = expression->getType();
			assert(expressionType.isa<PackedType>());
			auto& packedType = expressionType.get<PackedType>();
			assert(packedType.size() >= i);
			destinationsTuple->getArg(i)->setType(packedType[i]);
		}
	}

	// If the function call has more return values than the provided
	// destinations, then we need to add more dummy references.

	if (expression->getType().isa<PackedType>())
	{
		auto& packedType = expression->getType().get<PackedType>();
		size_t returns = packedType.size();

		if (destinationsTuple->size() < returns)
		{
			llvm::SmallVector<std::unique_ptr<Expression>, 3> newDestinations;
			llvm::SmallVector<Type, 3> destinationsTypes;

			for (auto& destination : *destinationsTuple)
			{
				destinationsTypes.push_back(destination->getType());
				newDestinations.push_back(std::move(destination));
			}

			for (size_t i = newDestinations.size(); i < returns; ++i)
			{
				destinationsTypes.push_back(packedType[i]);
				newDestinations.emplace_back(ReferenceAccess::dummy(statement.getLocation(), packedType[i]));
			}

			assignmentStatement->setDestinations(
					Expression::tuple(destinations->getLocation(), Type(PackedType(destinationsTypes)), std::move(newDestinations)));
		}
	}

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<BreakStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<ForStatement>(Statement& statement)
{
	auto* forStatement = statement.get<ForStatement>();

	if (auto error = run(*forStatement->getInduction()); error)
		return error;

	auto* induction = forStatement->getInduction();
	symbolTable.insert(induction->getName(), Symbol(*induction));

	for (auto& stmnt : forStatement->getBody())
		if (auto error = run<Statement>(*stmnt); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Induction& induction)
{
	if (auto error = run<Expression>(*induction.getBegin()); error)
		return error;

	if (!induction.getBegin()->getType().isa<int>())
		return llvm::make_error<IncompatibleType>(
				induction.getBegin()->getLocation(), "start value must be an integer");

	if (auto error = run<Expression>(*induction.getEnd()); error)
		return error;

	if (!induction.getEnd()->getType().isa<int>())
		return llvm::make_error<IncompatibleType>(
				induction.getBegin()->getLocation(), "end value must be an integer");

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<IfStatement>(Statement& statement)
{
	auto* ifStatement = statement.get<IfStatement>();

	for (auto& block : *ifStatement)
	{
		if (auto error = run<Expression>(*block.getCondition()); error)
			return error;

		if (!block.getCondition()->getType().isa<bool>())
			return llvm::make_error<IncompatibleType>(
					block.getCondition()->getLocation(), "condition must be a boolean");

		for (auto& stmnt : block)
			if (auto error = run<Statement>(*stmnt); error)
				return error;
	}

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<ReturnStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<WhenStatement>(Statement& statement)
{
	auto* whenStatement = statement.get<WhenStatement>();

	if (auto error = run<Expression>(*whenStatement->getCondition()); error)
		return error;

	if (!whenStatement->getCondition()->getType().isa<bool>())
		return llvm::make_error<IncompatibleType>(
				whenStatement->getCondition()->getLocation(), "condition must be a boolean");

	for (auto& stmnt : whenStatement->getBody())
		if (auto error = run<Statement>(*stmnt); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<WhileStatement>(Statement& statement)
{
	auto* whileStatement = statement.get<WhileStatement>();

	if (auto error = run<Expression>(*whileStatement->getCondition()); error)
		return error;

	if (!whileStatement->getCondition()->getType().isa<bool>())
		return llvm::make_error<IncompatibleType>(
				whileStatement->getCondition()->getLocation(), "condition must be a boolean");

	for (auto& stmnt : whileStatement->getBody())
		if (auto error = run<Statement>(*stmnt); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Algorithm& algorithm)
{
	for (auto& statement : algorithm)
		if (auto error = run<Statement>(*statement); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::checkGenericOperation(Expression& expression)
{
	auto* operation = expression.get<Operation>();

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	Type type = expression.getType();

	for (auto& arg : operation->getArguments())
		if (auto& argType = arg->getType(); argType >= type)
			type = argType;

	expression.setType(std::move(type));
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkAddOp(Expression& expression)
{
	return checkGenericOperation(expression);
}

llvm::Error TypeChecker::checkDifferentOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::different);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

/**
 * Get the type resulting from the division of two types.
 *
 * @param x  	 first type
 * @param y	 	 second type
 * @param loc	 second type location
 * @return result type
 */
static llvm::Expected<Type> divPairResultType(Type x, Type y, SourceRange loc)
{
	if (x.isScalar() && y.isScalar())
		return makeType<float>();

	if (x.getRank() == 1 && y.isScalar())
		return x.to(BuiltInType::Float);

	return llvm::make_error<IncompatibleTypes>(loc, x, y);
}

llvm::Error TypeChecker::checkDivOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::divide);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	assert(operation->getArguments().size() >= 2);
	Type resultType = operation->getArg(0)->getType();

	for (size_t i = 1, end = operation->argumentsCount(); i < end; ++i)
	{
		auto* arg = operation->getArg(i);
		auto pairResultType = divPairResultType(resultType, arg->getType(), arg->getLocation());

		if (!pairResultType)
			return pairResultType.takeError();

		resultType = std::move(*pairResultType);
	}

	expression.setType(resultType);
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkEqualOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::equal);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkGreaterOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::greater);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkGreaterEqualOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::greaterEqual);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkIfElseOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::ifelse);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto* condition = operation->getArg(0);
	auto* trueValue = operation->getArg(1);
	auto* falseValue = operation->getArg(2);

	if (condition->getType() != makeType<bool>())
		return llvm::make_error<IncompatibleType>(
				condition->getLocation(),
				"condition must be a boolean value");

	if (trueValue->getType() != falseValue->getType())
		return llvm::make_error<IncompatibleType>(
				falseValue->getLocation(),
				"ternary operator values must have the same type");

	expression.setType(operation->getArg(1)->getType());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkLessOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::less);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkLessEqualOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::lessEqual);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkLogicalAndOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::land);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	llvm::SmallVector<ArrayDimension, 3> dimensions;

	assert(operation->argumentsCount() == 2);
	auto* lhs = operation->getArg(0);
	auto* rhs = operation->getArg(1);

	auto& lhsType = lhs->getType();
	auto& rhsType = rhs->getType();

	// The arguments must be booleans or array of booleans
	if (!lhsType.isa<bool>())
		return llvm::make_error<IncompatibleType>(lhs->getLocation(), "argument must be a boolean or an array of booleans");

	if (!rhsType.isa<bool>())
		return llvm::make_error<IncompatibleType>(rhs->getLocation(), "argument must be a boolean or an array of booleans");

	// The ranks must match
	if (lhsType.getRank() != rhsType.getRank())
		return llvm::make_error<IncompatibleTypes>(operation->getLocation(), lhsType, rhsType);

	// If the arguments are arrays, then also their dimensions must match
	for (const auto& [l, r] : llvm::zip(lhsType.getDimensions(), rhsType.getDimensions()))
	{
		long dimension = -1;

		if (!l.isDynamic() && !r.isDynamic())
			if (l.getNumericSize() != r.getNumericSize())
				return llvm::make_error<IncompatibleTypes>(operation->getLocation(), lhsType, rhsType);

		if (!l.isDynamic())
			dimension = l.getNumericSize();
		else if (!r.isDynamic())
			dimension = r.getNumericSize();

		dimensions.push_back(dimension);
	}

	expression.setType(Type(BuiltInType::Boolean, dimensions));
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkLogicalOrOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::lor);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	llvm::SmallVector<ArrayDimension, 3> dimensions;

	assert(operation->argumentsCount() == 2);
	auto* lhs = operation->getArg(0);
	auto* rhs = operation->getArg(1);

	auto& lhsType = lhs->getType();
	auto& rhsType = rhs->getType();

	// The arguments must be booleans or array of booleans
	if (!lhsType.isa<bool>())
		return llvm::make_error<IncompatibleType>(lhs->getLocation(), "argument must be a boolean or an array of booleans");

	if (!rhsType.isa<bool>())
		return llvm::make_error<IncompatibleType>(rhs->getLocation(), "argument must be a boolean or an array of booleans");

	// The ranks must match
	if (lhsType.getRank() != rhsType.getRank())
		return llvm::make_error<IncompatibleTypes>(operation->getLocation(), lhsType, rhsType);

	// If the arguments are arrays, then also their dimensions must match
	for (const auto& [l, r] : llvm::zip(lhsType.getDimensions(), rhsType.getDimensions()))
	{
		long dimension = -1;

		if (!l.isDynamic() && !r.isDynamic())
			if (l.getNumericSize() != r.getNumericSize())
				return llvm::make_error<IncompatibleTypes>(operation->getLocation(), lhsType, rhsType);

		if (!l.isDynamic())
			dimension = l.getNumericSize();
		else if (!r.isDynamic())
			dimension = r.getNumericSize();

		dimensions.push_back(dimension);
	}

	expression.setType(Type(BuiltInType::Boolean, dimensions));
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkMemberLookupOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::memberLookup);
	return llvm::make_error<NotImplemented>("member lookup is not implemented yet");
}

/**
 * Get the type resulting from the multiplication of two types.
 *
 * @param x  	 first type
 * @param y	 	 second type
 * @param loc	 second type location
 * @return result type
 */
static llvm::Expected<Type> mulPairResultType(Type x, Type y, SourceRange loc)
{
	if (x.isScalar())
		return y.to(getMostGenericBaseType(x, y));

	if (y.isScalar())
		return x.to(getMostGenericBaseType(x, y));

	if (x.getRank() == 1 && y.getRank() == 1)
	{
		auto baseType = getMostGenericBaseType(x, y);

		if (!x[0].isDynamic() && !y[0].isDynamic())
			if (x[0].getNumericSize() != y[0].getNumericSize())
				return llvm::make_error<IncompatibleTypes>(loc, x, y);

		return Type(baseType);
	}

	if (x.getRank() == 1 && y.getRank() == 2)
	{
		auto baseType = getMostGenericBaseType(x, y);

		if (!x[0].isDynamic() && !y[0].isDynamic())
			if (x[0].getNumericSize() != y[0].getNumericSize())
				return llvm::make_error<IncompatibleTypes>(loc, x, y);

		llvm::SmallVector<ArrayDimension, 1> dimensions;
		dimensions.emplace_back(y[1].isDynamic() ? -1 : y[1].getNumericSize());
		return Type(baseType, dimensions);
	}

	if (x.getRank() == 2 && y.getRank() == 1)
	{
		auto baseType = getMostGenericBaseType(x, y);

		if (!x[1].isDynamic() && !y[0].isDynamic())
			if (x[1].getNumericSize() != y[0].getNumericSize())
				return llvm::make_error<IncompatibleTypes>(loc, x, y);

		llvm::SmallVector<ArrayDimension, 1> dimensions;
		dimensions.emplace_back(x[0].isDynamic() ? -1 : x[0].getNumericSize());
		return Type(baseType, dimensions);
	}

	if (x.getRank() == 2 && y.getRank() == 2)
	{
		auto baseType = getMostGenericBaseType(x, y);

		if (!x[1].isDynamic() && !y[0].isDynamic())
			if (x[1].getNumericSize() != y[0].getNumericSize())
				return llvm::make_error<IncompatibleTypes>(loc, x, y);

		llvm::SmallVector<ArrayDimension, 2> dimensions;
		dimensions.emplace_back(x[0].isDynamic() ? -1 : x[0].getNumericSize());
		dimensions.emplace_back(y[1].isDynamic() ? -1 : y[1].getNumericSize());
		return Type(baseType, dimensions);
	}

	return llvm::make_error<IncompatibleTypes>(loc, x, y);
}

llvm::Error TypeChecker::checkMulOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::multiply);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	assert(operation->getArguments().size() >= 2);
	Type resultType = operation->getArg(0)->getType();

	for (size_t i = 1, end = operation->getArguments().size(); i < end; ++i)
	{
		auto* arg = operation->getArg(i);
		auto pairResultType = mulPairResultType(resultType, arg->getType(), arg->getLocation());

		if (!pairResultType)
			return pairResultType.takeError();

		resultType = std::move(*pairResultType);
	}

	expression.setType(resultType);
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkNegateOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::negate);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(operation->getArg(0)->getType());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkPowerOfOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::powerOf);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto* base = operation->getArg(0);

	if (auto baseType = base->getType(); baseType.getRank() == 0)
	{
		if (!baseType.isNumeric())
			return llvm::make_error<IncompatibleType>(
					base->getLocation(), "base must be a numeric value");
	}
	else
	{
		if (baseType.getRank() == 2)
		{
			if (!baseType[0].isDynamic() && !baseType[1].isDynamic())
				if (baseType[0] != baseType[1])
					return llvm::make_error<IncompatibleType>(
							base->getLocation(), "base must be a scalar or a square matrix");
		}
		else
		{
			return llvm::make_error<IncompatibleType>(
					base->getLocation(), "base must be a scalar or a square matrix");
		}
	}

	auto* exponent = operation->getArg(1);

	if (exponent->getType().getRank() != 0)
		return llvm::make_error<IncompatibleType>(
				exponent->getLocation(), "the exponent must be a scalar value");

	expression.setType(base->getType());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkSubOp(Expression& expression)
{
	return checkGenericOperation(expression);
}

llvm::Error TypeChecker::checkSubscriptionOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::subscription);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto* source = operation->getArg(0);
	size_t subscriptionIndexesCount = operation->argumentsCount() - 1;

	if (subscriptionIndexesCount > source->getType().dimensionsCount())
		return llvm::make_error<BadSemantic>(
				operation->getLocation(), "too many subscriptions");

	for (size_t i = 1; i < operation->argumentsCount(); ++i)
		if (auto* index = operation->getArg(i); index->getType() != makeType<int>())
			return llvm::make_error<BadSemantic>(
					index->getLocation(), "index expression must be an integer");

	expression.setType(source->getType().subscript(subscriptionIndexesCount));
	return llvm::Error::success();
}

template<class T>
std::string getTemporaryVariableName(T& cls)
{
	const auto& members = cls.getMembers();
	int counter = 0;

	while (*(members.end()) !=
				 *find_if(members.begin(), members.end(), [=](const auto& obj) {
					 return obj->getName() == "_temp" + std::to_string(counter);
				 }))
		counter++;

	return "_temp" + std::to_string(counter);
}

llvm::Error resolveDummyReferences(StandardFunction& function)
{
	for (auto& algorithm : function.getAlgorithms())
	{
		for (auto& statement : algorithm->getBody())
		{
			for (auto& assignment : *statement)
			{
				for (auto& destination : *assignment.getDestinations()->get<Tuple>())
				{
					if (!destination->isa<ReferenceAccess>())
						continue;

					auto* ref = destination->get<ReferenceAccess>();

					if (!ref->isDummy())
						continue;

					std::string name = getTemporaryVariableName(function);
					auto temp = Member::build(destination->getLocation(), name, destination->getType(), TypePrefix::none(), llvm::None);
					ref->setName(temp->getName());
					function.addMember(std::move(temp));

					// Note that there is no need to add the dummy variable to the
					// symbol table, because it will never be referenced.
				}
			}
		}
	}

	return llvm::Error::success();
}

llvm::Error resolveDummyReferences(Model& model)
{
	for (auto& equation : model.getEquations())
	{
		auto* lhs = equation->getLhsExpression();

		if (auto* lhsTuple = lhs->dyn_get<Tuple>())
		{
			for (auto& expression : *lhsTuple)
			{
				if (!expression->isa<ReferenceAccess>())
					continue;

				auto* ref = expression->get<ReferenceAccess>();

				if (!ref->isDummy())
					continue;

				std::string name = getTemporaryVariableName(model);
				auto temp = Member::build(expression->getLocation(), name, expression->getType(), TypePrefix::none(), llvm::None);
				ref->setName(temp->getName());
				model.addMember(std::move(temp));
			}
		}
	}

	for (auto& forEquation : model.getForEquations())
	{
		auto* equation = forEquation->getEquation();
		auto* lhs = equation->getLhsExpression();

		if (auto* lhsTuple = lhs->dyn_get<Tuple>())
		{
			for (auto& expression : *lhsTuple)
			{
				if (!expression->isa<ReferenceAccess>())
					continue;

				auto* ref = expression->get<ReferenceAccess>();

				if (!ref->isDummy())
					continue;

				std::string name = getTemporaryVariableName(model);
				auto temp = Member::build(expression->getLocation(), name, expression->getType(), TypePrefix::none(), llvm::None);
				ref->setName(temp->getName());
				model.addMember(std::move(temp));
			}
		}
	}

	for (auto& algorithm : model.getAlgorithms())
	{
		for (auto& statement : algorithm->getBody())
		{
			for (auto& assignment : *statement)
			{
				for (auto& destination : *assignment.getDestinations()->get<Tuple>())
				{
					if (!destination->isa<ReferenceAccess>())
						continue;

					auto* ref = destination->get<ReferenceAccess>();

					if (!ref->isDummy())
						continue;

					std::string name = getTemporaryVariableName(model);
					auto temp = Member::build(destination->getLocation(), name, destination->getType(), TypePrefix::none(), llvm::None);
					ref->setName(temp->getName());
					model.addMember(std::move(temp));

					// Note that there is no need to add the dummy variable to the
					// symbol table, because it will never be referenced.
				}
			}
		}
	}

	return llvm::Error::success();
}

std::unique_ptr<Pass> marco::frontend::createTypeCheckingPass()
{
	return std::make_unique<TypeChecker>();
}

namespace marco::frontend::detail
{
	TypeCheckingErrorCategory TypeCheckingErrorCategory::category;

	std::error_condition TypeCheckingErrorCategory::default_error_condition(int ev) const noexcept
	{
		if (ev == 1)
			return std::error_condition(TypeCheckingErrorCode::bad_semantic);

		if (ev == 2)
			return std::error_condition(TypeCheckingErrorCode::not_found);

		return std::error_condition(TypeCheckingErrorCode::success);
	}

	bool TypeCheckingErrorCategory::equivalent(const std::error_code& code, int condition) const noexcept
	{
		bool equal = *this == code.category();
		auto v = default_error_condition(code.value()).value();
		equal = equal && static_cast<int>(v) == condition;
		return equal;
	}

	std::string TypeCheckingErrorCategory::message(int ev) const noexcept
	{
		switch (ev)
		{
			case (0):
				return "Success";

			case (1):
				return "Assignment to input member";

			case (2):
				return "Bad semantic";

			case (3):
				return "Incompatible type";

			case (4):
				return "Incompatible types";

			case (5):
				return "Multiple algorithms";

			case (6):
				return "Not found";

			default:
				return "Unknown Error";
		}
	}

	std::error_condition make_error_condition(TypeCheckingErrorCode errc)
	{
		return std::error_condition(
				static_cast<int>(errc), TypeCheckingErrorCategory::category);
	}
}

char AssignmentToInputMember::ID;
char BadSemantic::ID;
char IncompatibleType::ID;
char IncompatibleTypes::ID;
char MultipleAlgorithmsFunction::ID;
char NotFound::ID;

AssignmentToInputMember::AssignmentToInputMember(SourceRange location, llvm::StringRef className)
		: location(std::move(location)),
			className(className.str())
{
}

SourceRange AssignmentToInputMember::getLocation() const
{
	return location;
}

bool AssignmentToInputMember::printBeforeMessage(llvm::raw_ostream& os) const
{
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << *location.fileName << ": ";
	os.resetColor();
	os << "in class \"";
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << className;
	os.resetColor();
	os << "\"";

	return true;
}

void AssignmentToInputMember::printMessage(llvm::raw_ostream& os) const
{
	os << "input member can't receive a new value";
}

void AssignmentToInputMember::log(llvm::raw_ostream& os) const
{
	print(os);
}

BadSemantic::BadSemantic(SourceRange location, llvm::StringRef message)
		: location(std::move(location)),
			message(message.str())
{
}

SourceRange BadSemantic::getLocation() const
{
	return location;
}

void BadSemantic::printMessage(llvm::raw_ostream& os) const
{
	os << message;
}

void BadSemantic::log(llvm::raw_ostream& os) const
{
	print(os);
}

IncompatibleType::IncompatibleType(SourceRange location, llvm::StringRef message)
		: location(std::move(location)),
			message(message.str())
{
}

SourceRange IncompatibleType::getLocation() const
{
	return location;
}

void IncompatibleType::printMessage(llvm::raw_ostream& os) const
{
	os << message;
}

void IncompatibleType::log(llvm::raw_ostream& os) const
{
	print(os);
}

IncompatibleTypes::IncompatibleTypes(SourceRange location, Type first, Type second)
		: location(std::move(location)),
			first(std::move(first)),
			second(std::move(second))
{
}

SourceRange IncompatibleTypes::getLocation() const
{
	return location;
}

void IncompatibleTypes::printMessage(llvm::raw_ostream& os) const
{
	os << "incompatible types: \"";
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << first;
	os.resetColor();
	os << "\" and \"";
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << second;
	os.resetColor();
	os << "\"";
}

void IncompatibleTypes::log(llvm::raw_ostream& os) const
{
	print(os);
}

MultipleAlgorithmsFunction::MultipleAlgorithmsFunction(SourceRange location, llvm::StringRef functionName)
		: location(std::move(location)),
			functionName(functionName.str())
{
}

SourceRange MultipleAlgorithmsFunction::getLocation() const
{
	return location;
}

bool MultipleAlgorithmsFunction::printBeforeMessage(llvm::raw_ostream& os) const
{
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << *location.fileName << ": ";
	os.resetColor();
	os << "in function \"";
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << functionName;
	os.resetColor();
	os << "\"";

	return true;
}

void MultipleAlgorithmsFunction::printMessage(llvm::raw_ostream& os) const
{
	os << "functions can have at most one algorithm section";
}

void MultipleAlgorithmsFunction::log(llvm::raw_ostream& os) const
{
	print(os);
}

NotFound::NotFound(SourceRange location, llvm::StringRef variableName)
		: location(std::move(location)),
			variableName(variableName.str())
{
}

SourceRange NotFound::getLocation() const
{
	return location;
}

void NotFound::printMessage(llvm::raw_ostream& os) const
{
	os << "unknown identifier \"";
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << variableName;
	os.resetColor();
	os << "\"";
}

void NotFound::log(llvm::raw_ostream& os) const
{
	print(os);
}
