#include <gtest/gtest.h>
#include <marco/ast/AST.h>
#include <marco/ast/Parser.h>
#include <marco/ast/Passes.h>

using namespace marco;
using namespace ast;
using namespace std;

const std::string common_code =R"(
record Complex
  input Real re;
  input Real im;
end Complex;

function Complex.'*'.multiply "Multiply two complex numbers"
  input Complex c1 "Complex number 1";
  input Complex c2 "Complex number 2";
  output Complex c3 "= c1*c2";
algorithm
  c3 := Complex.'constructor'.fromReal(c1.re * c2.re - c1.im * c2.im, c1.re * c2.im + c1.im * c2.re);
  annotation(Inline = true);
end Complex.'*'.multiply;

function Complex.'constructor'.fromReal "Construct Complex from Real"
  input Real re "Real part of complex number";
  input Real im = 0.0 "Imaginary part of complex number";
  output Complex result "Complex number";
algorithm
  annotation(Inline = true);
end Complex.'constructor'.fromReal;

record Nested
  input Complex a;
  input Complex[2] b;
  input Real re;
end Nested;
)";

// inspired by https://en.cppreference.com/w/cpp/utility/variant/visit method #4 
// usage:
// (ASTConstVisitor{
// 	[&](const Model &m){
// 		...
// 	},
//	... 
// 	[&](const auto &c){} always include default callback
// }).visit(*ast);
template<class... Ts> struct ASTConstVisitor : Ts... { using Ts::operator()...; 
	template<typename ...Args>
	void visit(const std::unique_ptr<Class> &c, Args ...args){
		visit(*c,args...);
	}
	template<typename ...Args>
	void visit(const Class &c, Args ...args){
		c.visit([&](const auto &obj){
			return visit(obj,args...);
		});
	}
	template<typename T,typename ...Args>
	void visit(const T &t, Args ...args){
		(*this)(t,args...);
	}
	template<typename ...Args>
	void visit(const Package& p, Args ...args){
		(*this)(p,args...);
		for(const auto &c : p.getInnerClasses())
			visit(*c,args...);
	}
	template<typename ...Args>
	void visit(const Model& m, Args ...args){
		(*this)(m,args...);
		for(const auto &member : m.getMembers())
			visit(*member,args...);
		for(const auto &a : m.getAlgorithms())
			visit(*a,args...);
		for(const auto &e : m.getEquations())
			visit(*e,args...);
		for(const auto &e : m.getForEquations())
			visit(*e,args...);
		for(const auto &c : m.getInnerClasses())
			visit(*c,args...);
	}
	template<typename ...Args>
	void visit(const Statement& s, Args ...args){
		(*this)(s,args...);
		s.visit([&](const auto& obj) {
			visit(obj,args...);
			for(const auto &e : obj)
				visit(*e,args...);
		});
	}
	template<typename ...Args>
	void visit(const Equation& e, Args ...args){
		(*this)(e,args...);
		visit(*e.getRhsExpression(),args...);
		visit(*e.getLhsExpression(),args...);
	}
	template<typename ...Args>
	void visit(const ForEquation& f, Args ...args){
		(*this)(f,args...);
		for(const auto& i: f.getInductions())
			visit(*i,args...);
		visit(*f.getEquation(),args...);
	}
	template<typename ...Args>
	void visit(const Induction& i, Args ...args){
		(*this)(i,args...);
		visit(*i.getBegin(),args...);
		visit(*i.getEnd(),args...);
	}
	template<typename ...Args>
	void visit(const Expression& e, Args ...args){
		(*this)(e,args...);
		e.visit([&](const auto& obj) {
			visit(obj,args...);
		});
	}
	template<typename ...Args>
	void visit(const Operation& o, Args ...args){
		(*this)(o,args...);
		for(const auto &a : o)
			visit(*a,args...);
	} 
	template<typename ...Args>
	void visit(const Call& c, Args ...args){
		(*this)(c,args...);
		for(const auto &a : c)
			visit(*a,args...);
	} 
};
// explicit deduction guide (not needed as of C++20)
template<class... Ts> ASTConstVisitor(Ts...) -> ASTConstVisitor<Ts...>;

std::unique_ptr<Class> parseString(std::string s, bool constantFolding=false){
	Parser parser(s);
	
	auto ast = parser.classDefinition();

	if (!ast || !*ast)
		return nullptr;

	TypeChecker checker;
	InlineExpanser inliner;

	if (checker.run(*ast) || inliner.run(*ast))
		return nullptr;

	if(constantFolding){
		ConstantFolder folder;
		if(folder.run(*ast))
			return nullptr;
	}

	return std::move(*ast);
}

void countOccurences(const std::unique_ptr<Class> &ast, int &models, int &members, int &records,int &equations,int &fors){
	(ASTConstVisitor{
		[&](const Model &m){
			models++;
		},
		[&](const Member &m){
			members++;
		},
		[&](const Expression &e){
			EXPECT_FALSE(e.getType().isa<Record*>());
			EXPECT_FALSE(e.getType().isa<UserDefinedType>());
			EXPECT_FALSE(e.isa<RecordInstance>());
		},
		[&](const Record &r){
			records++;
		},	
		[&](const Equation &e){
			equations++;
		},
		[&](const ForEquation &e){
			fors++;
		},
		[](const auto &c){}
	}).visit(*ast);
}

TEST(InlinerTest, recordMemberExplosion)	 // NOLINT
{
	auto ast = parseString(R"(
		record Complex
			input Real re;
			input Real im;
		end Complex;

		class A
			output Complex y;
		equation
			y = Complex(2,3);
		end A;
	)");

	if(!ast)
		FAIL();

	int models=0, members=0, records=0, equations=0, fors=0;
	
	countOccurences(ast,models,members,records,equations,fors);

	EXPECT_EQ(models,1);
	EXPECT_EQ(members,2);
	EXPECT_EQ(records,1);
	EXPECT_EQ(equations,2);
	EXPECT_EQ(fors,0);
}

TEST(InlinerTest, withArray)	 // NOLINT
{
	//testing :  y = r[1] * Complex(r[1].re,r[2].im); 
	auto ast = parseString(
	common_code +
	R"(
		class A
			input Complex[2] r;
			output Complex y;
		equation
  			y = Complex.'*'.multiply( r[1], Complex.'constructor'.fromReal(r.re[1], r[2].im) ); // robust to different subscription positions
		end A;
	)");

	if(!ast)
		FAIL();

	int models=0, members=0, records=0, equations=0, fors=0;

	countOccurences(ast,models,members,records,equations,fors);

	EXPECT_EQ(models,1);
	EXPECT_EQ(members,4);
	EXPECT_EQ(records,2);
	EXPECT_EQ(equations,2);
	EXPECT_EQ(fors,0);
}

TEST(InlinerTest, NestedRecordsAndArrays)	 // NOLINT
{
	//testing :  y = x * 2 * r[1] * Complex(r[2].re,n[1].a.im) * n[2].b[1] ; 
	auto ast = parseString(
	common_code +
	R"(
		class A
			parameter Complex p = Complex(1.0, 2.0);
			input Complex[2] r;
			input Nested[3] n;
			input Complex x;
			output Complex y;
			output Complex[3] z;
		equation
  			y = Complex.'*'.multiply(Complex.'*'.multiply(Complex.'*'.multiply(Complex.'*'.multiply(x, Complex(2.0, 0.0)), r[1]), Complex.'constructor'.fromReal(r.re[2], n.a.im[1])), n.b[2,1]);
			for i in 1:3 loop
				z[i] = n[i].a; 
			end for;
		end A;
	)");

	if(!ast)
		FAIL();
	
	int models=0, members=0, records=0, equations=0, fors=0;

	countOccurences(ast,models,members,records,equations,fors);

	EXPECT_EQ(models,1);
	EXPECT_EQ(members,10 + 5);
	EXPECT_EQ(records,2);
	EXPECT_EQ(equations,4);
	EXPECT_EQ(fors,2);
}
