function foo
	input Real x;
	output Real y;
algorithm
	y := x + sin(x);
end foo;

model CallOp
	Real x;
	Real y;
equation
	der(x) = 2;
	der(y) = foo(x);
end CallOp;
