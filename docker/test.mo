// 1. DEFINIZIONE DELLA FUNZIONE
function factorial
  input Integer n;
  output Integer res;
algorithm
  res := res * 2;
end factorial;


// 2. DEFINIZIONE DEL MODELLO CHE USA LA FUNZIONE
model FactorialGrowth
  // Parametro intero, il cui fattoriale controllerà la crescita
  parameter Integer growthFactor = 4;
  
  // La variabile di stato del nostro sistema
  Real y(start = 0, fixed = true);

equation
  // L'equazione differenziale che governa il sistema
  // La derivata di y è costante e pari a factorial(4), cioè 24.
  der(y) = factorial(growthFactor);

end FactorialGrowth;