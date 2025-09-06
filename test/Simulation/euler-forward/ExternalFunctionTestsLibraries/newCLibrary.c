long long arraySum(void* allocatedPtr, void* alignedDataPtr, long offset, long dim0_size, long dim0_stride)
/**
 * It computes the sum between all element of the given array
 * @param alignedDataPtr the actual pointer to the array
 * @param dim0_size array size
 *
 * @return sum between each element of "alignedDataPtr"
 */
	{
	    long long* arr = (long long*) alignedDataPtr;
	    long long sum = 0;
	    
	    for (int i = 0; i < dim0_size; ++i)
		    {
		        sum += arr[i];
		    }
	    
	    return sum;
	}
int integerLog (int base, int val)
/**
 * It computes the integer logarithm
 * @param base logarithm base
 * @param val logarithm argument
 *
 * @return integer logarithm of "val" base "base" ("-1" if it is not computable)
 */
	{
		int cont = 0;

		if (base <= 1)
			return -1;

		while (val >= base)
			{
				val /= base;
				cont++;
			}

		return cont;
	}
int intAdder (int x, int y)
/**
 * It sums two "int" values
 * @param x first value to sum
 * @param y second value to sum
 * 
 * @return sum between "x" and "y"
 */
	{
		return x+y;
	}
double doubleAdder (double x, double y)
/**
 * It sums two "double" values
 * @param x first value to sum
 * @param y second value to sum
 * 
 * @return sum between "x" and "y"
 */
	{
		return x+y;
	}
double inverse (double x)
/**
 * It computes the inverse of a double value
 * @param x value to invert
 * 
 * @return inverse of "x" ("0.0" if it is not computable)
 */
	{
		if (x == 0.0)
			return 0.0;
		else
			return (1/x);
	}
double absolute (double x)
/**
 * It computes the absolute value of a double value
 * @param x the value
 * 
 * @return absolute value of "x"
 */
	{
		return (x>=0 ? x : -x);
	}
double max (double x, double y)
/**
 * It computes the maximum value of two double values
 * @param x the first value
 * @param y the second value
 * 
 * @return maximum between the two values
 */
	{
		return (x>=y ? x : y);
	}
double min (double x, double y)
/**
 * It computes the minimum value of two double values
 * @param x the first value
 * @param y the second value
 * 
 * @return minimum between the two values
 */
	{
		return (x<=y ? x : y);
	}
int factorial(int n)
/**
 * It computes factorial of a given number
 * @param n the given number
 * 
 * @return the factorial of the given number ("1" if the factorial is not computable)
 */
	{
		if (n <= 1)
			return 1;
		if (n == 2)
			return n;

		return n*factorial(n-1);
	}
int fib (int pos)
/**
 * It calculates a precise element of the Fibonacci serie
 * @param pos the position (starting from "1") of the request serie's value
 * 
 * @return the element of the Fibonacci serie at the requested position
 */
	{
		if (pos <= 0)
			return 0;

		if (pos == 1 || pos == 2)
			return 1;

		int res;
		int prev = 1;
		int prev_prev = 1;
		for (int i = 3 ; i <= pos ; i++)
			{
				res = prev + prev_prev;
				prev_prev = prev;
				prev = res;
			}

		return res;
	}
int logicXor (int v1, int v2)
/**
 * It calculates the XOR between two values with the common convention "false <=> == 0"
 * @param v1 the first logical value
 * @param v2 the second logical value
 * 
 * @return XOR between the two values
 */  
	{		
		if (v1 > 0)
			return (v2 == 0);
		else
			return (v2 > 0);
				
	}