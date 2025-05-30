#include<stdio.h>
int discreteLog (int base, int val)
/**
 *	input : base (>= 2), it is the logarithm base 
 * 			val (> 0), it is the value which we want to calculate the logarithm of
*/ 
	{
		int cont = 0;
		while (val > 1)
			{
				cont++;
				val = val/base;
			}	

		return (cont);
	}

double continuosLog (int base, int val)
/**
 *	input : base (>= 2), it is the logarithm base 
 * 			val (> 0), it is the value which we want to calculate the logarithm of
*/ 
	{
		double ret = val;
		int cont = 0;
		while (ret > 1)
			{
				cont++;
				ret = ret/(double)base;
			}	

		return (cont);
	}
int main ()
	{
		for (int i = 1 ; i < 100 ; i++)
			{
				printf("%lf\n", continuosLog(2, i));
			}
	}
