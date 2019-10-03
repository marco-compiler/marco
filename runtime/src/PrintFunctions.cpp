#include <stdio.h>

extern "C"
{
	void modelicaPrint(char* name, float value)
	{
		printf("%s: %f\n", name, value);
	}
}
