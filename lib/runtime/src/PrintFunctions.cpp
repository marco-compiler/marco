#include <stdio.h>

extern "C"
{
	void modelicaPrint(char* name, float value)
	{
		printf("%s: %f\n", name, value);
	}

	void modelicaPrintFVector(char* name, float* value, int count)
	{
		printf("%s:\n", name);
		for (int a = 0; a < count; a++)
			printf("\t%f\n", value[a]);
	}

	void modelicaPrintBVector(char* name, char* value, int count)
	{
		printf("%s:\n", name);
		for (int a = 0; a < count; a++)
		{
			if (value[a])
				printf("\tTrue\n");
			else
				printf("\tFalse\n");
		}
	}

	void modelicaPrintIVector(char* name, int* value, int count)
	{
		printf("%s:\n", name);
		for (int a = 0; a < count; a++)
			printf("\t%d\n", value[a]);
	}
}
