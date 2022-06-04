#include <iostream>
#include <stdarg.h>
#include <cstdint>
#include <math.h>

float power(const int n,const int p){
	return p == 0 ? 1 : n*power(n,p-1);
}



char* tochar(const float x, const int precision,char* p){
	//dovrebbe stampare 0.231454253242
	std::cout<< "X " << x << std::endl; //stampa .231454253197

	int a,b,c,k,l=0,m,i=0,j;
	double f = x ;
	int o = 0;
	
	// check for negetive float
	if(f<0.0)
	{
		
		p[i++]='-';
		f*=-1;
		o++;
	}
	
	a=f;	// extracting whole number
	f-=(float)a;	// extracting decimal part
	k = precision;
	
	// number of digits in whole number
	do
	{
		l = std::pow(10,k);
		m = a/l;
		if(m>0)
		{
			break;
		}
	k--;
	}while(k>-1);

	// number of digits in whole number are k+1
	
	for(; a>0; ++i)
   {	
      p[i] = a%10+'0';
      a/=10; 
   }
   
   //now we need to reverse res
   for(; o < i/2; ++o)
   {
       char c = p[o]; p[o] = p[i-o-1]; p[i-o-1] = c;
   }
	p[i++] = '.';
	

	for(l=0;l<precision;l++)
	{	std::cout<<"f before "<< f;
		f*=10.0;
		b = f ;//+ 5/std::pow(10,l + 5);
		p[i++]=b+48;
		std::cout<<" f middle "<< f;
		f-=(float)b;
		std::cout<<"f after "<< f <<std::endl;
		//std::cout<<"f "<< f <<" b "<< b <<std::endl;
	}
	if( f* 10 >= 5) p[i-1] = p[i-1] + 1;
	p[i]='\0';
  return p;
}



int main(int argc, char const *argv[])
{   char s[64] = {0};
	std::cout.precision(12);
    std::cout << tochar(0.231454253242,12,s) << std::endl;
    return 0;
}
